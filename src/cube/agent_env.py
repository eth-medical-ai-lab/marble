"""CUBE agent observations, action parsing, and feedback."""
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Optional, Any
from PIL import Image as _PILImage
from agent_loop import AgentEnv, AgentLoopState, StepOutcome
from cube.env import (
    CubeEnv,
    LETTERS,
    SolutionValidator,
    edges_to_rotation_flips,
    normalize_cube_solution,
    select_locked_placements,
)
from cube.prompts import (
    format_faces_ascii,
    build_cube_feedback_text,
    build_cube_initial_state_text,
    build_cube_invalid_orientation_text,
    build_cube_locked_face_text,
    build_cube_locked_piece_text,
    build_cube_parse_error_text,
    build_cube_prompt,
    build_cube_success_text,
)
from cube.rendering import render_faces_3d
from eval_utils import image_content_block_from_pil, strip_images_for_log
from cube.config import CubeEnvConfig


STEP_PATTERN = re.compile(
    r"\b([A-F])\s*:\s*\(\s*(Top|Bottom|Left|Right|Front|Back)\s*,\s*([1-4])\s*,\s*([1-4])\s*\)",
    re.IGNORECASE,
)


def extract_single_step(text: str) -> Optional[Tuple[str, str, int, int]]:
    """Extract the last single placement like ``E: (Right, 4, 3)`` from text."""
    from eval_utils import strip_thinking_blocks

    text = strip_thinking_blocks(text)
    last = None
    for match in STEP_PATTERN.finditer(text):
        last = match
    if not last:
        return None
    letter = last.group(1).upper()
    face = last.group(2).capitalize()
    left = int(last.group(3))
    top = int(last.group(4))
    return letter, face, left, top


def extract_all_steps(text: str) -> List[Tuple[str, str, int, int]]:
    """Extract all placement steps, keeping the last placement per piece."""
    from eval_utils import strip_thinking_blocks

    text = strip_thinking_blocks(text)
    seen: Dict[str, Tuple[str, str, int, int]] = {}
    for match in STEP_PATTERN.finditer(text):
        letter = match.group(1).upper()
        face = match.group(2).capitalize()
        left = int(match.group(3))
        top = int(match.group(4))
        seen[letter] = (letter, face, left, top)
    return list(seen.values())


def extract_happycube_solution(response):
    """
    Extracts the Happy Cube solution from the given response string.

    Args:
      response_string: The string containing the Happy Cube solution.

    Returns:
      A dictionary where keys are piece labels (A, B, C, D, E, F)
      and values are tuples containing the cube face, left edge number,
      and top edge number.  Returns an None if no solution is found.
    """
    from eval_utils import strip_thinking_blocks
    response = strip_thinking_blocks(response)
    # --- Step 1: grab the last matching code-block -------------
    code_blocks = re.findall(r"```([\s\S]*?)```", response)
    answer_block = next(
        (blk for blk in reversed(code_blocks)
         if all(f"{ch}:" in blk for ch in "ABCDEF")),
        None
    )
    if answer_block is None:
        answer_block = response

    answer_block = answer_block.replace('Left-Edge', '').replace('Top-Edge', '').replace('#', '')
    # --- Step 2: parse the A-F answer -------------
    pattern = re.compile(r'''
        ^\s*             # start of line, optional whitespace
        (?P<key>[A-Z])   # single uppercase letter key
        \s*:\s*          # colon surrounded by optional whitespace
        \(\s*            # opening parenthesis
        (?P<face>\w+)    # word for the face
        \s*,\s*          # comma separator
        (?:[A-Za-z\-\s]+)? # optional prefix
        (?P<x>\d+)       # first number
        \s*,\s*          # comma separator
        (?:[A-Za-z\-\s]+)? # optional prefix
        (?P<y>\d+)       # second number
        \s*\)            # closing parenthesis
    ''', re.MULTILINE | re.VERBOSE)

    result = {}
    for m in pattern.finditer(answer_block):
        key = m.group('key')
        face = m.group('face').lower()
        x = int(m.group('x'))
        y = int(m.group('y'))
        result[key] = (face, x, y)

    expected_keys = {'A', 'B', 'C', 'D', 'E', 'F'}
    actual_keys = set(result.keys())

    if actual_keys == expected_keys:
        return result
    else:
        return None


class CubeAgentEnv(AgentEnv):
    """AgentEnv wrapper for the Happy Cube task."""

    def __init__(
        self,
        env_cfg: CubeEnvConfig,
        sample: Dict[str, Any],
        sample_idx: int = 0,
        loop_mode: str = "multi_turn_conversation",
    ) -> None:
        super().__init__(loop_mode=loop_mode)
        self.env_cfg = env_cfg
        self.sample = sample
        self.sample_idx = sample_idx
        self.env: Optional[CubeEnv] = None
        self.actions_taken: List[str] = []
        self.success: bool = False
        self.correct: bool = False
        self.remaining_letters: List[str] = []
        self.initial_messages: List[Dict[str, Any]] = []

    @property
    def max_steps(self) -> int:
        return int(self.env_cfg.max_steps)

    def _face_arrays(self) -> Any:
        face_arrays = self.sample.get("faces")
        if face_arrays is None:
            face_arrays = self.sample.get("face_arrays")
        if face_arrays is None:
            raise KeyError("Cube sample is missing `faces`.")
        return face_arrays

    def _solution_map(self) -> Dict[str, Tuple[str, int, int]]:
        structured = normalize_cube_solution(self.sample.get("solution"))
        if structured:
            return structured

        ref_sol_text = str(self.sample.get("reference_solution", "") or "")
        parsed = extract_happycube_solution(ref_sol_text) or {}
        return {
            letter: (str(face).lower(), int(left_edge), int(top_edge))
            for letter, (face, left_edge, top_edge) in parsed.items()
        }

    def _sample_missing(self) -> int:
        if self.env_cfg.missing is not None and int(self.env_cfg.missing) >= 0:
            return int(self.env_cfg.missing)

        meta_info = self.sample.get("meta_info")
        if isinstance(meta_info, dict) and "missing" in meta_info:
            try:
                return int(meta_info["missing"])
            except (TypeError, ValueError):
                pass

        raise ValueError("Cube config must define missing or sample meta_info['missing'].")

    def _locked_placements(self) -> Dict[str, Tuple[str, int, int]]:
        explicit = normalize_cube_solution(self.sample.get("locked_placements"))
        expected_locked = len(LETTERS) - self._sample_missing()
        if explicit and len(explicit) == expected_locked:
            return explicit

        solution = self._solution_map()
        if not solution:
            return explicit

        return select_locked_placements(
            solution=solution,
            desired_missing=self._sample_missing(),
            seed=self.env_cfg.seed + self.sample_idx,
        )


    def _question_text(self) -> str:
        return build_cube_prompt(
            face_arrays=self._face_arrays(),
            locked_placements=self._locked_placements(),
            loop_mode=self.loop_mode(),
            symbolic=self.env_cfg.symbolic,
        )

    def _build_single_turn_messages(self) -> List[Dict[str, Any]]:
        face_arrays = self._face_arrays()
        question_text = self._question_text()
        content: List[Dict[str, Any]] = []

        if self.env_cfg.with_image:
            pieces_pil = self.sample.get("image")
            if pieces_pil is None:
                pieces_pil = render_faces_3d(
                    face_arrays,
                    titles=["A", "B", "C", "D", "E", "F"],
                    seed=21,
                    return_pil=True,
                    random_view=False,
                )
            if pieces_pil is not None:
                content.append(image_content_block_from_pil(pieces_pil))

        self.last_prompt_text = question_text
        content.append({"type": "text", "text": question_text})
        self.initial_messages = [{"role": "user", "content": content}]
        return list(self.initial_messages)

    def _build_feedback(
        self,
        action_desc: str,
        step_i: int,
        run_dir: Optional[str],
        evicted: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        del run_dir
        assert self.env is not None
        parts: List[Dict[str, Any]] = []

        render_flags = [flag for flag in ("net", "view", "views") if flag in self.env_cfg.feedback_flags]
        if render_flags:
            render_result = self.env.render(save=False, formats=render_flags)
            for flag in render_flags:
                arr = render_result.get(flag)
                if arr is not None:
                    parts.append(image_content_block_from_pil(_PILImage.fromarray(arr)))

        conflict_lines: List[str] = []
        if evicted:
            conflict_lines.append(f"Note: piece {evicted} was removed from that face to make room.")

        if self.env_cfg.with_conflicts:
            conflicts = self.env._check_conflicts()
            if conflicts:
                seen_conflicts = set()
                conflict_strs: List[str] = []
                for conflict in conflicts:
                    faces = tuple(str(face) for face in conflict.get("faces", ()))
                    pieces = tuple(str(piece) for piece in conflict.get("pieces", ()))
                    if not faces or not pieces:
                        continue

                    dedupe_key = (str(conflict.get("location", "seam")), tuple(sorted(faces)))
                    if dedupe_key in seen_conflicts:
                        continue
                    seen_conflicts.add(dedupe_key)

                    participants = ", ".join(
                        f"{piece} ({face})"
                        for piece, face in zip(pieces, faces)
                    )
                    if str(conflict.get("location", "")) == "corner":
                        corner_name = str(conflict.get("corner", "-".join(faces)))
                        conflict_strs.append(f"  - Corner {corner_name}: {participants}")
                    elif len(faces) >= 2 and len(pieces) >= 2:
                        conflict_strs.append(
                            f"  - {pieces[0]} ({faces[0]}) and {pieces[1]} ({faces[1]})"
                        )
                    else:
                        conflict_strs.append(f"  - {participants}")
                conflict_lines.append("Conflicts detected:")
                conflict_lines.extend(conflict_strs)
            else:
                num_placed = sum(1 for value in self.env._placements.values() if value is not None)
                if num_placed >= 2:
                    conflict_lines.append("No seam conflicts so far.")

        if "symbolic" in self.env_cfg.feedback_flags:
            symbolic_faces_text = format_faces_ascii(self.env.faces)
        else:
            symbolic_faces_text = None

        placed_strs: List[str] = []
        remaining: List[str] = []
        for letter, triple in self.env._placements.items():
            if triple is None:
                remaining.append(letter)
            else:
                face, left_edge, top_edge = triple
                placed_strs.append(f"{letter}: ({face.capitalize()}, {left_edge}, {top_edge})")
        sol = self.env._solution_dict()
        validator_line = None
        if sol is not None:
            ok, msg = self.env.validator.validate_with_text_feedback(self.env.faces, sol, binary=False)
            if ok:
                validator_line = "VALID SOLUTION! The cube is correctly assembled."
            else:
                validator_line = f"Validator feedback: {msg}"

        feedback_text = build_cube_feedback_text(
            action_desc=action_desc,
            step_idx=step_i,
            evicted=evicted,
            conflict_lines=conflict_lines,
            validator_line=validator_line,
            symbolic_faces_text=symbolic_faces_text,
            placed_summary=", ".join(placed_strs) if placed_strs else "(none)",
            remaining_summary=", ".join(remaining) if remaining else "(none)",
            remaining_budget=self.max_steps - step_i - 1,
            loop_mode=self.loop_mode(),
        )
        return self._build_followup_content(observation=parts, feedback_text=feedback_text)

    def reset(self, run_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        del run_dir
        self.actions_taken = []
        self.success = False
        self.correct = False

        if self.loop_mode() == "single_turn_offline":
            self.env = None
            return self._build_single_turn_messages()

        face_arrays = self._face_arrays()
        question_text = self._question_text()
        locked_placements = self._locked_placements()

        self.env = CubeEnv(faces=face_arrays, max_steps=self.max_steps, report_conflicts=True)
        self.env.reset()

        if locked_placements:
            self.env.apply_solution_mapping(locked_placements, clear=False)
            self.env._locked_faces = {str(face).capitalize() for face, _, _ in locked_placements.values()}

        self.remaining_letters = [letter for letter in "ABCDEF" if letter not in set(locked_placements)]

        initial_parts: List[Dict[str, Any]] = []
        pieces_pil = None
        if self.env_cfg.with_image:
            pieces_pil = render_faces_3d(
                face_arrays,
                titles=["A", "B", "C", "D", "E", "F"],
                seed=21,
                return_pil=True,
                random_view=False,
            )
        if self.env_cfg.with_image and pieces_pil is not None:
            initial_parts.append(image_content_block_from_pil(pieces_pil))

        if self.env_cfg.with_image and locked_placements:
            render_flags = [flag for flag in ("net", "view", "views") if flag in self.env_cfg.feedback_flags] or ["net"]
            render_result = self.env.render(save=False, formats=render_flags)
            for flag in render_flags:
                arr = render_result.get(flag)
                if arr is not None:
                    initial_parts.append(image_content_block_from_pil(_PILImage.fromarray(arr)))

        img_desc = {
            "net": "an updated 2D cube net",
            "view": "an updated 3D view of the cube",
            "views": "updated 3D views of the cube",
        }
        feedback_desc = [img_desc[flag] for flag in ("net", "view", "views") if flag in self.env_cfg.feedback_flags]
        if "symbolic" in self.env_cfg.feedback_flags:
            feedback_desc.append("the piece shapes as text grids")
        if self.env_cfg.with_conflicts:
            feedback_desc.append("a list of seam conflicts (if any)")
        feedback_str = ", ".join(feedback_desc) if feedback_desc else "a status update"

        initial_text = build_cube_initial_state_text(
            question_text,
            remaining_letters=self.remaining_letters,
            max_steps=self.max_steps,
            feedback_summary=feedback_str,
            with_conflicts=self.env_cfg.with_conflicts,
            has_locked_placements=bool(locked_placements),
        )
        self.last_prompt_text = initial_text
        initial_parts.append({"type": "text", "text": initial_text})

        self.initial_messages = [{"role": "user", "content": initial_parts}]
        return list(self.initial_messages)

    def step_multi_turn_conversation(
        self,
        response_text: str,
        step_idx: int,
        run_dir: Optional[str] = None,
    ) -> StepOutcome:
        assert self.env is not None

        parsed = extract_single_step(response_text or "")
        if parsed is None:
            remaining_budget = self.max_steps - step_idx - 1
            err_msg = build_cube_parse_error_text(remaining_budget, self.loop_mode())
            return self._build_text_step_outcome(
                err_msg,
                metadata={"status": "parse_error"},
            )

        letter, face, left_edge, top_edge = parsed
        action_desc = f"{letter}: ({face}, {left_edge}, {top_edge})"

        if letter.upper() not in [item.upper() for item in self.remaining_letters]:
            remaining_budget = self.max_steps - step_idx - 1
            err_msg = build_cube_locked_piece_text(
                letter,
                self.remaining_letters,
                remaining_budget,
                self.loop_mode(),
            )
            return self._build_text_step_outcome(
                err_msg,
                metadata={"status": "locked_piece", "piece": letter},
            )

        if (left_edge, top_edge) not in edges_to_rotation_flips:
            remaining_budget = self.max_steps - step_idx - 1
            err_msg = build_cube_invalid_orientation_text(
                action_desc,
                left_edge,
                top_edge,
                remaining_budget,
                self.loop_mode(),
            )
            return self._build_text_step_outcome(
                err_msg,
                metadata={"status": "invalid_orientation"},
            )

        face_lower = face.lower()
        locked_faces = {str(locked_face).lower() for locked_face in getattr(self.env, "_locked_faces", set())}
        if face_lower in locked_faces:
            locked_piece = None
            for other, placement in self.env._placements.items():
                if placement is not None and placement[0] == face_lower:
                    locked_piece = other
                    break
            remaining_budget = self.max_steps - step_idx - 1
            err_msg = build_cube_locked_face_text(
                action_desc,
                face,
                locked_piece,
                remaining_budget,
                self.loop_mode(),
            )
            return self._build_text_step_outcome(
                err_msg,
                metadata={"status": "locked_face", "face": face, "piece": locked_piece},
            )

        self.actions_taken.append(action_desc)

        evicted = None
        for other, placement in self.env._placements.items():
            if placement is not None and placement[0] == face_lower and other != letter.upper():
                evicted = other
                break

        self.env.apply_solution_mapping({letter: (face, left_edge, top_edge)}, clear=False)
        sol = self.env._solution_dict()
        if sol is not None:
            ok, _ = self.env.validator.validate_with_text_feedback(self.env.faces, sol, binary=False)
            if ok:
                self.success = True
                success_text = build_cube_success_text(action_desc, step_idx)
                return self._build_text_step_outcome(
                    success_text,
                    action_summary=action_desc,
                    metadata={"status": "solved"},
                    done=True,
                    success=True,
                )

        feedback_parts = self._build_feedback(action_desc, step_idx, run_dir, evicted=evicted)
        return self._build_step_outcome(
            content=feedback_parts,
            action_summary=action_desc,
            metadata={"status": "continue"},
        )

    def step_single_turn(
        self,
        response_text: str,
        reasoning_text: str = "",
        run_dir: Optional[str] = None,
    ) -> None:
        """Validate a full Cube solution from either answer text or reasoning.

        Some reasoning models put the final placement table in the reasoning
        field instead of the visible answer. We therefore try the visible answer
        first and only fall back to reasoning when needed.
        """
        del run_dir
        solution = extract_happycube_solution(response_text or "")
        if solution is None and reasoning_text:
            solution = extract_happycube_solution(reasoning_text)
        validator = SolutionValidator()
        self.correct = bool(validator.validate(self._face_arrays(), solution)) if solution is not None else False
        self.success = self.correct
        self.last_response_text = response_text or ""
        self.last_reasoning_text = reasoning_text or ""

    def build_result(
        self,
        loop_state: AgentLoopState,
        run_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        del run_dir
        if self.loop_mode() == "single_turn_offline":
            usage = None
            if loop_state.usage_log:
                usage = loop_state.usage_log[0].get("usage")
            return {
                "id": self.sample_idx,
                "face_arrays": self._face_arrays(),
                "messages": strip_images_for_log(self.initial_messages),
                "response": self.last_response_text,
                "finish_reason": self.last_finish_reason,
                "correct": self.correct,
                "success": self.success,
                "reasoning_content": self.last_reasoning_text,
                "usage": usage,
            }

        return {
            "id": self.sample_idx,
            "success": self.success,
            "actions_taken": list(self.actions_taken),
            "num_steps": len(self.actions_taken),
            "max_steps": self.max_steps,
            "usage": loop_state.usage_log,
        }
