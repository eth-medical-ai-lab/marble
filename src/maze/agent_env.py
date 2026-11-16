"""MAZE agent observations, action parsing, and feedback."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple
import re
import io
import numpy as np
from PIL import Image
from agent_loop import AgentEnv, AgentLoopState, StepOutcome
from eval_utils import encode_pil_image, image_content_block
from maze.env import decode_insertion, encode_insertion
from maze.constants import PIECE_NAMES
from maze.prompts import (
    board_to_tokens,
    extra_token,
    objects_to_grid,
    build_followup_prompt,
    build_instruction_prompt,
    build_invalid_env_action_feedback,
    build_invalid_llm_action_feedback,
)
from maze.config import MazeEnvConfig


STEP_INSERT_PATTERN = re.compile(
    r"(?:Step|Turn)\s+\d+:\s*INSERT\(side='(left|right|top|bottom)'\s*,\s*slot=(\d+)\s*,\s*rotation=(0|1|2|3)\)",
    flags=re.I,
)


STEP_MOVE_PATTERN = re.compile(
    r"(?:Step|Turn)\s+\d+:\s*MOVE\(\s*(?:x=(\d+)\s*,\s*y=(\d+)\s*)?\)",
    flags=re.I,
)


TURN_PATTERN = re.compile(
    r"Turn\s+\d+:\s*INSERT\(side='(left|right|top|bottom)'\s*,\s*slot=(\d+)\s*,\s*rotation=(0|1|2|3)\)\s*;\s*MOVE\(\s*(?:x=(\d+)\s*,\s*y=(\d+)\s*)?\)",
    flags=re.I,
)


@dataclass
class Insertion:
    side: str
    slot: int
    rotation: int


@dataclass
class Move:
    x: int
    y: int


@dataclass
class Step:
    """A single action step — either an INSERT or a MOVE."""
    kind: str  # "insert" or "move"
    insertion: Optional[Insertion] = None
    move: Optional[Move] = None


@dataclass
class Plan:
    steps: List[Step]

    @classmethod
    def from_text(cls, text: str, size: int) -> "Plan":
        from eval_utils import strip_thinking_blocks
        chosen_text = strip_thinking_blocks(text or "")

        code_blocks = re.findall(r"```([\s\S]*?)```", chosen_text, flags=re.M)
        for block in reversed(code_blocks):
            if "Step" in block or "Turn" in block:
                chosen_text = block
                break

        # First try: parse combined Turn format (legacy) and expand into individual steps
        combined_steps: List[Step] = []
        for m in TURN_PATTERN.finditer(chosen_text):
            side = m.group(1).lower()
            slot = int(m.group(2))
            rot = int(m.group(3))
            gx = m.group(4)
            gy = m.group(5)
            ins = Insertion(side=side, slot=slot, rotation=rot)
            combined_steps.append(Step(kind="insert", insertion=ins))
            if gx is None or gy is None:
                mv = Move(x=-1, y=-1)
            else:
                mv = Move(x=int(gx), y=int(gy))
            combined_steps.append(Step(kind="move", move=mv))

        if combined_steps:
            return cls(steps=combined_steps)

        # Second try: parse individual Step format
        # Collect all matches with their positions to maintain order
        all_matches = []
        for m in STEP_INSERT_PATTERN.finditer(chosen_text):
            side = m.group(1).lower()
            slot = int(m.group(2))
            rot = int(m.group(3))
            ins = Insertion(side=side, slot=slot, rotation=rot)
            all_matches.append((m.start(), Step(kind="insert", insertion=ins)))

        for m in STEP_MOVE_PATTERN.finditer(chosen_text):
            # Skip if this match is part of a combined Turn (INSERT...;MOVE)
            gx = m.group(1)
            gy = m.group(2)
            if gx is None or gy is None:
                mv = Move(x=-1, y=-1)
            else:
                mv = Move(x=int(gx), y=int(gy))
            all_matches.append((m.start(), Step(kind="move", move=mv)))

        # Sort by position in text to maintain order
        all_matches.sort(key=lambda x: x[0])
        steps = [s for _, s in all_matches]

        return cls(steps=steps)

    def to_pretty_string(self) -> str:
        lines: List[str] = []
        for idx, s in enumerate(self.steps or [], start=1):
            if s.kind == "insert" and s.insertion:
                line = f"Step {idx}: INSERT(side='{s.insertion.side}', slot={s.insertion.slot}, rotation={s.insertion.rotation})"
            elif s.kind == "move" and s.move:
                if s.move.x < 0 or s.move.y < 0:
                    line = f"Step {idx}: MOVE()"
                else:
                    line = f"Step {idx}: MOVE(x={s.move.x}, y={s.move.y})"
            else:
                line = f"Step {idx}: ???"
            lines.append(line)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_pretty_string()


def endpoint_idx(x: int, y: int, size: int) -> int:
    return y * size + x


class MazeAgentEnv(AgentEnv):
    """AgentEnv wrapper for the Labyrinth multi-step task."""

    def __init__(
        self,
        env_cfg: MazeEnvConfig,
        env: Any,
        max_steps: int,
        info: Optional[Dict[str, Any]] = None,
        loop_mode: str = "multi_turn_conversation",
    ) -> None:
        super().__init__(loop_mode=loop_mode)
        self.env_cfg = env_cfg
        self.env = env
        self._max_steps = int(max_steps)
        self._initial_info = info
        self.info: Optional[Dict[str, Any]] = None
        self.success: bool = False
        self.executed_actions: List[Dict[str, Any]] = []
        self.offline_plan: List[Dict[str, Any]] = []
        self.offline_num_turns: Optional[int] = None
        self.offline_response: str = ""

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def _build_observation_text(self) -> str:
        board = self.env._board
        tokens = board_to_tokens(board)
        objects = objects_to_grid(board)
        pawn = self.env._gs.players.get("P1")
        target_lab = self.env._target_label
        last_insert = self.env._last_lane
        extra = extra_token(board)

        text_lines = [
            "STATE",
            f"Last Insert: {last_insert}",
        ]
        target_name = PIECE_NAMES.get(target_lab, target_lab)
        if self.env_cfg.include_symbolic_obs:
            text_lines.extend(
                [
                    f"Board grid: {tokens}",
                    f"Objects grid: {objects}",
                    f"Target: {target_name} - {target_lab}",
                    f"Player Position: P1 at {pawn}",
                    f"Extra tile: {extra}",
                ]
            )
        else:
            text_lines.append(f"Target: {target_name} - {target_lab}")
        return "\n".join(text_lines)

    def _rgb_image_b64(self) -> Optional[str]:
        prev_mode = self.env.render_mode
        self.env.render_mode = "rgb_array"
        try:
            rendered = self.env.render()
        finally:
            self.env.render_mode = prev_mode
        if rendered is None:
            return None
        if isinstance(rendered, str):
            # Some environments fall back to raw SVG when PNG conversion deps are
            # unavailable inside env.render(). Convert it here instead of
            # crashing on np.uint8("<svg ...>").
            try:
                import cairosvg
            except Exception:
                return None
            png_bytes = cairosvg.svg2png(bytestring=rendered.encode("utf-8"))
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            return encode_pil_image(img)
        if isinstance(rendered, Image.Image):
            return encode_pil_image(rendered.convert("RGB"))
        img = Image.fromarray(np.uint8(rendered), mode="RGB")
        return encode_pil_image(img)

    def _prompt_mode(self) -> str:
        if self.loop_mode() == "single_turn_offline":
            return "single_shot"
        return "multi_turn"

    def _build_observation_content(
        self,
        include_instructions: bool,
        remaining_budget: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        b64 = self._rgb_image_b64() if self.env_cfg.with_image else None
        instruction_prompt = build_instruction_prompt(
            self.env.size,
            self.env_cfg.include_symbolic_obs,
            bool(b64),
            mode=self._prompt_mode(),
        )

        obs_text = self._build_observation_text()
        content: List[Dict[str, Any]] = []
        if b64:
            content.append(image_content_block(b64))

        if include_instructions:
            combined_text = instruction_prompt + "\n\n" + obs_text
        else:
            combined_text = build_followup_prompt(
                obs_text,
                self.env.size,
                mode=self._prompt_mode(),
                remaining_budget=remaining_budget,
            )
        content.append({"type": "text", "text": combined_text})
        return content

    def _parse_action(self, text: str) -> Tuple[str, Dict[str, int]]:
        from eval_utils import strip_thinking_blocks

        text = strip_thinking_blocks(text)
        for match in reversed(
            list(
                re.finditer(
                    r"Action:\s*insert\(side='(left|right|top|bottom)'\s*,\s*slot=(1|3|5)\s*,\s*rotation=(0|1|2|3)\)",
                    text,
                    re.I,
                )
            )
        ):
            return "insert", {
                "side": match.group(1).lower(),
                "slot": int(match.group(2)),
                "rotation": int(match.group(3)),
            }
        for match in reversed(list(re.finditer(r"Action:\s*move\(\s*\)", text, re.I))):
            return "move", {}
        for match in reversed(list(re.finditer(r"Action:\s*move\(x=(\d+),\s*y=(\d+)\)", text, re.I))):
            return "move", {"x": int(match.group(1)), "y": int(match.group(2))}
        return "", {}

    def _validate_action(self, kind: str, params: Dict[str, int]) -> Tuple[Optional[List[int]], Dict[str, Any]]:
        assert self.info is not None
        size = self.env.size
        meta: Dict[str, Any] = {"llm_legal": True, "reason": "", "proposed": {"kind": kind, "params": params}}

        if kind == "insert":
            mask = self.info.get("insertion_mask")
            legal = [i for i, ok in enumerate(mask if mask is not None else []) if bool(ok)]
            try:
                ins_id = encode_insertion(params["side"], params["slot"], params["rotation"])
            except Exception:
                meta["llm_legal"] = False
                meta["reason"] = "parse_error_insert"
                return [0, legal[0] if legal else 0], meta
            if ins_id not in legal:
                meta["llm_legal"] = False
                meta["reason"] = "illegal_insertion_masked"
                return [0, legal[0] if legal else 0], meta
            return [0, ins_id], meta

        if kind == "move":
            mask = self.info.get("movement_mask")
            legal = [i for i, ok in enumerate(mask if mask is not None else []) if bool(ok)]
            if not params or "x" not in params or "y" not in params:
                return [1, size * size], meta
            try:
                mv = endpoint_idx(params["x"], params["y"], size)
            except Exception:
                meta["llm_legal"] = False
                meta["reason"] = "parse_error_move"
                return [1, legal[-1] if legal else size * size], meta
            if mv not in legal:
                meta["llm_legal"] = False
                meta["reason"] = "illegal_move_masked"
                return [1, legal[-1] if legal else size * size], meta
            return [1, mv], meta

        meta["llm_legal"] = False
        meta["reason"] = "unrecognized_action"
        return [0, 0], meta

    def _describe_action(self, action: List[int]) -> str:
        size = self.env.size
        action_type = int(action[0]) if action else 0
        param = int(action[1]) if len(action) >= 2 else 0
        if action_type == 0:
            side, slot, rot = decode_insertion(param)
            return f"INSERT(side={side}, slot={slot}, r={rot})"
        if param == size * size:
            px, py = self.env._gs.players.get("P1", (0, 0))
            return f"MOVE(stay at x={px}, y={py})"
        return f"MOVE(x={param % size}, y={param // size})"

    def _plan_to_list(self, plan_obj: Plan) -> List[Dict[str, Any]]:
        """Convert parsed plan steps into the compact result format we log.

        The offline Maze result stores one entry per insert/move pair. A trailing
        insert without an explicit move is preserved with a sentinel stay action
        so we do not silently lose the final parsed step.
        """
        plan_list: List[Dict[str, Any]] = []
        pending_insert: Optional[Dict[str, Any]] = None

        for step in plan_obj.steps:
            if step.kind == "insert" and step.insertion:
                if pending_insert is not None:
                    plan_list.append(
                        {
                            "insert": pending_insert,
                            "move": {"x": -1, "y": -1},
                        }
                    )
                pending_insert = {
                    "side": step.insertion.side,
                    "slot": step.insertion.slot,
                    "rotation": step.insertion.rotation,
                }
                continue

            if step.kind == "move" and step.move and pending_insert is not None:
                plan_list.append(
                    {
                        "insert": pending_insert,
                        "move": {"x": step.move.x, "y": step.move.y},
                    }
                )
                pending_insert = None

        if pending_insert is not None:
            plan_list.append(
                {
                    "insert": pending_insert,
                    "move": {"x": -1, "y": -1},
                }
            )

        return plan_list

    def reset(self, run_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        self.success = False
        self.executed_actions = []
        self.offline_plan = []
        self.offline_num_turns = None
        self.offline_response = ""
        if self._initial_info is None:
            _, self.info = self.env.reset()
        else:
            self.info = dict(self._initial_info)
        self.last_prompt_text = ""
        self.last_response_text = ""
        self.last_reasoning_text = ""
        self.last_usage = None
        content = self._build_observation_content(include_instructions=True)
        self.last_prompt_text = self._join_text_blocks(content)
        return [{"role": "user", "content": content}]


    def step_multi_turn_conversation(
        self,
        response_text: str,
        step_idx: int,
        run_dir: Optional[str] = None,
    ) -> StepOutcome:
        assert self.info is not None

        kind, params = self._parse_action(response_text)
        action, meta = self._validate_action(kind, params)
        if not meta.get("llm_legal", True):
            reason = meta.get("reason", "unknown")
            remaining_budget = max(self.max_steps - step_idx - 1, 0)
            feedback = build_invalid_llm_action_feedback(
                kind,
                reason,
                remaining_budget=remaining_budget,
            )
            return self._build_text_step_outcome(
                feedback,
                metadata={"status": "invalid_llm_action", "reason": reason},
            )

        action_type = int(action[0]) if action else 0
        _, reward, terminated, truncated, info = self.env.step(action)
        self.info = info

        is_invalid = info.get("invalid", False)
        movement_info = info.get("movement", {})
        is_invalid_movement = movement_info.get("invalid_movement", False) if isinstance(movement_info, dict) else False
        if is_invalid or is_invalid_movement:
            remaining_budget = max(self.max_steps - step_idx - 1, 0)
            feedback = build_invalid_env_action_feedback(
                "insert" if is_invalid else "move",
                remaining_budget=remaining_budget,
            )
            return self._build_text_step_outcome(
                feedback,
                metadata={"status": "env_invalid_action"},
            )

        action_desc = self._describe_action(action)
        if action_type == 0:
            ins = info.get("insertion", {})
            self.executed_actions.append(
                {
                    "action_type": "insert",
                    "step": step_idx,
                    "insert": {
                        "side": ins.get("side"),
                        "slot": ins.get("idx"),
                        "rotation": ins.get("rot", 0),
                    }
                    if ins
                    else {},
                }
            )
        else:
            mv = info.get("movement", {})
            end = mv.get("end") if isinstance(mv, dict) else None
            self.executed_actions.append(
                {
                    "action_type": "move",
                    "step": step_idx,
                    "move": {"x": end[0], "y": end[1]} if end else None,
                }
            )

        self.success = bool(terminated)
        remaining_budget = None if (terminated or truncated) else max(self.max_steps - step_idx - 1, 0)
        observation_content = self._build_observation_content(
            include_instructions=False,
            remaining_budget=remaining_budget,
        )
        terminal_text = None
        if self.success:
            terminal_text = "Target reached. Episode complete."
        elif truncated:
            terminal_text = "Episode ended without reaching the target."
        feedback_content = self._build_followup_content(
            observation=observation_content,
            terminal_text=terminal_text,
        )
        return self._build_step_outcome(
            content=feedback_content,
            action_summary=action_desc,
            metadata={"reward": reward, "terminated": terminated, "truncated": truncated},
            done=bool(terminated or truncated),
            success=self.success,
        )

    def step_single_turn(
        self,
        response_text: str,
        reasoning_text: str = "",
        run_dir: Optional[str] = None,
    ) -> None:
        """Parse and execute an offline Maze plan inside the env.

        This keeps single-shot evaluation consistent with multi-turn evaluation:
        success is determined by replaying the parsed plan against the real env,
        not just by checking whether the text matched the plan grammar.
        """
        self.last_response_text = response_text or ""
        self.last_reasoning_text = reasoning_text or ""
        self.offline_response = self.last_response_text

        parsed = Plan.from_text(self.last_response_text, self.env.size)
        if not parsed.steps and reasoning_text:
            alt = Plan.from_text(reasoning_text, self.env.size)
            if alt.steps:
                parsed = alt

        self.offline_plan = self._plan_to_list(parsed)
        self.offline_num_turns = len(self.offline_plan)

        self.success = False
        for step_idx, step in enumerate(parsed.steps):
            if step_idx >= self.max_steps:
                break

            kind = ""
            params: Dict[str, int] = {}
            if step.kind == "insert" and step.insertion:
                kind = "insert"
                params = {
                    "side": step.insertion.side,
                    "slot": step.insertion.slot,
                    "rotation": step.insertion.rotation,
                }
            elif step.kind == "move":
                kind = "move"
                # The plan parser represents MOVE() with (-1, -1); empty params mean stay.
                if step.move is not None and (step.move.x, step.move.y) != (-1, -1):
                    params = {"x": step.move.x, "y": step.move.y}

            if not kind:
                continue

            action, meta = self._validate_action(kind, params)
            if not meta.get("llm_legal", True):
                return

            action_type = int(action[0]) if action else 0
            _, _, terminated, truncated, info = self.env.step(action)
            self.info = info

            if terminated:
                self.success = True
                return
            if truncated or info.get("invalid") or info.get("movement", {}).get("invalid_movement", False):
                return

    def build_result(
        self,
        loop_state: AgentLoopState,
        run_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.loop_mode() == "single_turn_offline":
            usage = None
            if loop_state.usage_log:
                usage = loop_state.usage_log[0].get("usage")
            return {
                "success": self.success,
                "num_steps": self.offline_num_turns or 0,
                "num_turns": self.offline_num_turns,
                "usage": usage,
                "plan": list(self.offline_plan),
                "response": self.offline_response,
                "finish_reason": self.last_finish_reason,
            }

        return {
            "success": self.success,
            "num_steps": len(loop_state.turns),
            "usage": loop_state.usage_log,
            "plan": list(self.executed_actions),
        }
