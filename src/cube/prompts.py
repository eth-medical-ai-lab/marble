"""CUBE instructions, feedback, and prompt formatting."""
from __future__ import annotations
from cube.env import (
    LETTERS,
    _letter_to_index,
    edges_to_rotation_flips,
    is_mirror_symmetric,
    modulate_flips_for_allowed_letters_with_solution_adjustment,
    modulate_required_flips,
    modulate_required_flips_restricted,
    rotation_flips_to_edges,
    toggle_solution_flip_for_letters,
)
from cube.rendering import render_faces_3d
import numpy as np
import re
import random
from typing import List, Tuple, Dict, Optional


def format_faces_ascii(faces_arr, fmt: str = "numeric") -> str:
    """Format piece faces as an ASCII grid."""
    faces_arr = np.asarray(faces_arr)
    lines: list[str] = []
    try:
        n = int(faces_arr.shape[1])
    except Exception:
        return ""
    for idx, letter in enumerate(LETTERS):
        if idx >= int(faces_arr.shape[0]):
            break
        lines.append(f"{letter}:")
        grid = faces_arr[idx]
        for row in range(n):
            row_chars = []
            for col in range(n):
                val = int(grid[row, col])
                if fmt == "numeric":
                    row_chars.append(str(val))
                else:
                    row_chars.append("\u2588" if val == 1 else "\u00b7")
            lines.append("".join(row_chars))
        lines.append("")
    return "\n".join(lines).strip()


_PIECE_LINE_RE = re.compile(r"^\s*([A-F])\s*:\s*(.*)\s*$", re.MULTILINE)


def _parse_piece_map_from_text(text: str) -> Dict[str, str]:
    """
    Parse lines like 'A: (Face, x, y)' into a mapping letter→right-hand side string.
    Returns entries for any of A..F that are present.
    """
    result: Dict[str, str] = {}
    for m in _PIECE_LINE_RE.finditer(text or ""):
        letter = m.group(1)
        rhs = m.group(2).strip()
        result[letter] = rhs
    return result


def _find_code_blocks(text: str) -> List[Tuple[int, int, str]]:
    """
    Find triple-backtick code blocks, returning list of (start_idx, end_idx, content).
    """
    blocks: List[Tuple[int, int, str]] = []
    for m in re.finditer(r"```([\s\S]*?)```", text or "", re.MULTILINE):
        blocks.append((m.start(), m.end(), m.group(1)))
    return blocks


def _select_partial_block(prompt_text: str, answer_anchor: Optional[int]) -> Optional[Tuple[int, int, str]]:
    """
    Heuristic: choose the last code block before 'Answer:' if present; otherwise the last
    code block that contains A..F lines; otherwise None.
    """
    blocks = _find_code_blocks(prompt_text or "")
    if not blocks:
        return None
    # Prefer blocks before the Answer anchor
    candidates = blocks
    if answer_anchor is not None and answer_anchor >= 0:
        candidates = [b for b in blocks if b[0] < answer_anchor]
        if not candidates:
            candidates = blocks
    # Choose the last block that looks like piece lines
    for start, end, content in reversed(candidates):
        if _parse_piece_map_from_text(content):
            return (start, end, content)
    # Fallback: last block
    return blocks[-1]


def build_partial_block(ref_sol_text: str, desired_missing: int, seed: int = 20) -> str:
    """Build a 'To help you...' partial solution block (cube_easy format).

    Given a reference solution text (``A: (Face, le, te)`` lines) and the
    number of pieces to leave as ``...``, return the partial-solution
    text block that gets appended to the prompt.
    """
    full_map = _parse_piece_map_from_text(ref_sol_text)
    all_letters = ['A', 'B', 'C', 'D', 'E', 'F']
    rng = random.Random(seed)
    target = max(0, min(6, desired_missing))
    shuffled = list(all_letters)
    rng.shuffle(shuffled)
    missing = set(shuffled[:target])
    lines = [f"{l}: ..." if l in missing else f"{l}: {full_map.get(l, '...')}"
             for l in all_letters]
    return (
        "To help you to solve the problem, there is a partial solution "
        "below and you'll just need to fill the rest part to finish the answer:\n"
        "```\n" + "\n".join(lines) + "\n```"
    )


def modulate_prompt_for_symbolic(prompt: str) -> str:
    """Replace image-based edge descriptions with array-index wording.

    Used when ``--symbolic`` is set so the ``cube`` subset prompt reads
    like ``cube_easy`` (edges described as left/top/right/bottom in the
    2D array rather than 'as shown in the image').
    """
    prompt = prompt.replace(
        "(Edge1, Edge2, Edge3 and Edge4) as shown in the image",
        "(Edge1 = left, Edge2 = top, Edge3 = right, Edge4 = bottom in the 2D array)",
    )
    prompt = prompt.replace(
        "Six jigsaw-style pieces, labelled",
        "Six jigsaw-style pieces represented in 2D array, labelled",
    )
    return prompt


def modulate_prompt_disable_flips(prompt: str) -> str:
    """Replace flip-allowed wording with rotation-only.

    Used when ``--flips 0`` is set so the prompt tells the model that
    flipping is not allowed.
    """
    return prompt.replace(
        "proper orientation (rotation and/or flip)",
        "proper rotation (flip is not allowed)",
    )


def modulate_missing_pieces_in_prompt(prompt_text: str,
                                      full_solution_text: str,
                                      desired_missing: int,
                                      seed: Optional[int] = None) -> Tuple[str, str]:
    """
    Rewrites the partial A..F code block to target exactly 'desired_missing' occurrences of '...'.
    If full solution entries are unavailable, we only increase missing (never reduce).
    """
    text = prompt_text or ""
    ans_anchor = text.find("Answer:")
    block = _select_partial_block(text, ans_anchor if ans_anchor != -1 else None)
    if block is None:
        # No existing partial block: synthesize one and insert before "The final solution is"
        rng = random.Random(seed)
        all_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        target = max(0, min(6, int(desired_missing)))

        # Build canonical maps for A..F from the full solution text
        full_map = _parse_piece_map_from_text(full_solution_text or "")
        # Fallback: extract from anywhere in prompt if needed
        if len(full_map) < 6:
            full_map.update(_parse_piece_map_from_text(text))
        # Ensure defaults
        for l in all_letters:
            if l not in full_map:
                full_map[l] = "(Front, 1, 2)"

        # Choose which letters will be missing
        letters = list(all_letters)
        rng.shuffle(letters)
        missing = set(letters[:target])

        # Build new partial block
        new_lines: List[str] = []
        for l in all_letters:
            if l in missing:
                new_lines.append(f"{l}: ...")
            else:
                new_lines.append(f"{l}: {full_map[l]}")
        new_block_text = (
            "\nTo help you to solve the problem, there is a partial solution below and you'll just need to fill the rest part to finish the answer:\n"
            "```\n" + "\n".join(new_lines) + "\n```\n"
        )

        # Insert before "The final solution is" if present, else append
        insert_anchor = text.find("The final solution is")
        if insert_anchor != -1:
            new_prompt = text[:insert_anchor] + new_block_text + text[insert_anchor:]
        else:
            new_prompt = (text.rstrip() + "\n\n" + new_block_text)
        return new_prompt, full_solution_text

    start_idx, end_idx, content = block

    # Build canonical maps for A..F from the full solution (prefer after 'Answer:')
    full_map = _parse_piece_map_from_text(full_solution_text or "")
    if len(full_map) < 6:
        # Prefer 'Answer:' section if available
        ans_pos = text.find("Answer:")
        if ans_pos != -1:
            ans_map = _parse_piece_map_from_text(text[ans_pos:])
            if ans_map:
                full_map.update(ans_map)
        # Fallback: anywhere in prompt
        if len(full_map) < 6:
            full_map.update(_parse_piece_map_from_text(text))

    all_letters = ['A', 'B', 'C', 'D', 'E', 'F']
    partial_map = _parse_piece_map_from_text(content)
    current_missing_letters = [l for l in all_letters if partial_map.get(l, "...") == "..."]
    current_filled_letters  = [l for l in all_letters if l not in current_missing_letters]

    rng = random.Random(seed)
    target = max(0, min(6, int(desired_missing)))
    current_missing = len(current_missing_letters)

    if target == current_missing:
        final_missing = set(current_missing_letters)
    elif target > current_missing:
        need = target - current_missing
        choices = list(current_filled_letters)
        rng.shuffle(choices)
        to_hide = set(choices[:need])
        final_missing = set(current_missing_letters) | to_hide
    else:
        need_to_fill = current_missing - target
        fillable = [l for l in current_missing_letters if l in full_map]
        rng.shuffle(fillable)
        actually_fill = set(fillable[:need_to_fill])
        final_missing = set(current_missing_letters) - actually_fill

    # Rebuild block
    new_lines: List[str] = []
    for l in all_letters:
        if l in final_missing:
            new_lines.append(f"{l}: ...")
        else:
            if l in full_map:
                new_lines.append(f"{l}: {full_map[l]}")
            else:
                rhs = partial_map.get(l, '...')
                if rhs == '...':
                    rhs = "(Front, 1, 2)"
                new_lines.append(f"{l}: {rhs}")

    new_block_text = "```\n" + "\n".join(new_lines) + "\n```"
    # Replace original fenced block (start_idx..end_idx) with new one
    new_prompt = text[:start_idx] + new_block_text + text[end_idx:]
    return new_prompt, full_solution_text


def _proposition_letters_from_prompt(prompt_text: str) -> List[str]:
    """
    Parse the partial A..F block in the prompt and return letters that are still proposals,
    i.e., lines with '...'.
    """
    text = prompt_text or ""
    ans_anchor = text.find("Answer:")
    sel = _select_partial_block(text, ans_anchor if ans_anchor != -1 else None)
    if sel is None:
        return []
    _, _, content = sel
    partial_map = _parse_piece_map_from_text(content)
    all_letters = ['A', 'B', 'C', 'D', 'E', 'F']
    return [l for l in all_letters if partial_map.get(l, "...") == "..."]


def modulate_flips_from_prompt(face_arrays: List[np.ndarray],
                               solution: Dict[str, Tuple[str, int, int]],
                               prompt_text: str,
                               desired_total_flips: int,
                               seed: Optional[int] = None) -> Tuple[List[np.ndarray], Dict[str, List[str]]]:
    """
    Convenience wrapper to modulate flips only on proposition pieces inferred from the prompt.
    Proposition pieces are those with '...' in the partial solution block.
    """
    allowed = _proposition_letters_from_prompt(prompt_text)
    # If no partial solution exists in the prompt, fall back to allowing all pieces
    if not allowed:
        allowed = ['A', 'B', 'C', 'D', 'E', 'F']
    return modulate_required_flips_restricted(
        face_arrays=face_arrays,
        solution=solution,
        desired_total_flips=desired_total_flips,
        allowed_letters=allowed,
        seed=seed,
    )


def modulate_flips_from_prompt_with_solution_adjustment(
    face_arrays: List[np.ndarray],
    solution: Dict[str, Tuple[str, int, int]],
    prompt_text: str,
    desired_total_flips: int,
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], Dict[str, Tuple[str, int, int]], Dict[str, List[str]]]:
    """
    Modulate flips only on proposition pieces inferred from the prompt, but if desired flips is 0
    and a chosen piece is mirror-symmetric (so mirroring won't change effective parity), adjust the
    solution for that piece by toggling its flip bit (via edge mapping) to ensure flip is not useless.
    Returns: (new_face_arrays, new_solution, info)
    """
    allowed = _proposition_letters_from_prompt(prompt_text)
    if not allowed:
        allowed = list(LETTERS)
    return modulate_flips_for_allowed_letters_with_solution_adjustment(
        face_arrays=face_arrays,
        solution=solution,
        allowed_letters=allowed,
        desired_total_flips=desired_total_flips,
        seed=seed,
    )


def modulate_flips_and_render(
    face_arrays: List[np.ndarray],
    solution: Dict[str, Tuple[str, int, int]],
    prompt_text: str,
    desired_total_flips: int,
    seed: Optional[int] = None,
    *,
    render: bool = True,
    save_path: Optional[str] = None,
    return_pil: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Tuple[str, int, int]], Dict[str, List[str]], Optional["Image.Image"]]:
    """
    High-level helper:
    - Modulates flips only on propositions (per prompt) with symmetry-aware solution adjustment.
    - Optionally renders the post-modulation faces in 3D notebook style and returns a PIL image.
    Returns: (new_face_arrays, new_solution, info, image_or_None)
      info contains keys: 'mirrored', 'solution_toggled'
    """
    new_arrays, new_solution, info = modulate_flips_from_prompt_with_solution_adjustment(
        face_arrays=face_arrays,
        solution=solution,
        prompt_text=prompt_text,
        desired_total_flips=desired_total_flips,
        seed=seed,
    )

    pil_img = None

    if render:
        pil_img = render_faces_3d(
            new_arrays,
            titles=['A','B','C','D','E','F'],
            save_path=save_path,
            return_pil=return_pil,
            seed=(seed if seed is not None else 21),
        )
        
    return new_arrays, new_solution, info, pil_img


def modulate_flips_global_and_render(
    face_arrays: List[np.ndarray],
    solution: Dict[str, Tuple[str, int, int]],
    desired_total_flips: int,
    seed: Optional[int] = None,
    *,
    render: bool = True,
    save_path: Optional[str] = None,
    return_pil: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Tuple[str, int, int]], Dict[str, List[str]], Optional["Image.Image"]]:
    """
    Simplified global flip modulation that ignores partial-solution propositions:
    - Mirrors a subset of pieces across all A..F to reach desired_total_flips under the given solution.
    - Adjusts the solution for mirrored letters to preserve cube placement consistency.
    - Optionally renders the faces and returns a PIL image.
    Returns: (new_face_arrays, new_solution, info, image_or_None)
      info contains keys: 'mirrored', 'solution_toggled'
    """
    rng = random.Random(seed)
    # Normalize to numpy arrays for reliable flipping
    face_arrays_np = [np.array(a) if not isinstance(a, np.ndarray) else a for a in face_arrays]
    # Pick mirrors globally
    new_arrays, info_m = modulate_required_flips(
        face_arrays=face_arrays_np,
        solution=solution,
        desired_total_flips=desired_total_flips,
        seed=seed,
    )
    mirrored_letters = info_m.get('mirrored', []) or []
    info: Dict[str, List[str]] = {'mirrored': list(mirrored_letters), 'solution_toggled': []}
    new_solution = dict(solution)
    if mirrored_letters:
        new_solution = toggle_solution_flip_for_letters(solution, mirrored_letters)
        info['solution_toggled'].extend(mirrored_letters)

    pil_img = None
    if render:
        pil_img = render_faces_3d(
            new_arrays,
            titles=['A','B','C','D','E','F'],
            save_path=save_path,
            return_pil=return_pil,
            seed=(seed if seed is not None else 21),
        )
    return new_arrays, new_solution, info, pil_img


def modulate_partial_solution_flip_requirements(
    prompt_text: str,
    face_arrays: List[np.ndarray],
    desired_total_flips: int,
    seed: Optional[int] = None,
) -> str:
    """
    Rewrite ONLY the partial solution block in the prompt to reach the desired number of
    flip-requiring entries among placed pieces, choosing orientations for '...' (propositions)
    while respecting mirror symmetry: symmetric faces cannot contribute a useful flip.
    - We do NOT touch already-placed entries; we only fill '...'.
    - For flip=True entries, we assign an orientation whose (left, top) maps to a flip bit True.
    - For flip=False entries, we assign orientation mapping to flip bit False.
    - Faces that are mirror symmetric are skipped for flip=True assignments.
    """
    text = prompt_text or ""
    ans_anchor = text.find("Answer:")
    sel = _select_partial_block(text, ans_anchor if ans_anchor != -1 else None)
    if sel is None:
        return prompt_text
    start_idx, end_idx, content = sel

    # Parse current partial block
    partial_map = _parse_piece_map_from_text(content)
    letters = ['A','B','C','D','E','F']
    rng = random.Random(seed)

    # Helper to compute flip flag from a RHS like "(Face, L, T)"
    def rhs_to_flip(rhs: str) -> Optional[bool]:
        m = re.search(r"\(\s*([A-Za-z]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", rhs or "")
        if not m:
            return None
        try:
            le = int(m.group(2)); te = int(m.group(3))
            _, flip = edges_to_rotation_flips[(le, te)]
            return bool(flip)
        except Exception:
            return None

    # Count current flips among placed entries
    current_flips = 0
    for L in letters:
        rhs = partial_map.get(L, None)
        if rhs and rhs.strip() != "...":
            f = rhs_to_flip(rhs)
            if f is True:
                current_flips += 1

    target = max(0, min(6, int(desired_total_flips)))
    if target == current_flips:
        return prompt_text

    # Build assignment pools for propositions
    props = [L for L in letters if partial_map.get(L, "...") == "..."]
    # Face symmetry map
    is_sym = {}
    for L in letters:
        try:
            idx = _letter_to_index(L)
            is_sym[L] = is_mirror_symmetric(face_arrays[idx])
        except Exception:
            is_sym[L] = False

    # Orientation constructor from desired flip bit
    # Use rot_k=0 for simplicity: flip False -> (1,2); flip True -> (3,2)
    def make_rhs(face_name: str, want_flip: bool) -> str:
        rot_k = 0
        le, te = rotation_flips_to_edges[(rot_k, want_flip)]
        return f"({face_name}, {le}, {te})"

    # Choose a default face name; "Front" is fine for text (does not impact evaluation of flips)
    def_face = "Front"

    new_lines: List[str] = []
    # Decide how many to add/remove among propositions
    need = target - current_flips
    if need > 0:
        # Need more flip=True: fill some '...' with flip=True on non-symmetric faces
        candidates = [L for L in props if not is_sym.get(L, False)]
        rng.shuffle(candidates)
        to_set_true = candidates[:need]
        for L in letters:
            rhs = partial_map.get(L, "...")
            if rhs == "...":
                if L in to_set_true:
                    rhs = make_rhs(def_face, True)
                else:
                    # leave as '...' for now; will fill neutral (flip=False) to keep text valid
                    rhs = "..."
            new_lines.append(f"{L}: {rhs}")
    else:
        # Need fewer flips: we cannot modify already placed entries, so for propositions
        # fill them with flip=False orientations
        for L in letters:
            rhs = partial_map.get(L, "...")
            if rhs == "...":
                rhs = make_rhs(def_face, False)
            new_lines.append(f"{L}: {rhs}")

    # If still remaining '...' after trying to set flip True, fill neutrals (flip=False)
    if any(line.strip().endswith(": ...") for line in new_lines):
        fixed_lines = []
        for line in new_lines:
            if line.strip().endswith(": ..."):
                L = line.split(":")[0].strip()
                rhs = make_rhs(def_face, False)
                fixed_lines.append(f"{L}: {rhs}")
            else:
                fixed_lines.append(line)
        new_lines = fixed_lines

    new_block = "```\n" + "\n".join(new_lines) + "\n```"
    return text[:start_idx] + new_block + text[end_idx:]


LETTERS = ["A", "B", "C", "D", "E", "F"]


def _cube_single_action_block() -> str:
    return "```\n<piece>: (<face>, <left_edge>, <top_edge>)\n```"


def format_cube_solution(
    solution: Dict[str, Tuple[str, int, int]],
    letters: Optional[List[str]] = None,
) -> str:
    """Format a structured Cube solution mapping as prompt-ready text."""
    lines: List[str] = []
    for letter in letters or LETTERS:
        if letter not in solution:
            continue
        face, left_edge, top_edge = solution[letter]
        lines.append(f"{letter}: ({str(face).capitalize()}, {int(left_edge)}, {int(top_edge)})")
    return "\n".join(lines)


def format_face_grid(arr) -> str:
    """Pretty-print a face array to match cube_easy format."""
    rows = [[int(value) for value in row] for row in arr]
    lines: List[str] = []
    for index, row in enumerate(rows):
        prefix = "[" if index == 0 else " "
        suffix = "]" if index == len(rows) - 1 else ","
        lines.append(f"{prefix}{row}{suffix}")
    return "[" + "\n".join(lines)[1:]


def build_locked_placements_block(
    locked_placements: Dict[str, Tuple[str, int, int]],
) -> str:
    """Build the prompt block describing pre-placed locked pieces."""
    return (
        "The following pieces are already correctly placed and LOCKED:\n"
        "```\n"
        + format_cube_solution(locked_placements)
        + "\n```"
    )


def build_symbolic_faces_block(face_arrays, letters: Optional[List[str]] = None) -> str:
    """Build a text block with the 6 face grids (0 = gap, 1 = bump)."""
    letters = letters or LETTERS
    lines = ["\nBelow is the 6 faces where 0 = gap and 1 = bump."]
    for letter, arr in zip(letters, face_arrays):
        lines.append(f"Face {letter}:")
        lines.append(format_face_grid(arr))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_cube_next_action_instruction(loop_mode: str) -> str:
    """Return the response-format reminder for the next Cube action."""
    del loop_mode
    return ""


def build_cube_initial_state_text(
    question_text: str,
    *,
    remaining_letters: List[str],
    max_steps: int,
    feedback_summary: str,
    with_conflicts: bool,
    has_locked_placements: bool,
) -> str:
    """Build the initial multi-turn state summary shown after the task prompt."""
    lines = [
        question_text,
        "",
        "CURRENT STATE",
        f"- You need to place: {', '.join(remaining_letters) if remaining_letters else '(none)'}.",
        f"- You have at most {max_steps} turns.",
        f"- After each placement you will receive feedback: {feedback_summary}.",
        "- The feedback will show an updated 3D view of the current cube state.",
    ]
    if not with_conflicts:
        lines.append(
            "- Placed pieces remain on the cube even if they do not fit correctly; "
            "examine the 3D view to judge whether edges match."
        )
    if has_locked_placements:
        lines.append(
            "- You may re-place a non-locked piece on a different non-locked face; "
            "if that face already has another non-locked piece, the previous placement will be evicted."
        )
    else:
        lines.append(
            "- You may re-place a piece on a different face; the previous placement will be evicted."
        )
    return "\n".join(lines)


def build_cube_feedback_text(
    *,
    action_desc: str,
    step_idx: int,
    evicted: Optional[str],
    conflict_lines: List[str],
    validator_line: Optional[str],
    symbolic_faces_text: Optional[str],
    placed_summary: str,
    remaining_summary: str,
    remaining_budget: int,
    loop_mode: str,
) -> str:
    """Build the multi-turn Cube feedback text after one placement."""
    lines = [f"Step {step_idx + 1}: placed {action_desc}"]
    if evicted:
        lines.append(f"Note: piece {evicted} was removed from that face to make room.")
    lines.extend(conflict_lines)
    if validator_line:
        lines.append(validator_line)
    if symbolic_faces_text:
        lines.append("")
        lines.append("Piece shapes:")
        lines.append(symbolic_faces_text)
    lines.append(f"Placed: {placed_summary}")
    lines.append(f"Remaining: {remaining_summary}")
    lines.append(f"Budget remaining: {remaining_budget} steps")
    next_action_instruction = build_cube_next_action_instruction(loop_mode)
    if next_action_instruction:
        lines.append(next_action_instruction)
    return "\n".join(lines)


def build_cube_parse_error_text(remaining_budget: int, loop_mode: str) -> str:
    lines = [
        "Could not parse a valid placement from your response.",
        f"Budget remaining: {remaining_budget} turns.",
    ]
    next_action_instruction = build_cube_next_action_instruction(loop_mode)
    if next_action_instruction:
        lines.append(next_action_instruction)
    return "\n".join(lines)


def build_cube_locked_piece_text(
    piece: str,
    remaining_letters: List[str],
    remaining_budget: int,
    loop_mode: str,
) -> str:
    lines = [
        f"Piece {piece} is already correctly placed and locked.",
        f"You can only place: {', '.join(remaining_letters) if remaining_letters else '(none)'}.",
        f"Budget remaining: {remaining_budget} turns.",
    ]
    next_action_instruction = build_cube_next_action_instruction(loop_mode)
    if next_action_instruction:
        lines.append(next_action_instruction)
    return "\n".join(lines)


def build_cube_invalid_orientation_text(
    action_desc: str,
    left_edge: int,
    top_edge: int,
    remaining_budget: int,
    loop_mode: str,
) -> str:
    lines = [
        f"Invalid placement {action_desc}: left-edge {left_edge} and top-edge {top_edge} are not adjacent.",
        "The left and top edges must be consecutive (90-degree turn).",
        f"Budget remaining: {remaining_budget} turns.",
    ]
    next_action_instruction = build_cube_next_action_instruction(loop_mode)
    if next_action_instruction:
        lines.append(next_action_instruction)
    return "\n".join(lines)


def build_cube_locked_face_text(
    action_desc: str,
    face: str,
    locked_piece: Optional[str],
    remaining_budget: int,
    loop_mode: str,
) -> str:
    if locked_piece:
        header = (
            f"Invalid placement {action_desc}: face {face} is occupied by locked piece {locked_piece} "
            "and cannot be replaced."
        )
    else:
        header = (
            f"Invalid placement {action_desc}: face {face} is reserved for a locked pre-placed piece "
            "and cannot be replaced."
        )
    lines = [
        header,
        f"Budget remaining: {remaining_budget} turns.",
    ]
    next_action_instruction = build_cube_next_action_instruction(loop_mode)
    if next_action_instruction:
        lines.append(next_action_instruction)
    return "\n".join(lines)


def build_cube_success_text(action_desc: str, step_idx: int) -> str:
    return "\n".join(
        [
            f"Step {step_idx + 1}: placed {action_desc}",
            "VALID SOLUTION! The cube is correctly assembled.",
        ]
    )


def build_cube_prompt(
    face_arrays,
    locked_placements: Dict[str, Tuple[str, int, int]],
    *,
    loop_mode: str,
    flips: int = -1,
    symbolic: bool = False,
) -> str:
    """Build the initial Cube prompt for the requested evaluation mode."""
    rotation_text = (
        "(flip is not allowed)."
        if int(flips) == 0
        else "(rotation and/or flip)"
    )

#     prompt = f"""\
# You are solving a Happy-Cube (5x5x5) puzzle.

# CONTEXT
# - Six jigsaw pieces labelled A-F.
# - Each piece has four edges on the ORIGINAL piece:
#   Edge1 = left column, Edge2 = top row, Edge3 = right column, Edge4 = bottom row.
# - A placement is written as (CubeFace, LeftEdge, TopEdge) meaning:
#   "When that face is viewed head-on, which original edge is on the LEFT and which on the TOP."

# GOAL
# - Build a valid cube using every piece exactly once.
# - Faces are: Top, Bottom, Front, Back, Left, Right.
# - {rotation_text}
# - The finished cube must close perfectly: no overlaps and no gaps.

# SEAM RULES (very important)
# - LeftEdge and TopEdge must be adjacent (a 90-degree turn).
# - Non-corner edge cells must be complementary bit-for-bit between touching faces:
#   1 must meet 0, and 0 may meet 0; 1 meeting 1 is forbidden.
# - Corner cells contribute to cube corners and must not conflict with neighbors."""

    prompt =  f"""\
You are a spatial-reasoning assistant that solves Happy Cube puzzles.
  
INPUT
• Six jigsaw-style pieces, labelled A–F.
• Each piece is described by the bump/gap pattern on its four edges (Edge1, Edge2, Edge3 and Edge4) as shown in the image, the center squares are always filled.

TASK
Build a 5 × 5 × 5 cube that uses every piece exactly once. 
• Assign each piece to one cube face (Top, Bottom, Front, Back, Left, Right) with proper orientation {rotation_text}.
• The finished cube closed perfectly—no overlaps, no gaps.

The first and last 'square' of the edge contribute to the corners of the cube, make sure:
• When two cube faces touch, the non-corner part of their edges must be complementary bit-for-bit.
• At cube corners (where three faces meet), all connecting edges must align seamlessly.

NOTE
For each piece, report the orientation via the following mental exercise:
1. Hold the finished cube in both hands.
2. Turn the whole cube until the face you are about to name is now facing you like the front of a box.
3. In this view, write down the edge numbers located on left and top. 
    • Format: `<piece>: (<face>, <left_edge>, <top_edge>)`
Note that left-edge and top-edge must be adjacent."""

    if symbolic:
        prompt += f"""

{build_symbolic_faces_block(face_arrays).rstrip()}"""

    if locked_placements:
        prompt += f"""

{build_locked_placements_block(locked_placements)}"""

    if loop_mode == "single_turn_offline":
#         prompt += """

# OUTPUT
# - Return the complete set of six placements.
# - Return six lines in the format `<piece>: (<face>, <left_edge>, <top_edge>)`."""

        prompt += """
        
OUTPUT
Answer the question with the arrangement of 6 pieces. Your response should end with:
The final solution is
```
A: (<face>, <left_edge>, <top_edge>)
B: (<face>, <left_edge>, <top_edge>)
C: (<face>, <left_edge>, <top_edge>)
D: (<face>, <left_edge>, <top_edge>)
E: (<face>, <left_edge>, <top_edge>)
F: (<face>, <left_edge>, <top_edge>)
```
"""
    else:
        prompt += """

OUTPUT
- This is an interactive evaluation. Think step by step and place exactly ONE non-locked piece at each turn.
- If you place a piece onto a face occupied by another non-locked piece, the previous occupant is replaced.
- Locked pieces cannot be replaced.
- Your response should end with:
"""
        prompt += _cube_single_action_block()
        prompt += """
."""

    return prompt
