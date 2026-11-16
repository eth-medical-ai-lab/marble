"""Blox planning prompts."""

from __future__ import annotations
from typing import List, Optional
from blox.constants import Orientation, TileType
from blox.env import Block, Board, GameState




def _cell_char(
    board: Board,
    r: int,
    c: int,
    block_cells: dict,
    active_bridges: frozenset,
    collapsed_crumble: frozenset = frozenset(),
) -> str:
    """Return the display character for a single cell."""
    if (r, c) in block_cells:
        return block_cells[(r, c)]
    tile = board.effective_tile_at(r, c, collapsed_crumble)
    if tile == TileType.GOAL:
        return "G"
    if tile == TileType.VOID:
        return "#"
    if tile == TileType.SWITCH:
        return "S*" if (r, c) in board.hard_switches else "S"
    if tile == TileType.BRIDGE:
        group = board.bridges.get((r, c))
        is_on = group is not None and group in active_bridges
        return "=" if is_on else "~"
    if tile == TileType.FRAGILE:
        return "F"
    if tile == TileType.CRUMBLE:
        return "C"
    if (r, c) in board.elevators:
        return "E"
    return "."


def _render_single_grid(
    board: Board,
    block_cells: dict,
    active_bridges: frozenset,
    collapsed_crumble: frozenset = frozenset(),
) -> str:
    """Render a single ASCII grid without height levels."""
    lines = []
    col_header = "  x " + " ".join(f"{c}" for c in range(board.cols))
    lines.append(col_header)
    for r in range(board.rows):
        row_chars = []
        for c in range(board.cols):
            row_chars.append(_cell_char(board, r, c, block_cells, active_bridges, collapsed_crumble))
        lines.append(f"y{r:<2d}" + " ".join(row_chars))
    return "\n".join(lines)


def board_to_text(
    board: Board,
    block: Optional[Block] = None,
    goal_r: Optional[int] = None,
    goal_c: Optional[int] = None,
    active_bridges: frozenset = frozenset(),
    current_height: int = 0,
    collapsed_crumble: frozenset = frozenset(),
) -> str:
    """
    Create a symbolic ASCII grid representation.

    When the board has elevators/height_map, outputs TWO grids (LOW and HIGH).
    Otherwise returns a single grid.
    """
    del goal_r, goal_c

    has_height = bool(board.height_map) and bool(board.elevators)

    block_cells = {}
    if block is not None:
        orient_char = {
            Orientation.STANDING: "B",
            Orientation.LYING_H: "H",
            Orientation.LYING_V: "V",
        }[block.orientation]
        for r, c in block.cells():
            block_cells[(r, c)] = orient_char

    if not has_height:
        return _render_single_grid(board, block_cells, active_bridges, collapsed_crumble)

    grids = []
    for level in (0, 1):
        level_name = "LOW" if level == 0 else "HIGH"
        level_block = block_cells if level == current_height else {}
        lines = [f"--- {level_name} LEVEL (height {level}) ---"]
        col_header = "  x " + " ".join(f"{c}" for c in range(board.cols))
        lines.append(col_header)
        for r in range(board.rows):
            row_chars = []
            for c in range(board.cols):
                is_elevator = (r, c) in board.elevators
                cell_h = board.cell_height(r, c)
                tile = board.effective_tile_at(r, c, collapsed_crumble)
                if tile == TileType.VOID:
                    row_chars.append("#")
                elif is_elevator or cell_h == level:
                    row_chars.append(
                        _cell_char(
                            board,
                            r,
                            c,
                            level_block,
                            active_bridges,
                            collapsed_crumble,
                        )
                    )
                else:
                    row_chars.append("#")
            lines.append(f"y{r:<2d}" + " ".join(row_chars))
        grids.append("\n".join(lines))

    return "\n\n".join(grids)


def state_description(state: GameState) -> str:
    """
    Generate a natural-language description of the current game state
    suitable for inclusion in an LLM prompt.
    """
    orient_names = {
        Orientation.STANDING: "STANDING (upright on one cell)",
        Orientation.LYING_H: "LYING HORIZONTALLY (flat across two cells left-right)",
        Orientation.LYING_V: "LYING VERTICALLY (flat across two cells top-bottom)",
    }

    cells = state.block.cells()
    cells_str = ", ".join(f"(x={c}, y={r})" for r, c in cells)

    parts = [
        f"Block position: anchor at (x={state.block.c}, y={state.block.r})",
        f"Block orientation: {orient_names[state.block.orientation]}",
        f"Block occupies cells: {cells_str}",
        f"Goal: (x={state.goal_c}, y={state.goal_r})",
        f"Grid size: {state.board.cols} cols (x) x {state.board.rows} rows (y)",
    ]

    if state.board.elevators:
        height_name = "LOW (0)" if state.current_height == 0 else "HIGH (1)"
        parts.append(f"Block height level: {height_name}")

        elev_str = ", ".join(f"(x={c},y={r})" for r, c in sorted(state.board.elevators))
        parts.append(f"Elevator tiles ({len(state.board.elevators)}): {elev_str}")
        parts.append("  Elevators toggle the block between LOW and HIGH when the block lands on them.")

        if state.board.height_map:
            elevated = [
                (r, c)
                for r in range(state.board.rows)
                for c in range(state.board.cols)
                if state.board.cell_height(r, c) == 1 and (r, c) not in state.board.elevators
            ]
            if elevated:
                elev_str = ", ".join(f"(x={c},y={r})" for r, c in sorted(elevated))
                parts.append(f"Elevated tiles ({len(elevated)}): {elev_str}")

            goal_h = state.board.cell_height(state.goal_r, state.goal_c)
            goal_h_name = "LOW (0)" if goal_h == 0 else "HIGH (1)"
            parts.append(f"Goal height level: {goal_h_name}")

    fragile_cells = [
        (r, c)
        for r in range(state.board.rows)
        for c in range(state.board.cols)
        if state.board.grid[r][c] == TileType.FRAGILE
    ]
    if fragile_cells:
        cells_str = ", ".join(f"(x={c},y={r})" for r, c in sorted(fragile_cells))
        parts.append(f"Fragile tiles ({len(fragile_cells)}): {cells_str}")
        parts.append("  WARNING: Block falls if STANDING on a fragile tile. Must cross them while LYING.")

    crumble_cells = [
        (r, c)
        for r in range(state.board.rows)
        for c in range(state.board.cols)
        if state.board.grid[r][c] == TileType.CRUMBLE and (r, c) not in state.collapsed_crumble
    ]
    if crumble_cells:
        cells_str = ", ".join(f"(x={c},y={r})" for r, c in sorted(crumble_cells))
        parts.append(f"Crumble tiles ({len(crumble_cells)}): {cells_str}")
        parts.append("  WARNING: CRUMBLE tiles are safe while occupied, but collapse into holes after the block fully leaves them.")

    if state.collapsed_crumble:
        cells_str = ", ".join(f"(x={c},y={r})" for r, c in sorted(state.collapsed_crumble))
        parts.append(f"Collapsed crumble tiles ({len(state.collapsed_crumble)}): {cells_str}")

    if state.board.switches:
        n_groups = len(set(state.board.switches.values()))
        parts.append(f"Switch groups: {n_groups}")
        for (s_r, s_c), group in sorted(state.board.switches.items()):
            parts.append(f"  Switch at (x={s_c}, y={s_r}) controls bridge group {group}")

    if state.board.bridges:
        for group in sorted(set(state.board.bridges.values())):
            bcells = [(r, c) for (r, c), g in state.board.bridges.items() if g == group]
            status = "PASSABLE" if group in state.active_bridges else "IMPASSABLE"
            cells_str = ", ".join(f"(x={c},y={r})" for r, c in sorted(bcells))
            parts.append(f"  Bridge group {group} [{status}]: {cells_str}")

    return "\n".join(parts)


def _single_move_block() -> str:
    return "```\n<UP|DOWN|LEFT|RIGHT>\n```"


def build_blox_next_action_instruction(loop_mode: str) -> str:
    """Return the response-format reminder for the next Blox move."""
    del loop_mode
    return ""


def build_blox_parse_error_text(loop_mode: str) -> str:
    lines = ["I couldn't parse a valid move from your response."]
    next_action_instruction = build_blox_next_action_instruction(loop_mode)
    if next_action_instruction:
        lines.append(next_action_instruction)
    return "\n".join(lines)


def build_blox_fall_feedback_text(
    *,
    action_name: str,
    actions_taken: List[str],
    remaining_moves: int,
    loop_mode: str,
    env_feedback: Optional[str] = None,
    collapsed_text: Optional[str] = None,
) -> str:
    lines = [
        env_feedback
        or (
            f"This is an invalid action. Moving {action_name} would cause the block to fall off the board. "
            "The block remains in its previous position. Please try a different move."
        )
    ]
    if collapsed_text:
        lines.append(collapsed_text)
    lines.append(f"Moves so far: {', '.join(actions_taken)}")
    lines.append(f"Remaining moves: {remaining_moves}")
    next_action_instruction = build_blox_next_action_instruction(loop_mode)
    if next_action_instruction:
        lines.append(next_action_instruction)
    return "\n".join(lines)


def build_blox_success_feedback_text(
    *,
    action_name: str,
    step_idx: int,
    collapsed_text: Optional[str] = None,
) -> str:
    lines = [f"Move {step_idx + 1}: {action_name} - the block is STANDING on the GOAL. You win!"]
    if collapsed_text:
        lines.append(collapsed_text)
    return "\n".join(lines)


def build_blox_continue_feedback_text(
    *,
    action_name: str,
    step_idx: int,
    actions_taken: List[str],
    remaining_moves: int,
    loop_mode: str,
    collapsed_text: Optional[str] = None,
) -> str:
    lines = [f"Move {step_idx + 1}: {action_name} applied successfully."]
    if collapsed_text:
        lines.append(collapsed_text)
    lines.append(f"Moves so far: {', '.join(actions_taken)}")
    lines.append(f"Remaining moves: {remaining_moves}")
    next_action_instruction = build_blox_next_action_instruction(loop_mode)
    if next_action_instruction:
        lines.append(next_action_instruction)
    return "\n".join(lines)


def _shared_instruction_block(
    has_switches: bool = False,
    has_mixed_bridges: bool = False,
    has_fragile: bool = False,
    has_crumble: bool = False,
    has_hard_switches: bool = False,
    has_elevators: bool = False,
) -> str:
    safe_tile_names = ["NORMAL", "GOAL"]
    if has_fragile:
        safe_tile_names.append("FRAGILE")
    if has_crumble:
        safe_tile_names.append("CRUMBLE")
    if len(safe_tile_names) > 2:
        safe_tiles = ", ".join(safe_tile_names[:-1]) + f", or {safe_tile_names[-1]}"
    else:
        safe_tiles = " or ".join(safe_tile_names)

    base = f"""You are solving a Blox puzzle. A 1x1x2 rectangular block sits on a grid. Your goal is to roll the block so it stands UPRIGHT on the GOAL cell.

RULES:
- The block can be rolled UP, DOWN, LEFT, or RIGHT.
- The block has three orientations:
  * STANDING: upright on a single cell (1x1 footprint)
  * LYING HORIZONTALLY: flat, occupying two cells side by side (same row, adjacent columns)
  * LYING VERTICALLY: flat, occupying two cells stacked (same column, adjacent rows)
- When STANDING and you roll LEFT/RIGHT, the block tips over horizontally (occupies 2 cells in a row).
- When STANDING and you roll UP/DOWN, the block tips over vertically (occupies 2 cells in a column).
- When LYING HORIZONTALLY and you roll LEFT/RIGHT, the block stands up on one cell.
- When LYING HORIZONTALLY and you roll UP/DOWN, it slides (stays horizontal).
- When LYING VERTICALLY and you roll UP/DOWN, the block stands up on one cell.
- When LYING VERTICALLY and you roll LEFT/RIGHT, it slides (stays vertical).
- ALL cells occupied by the block must be on {safe_tiles} tiles. If any cell is VOID or out of bounds, the block falls and you lose.
- You WIN when the block is STANDING on the GOAL cell."""

    hard_note = ""
    if has_hard_switches:
        hard_note = (
            "\n- IMPORTANT: Switches marked as HARD (S*) only activate when the block is STANDING upright on them. "
            "Lying on a hard switch does NOT toggle its bridges. You must plan your orientation to arrive STANDING on hard switches."
        )

    if has_mixed_bridges:
        switch_rules = f"""

SWITCH AND BRIDGE RULES:
- SWITCH tiles (S) toggle bridges. When ANY part of the block lands on a switch, all bridges in that switch's group TOGGLE: passable becomes impassable, impassable becomes passable.{hard_note}
- BRIDGE tiles can start IMPASSABLE (~) or PASSABLE (=). Check the state description to see each bridge group's current status.
- The block falls if it steps on an IMPASSABLE bridge (~).
- CAUTION: Some bridges start PASSABLE (=). Activating their switch will make them IMPASSABLE -- do NOT step on switches that control already-passable bridges!
- The block can safely rest on switch tiles (they behave like normal tiles for movement).
- Switches and bridges share group IDs: switch group N controls bridge group N."""
    else:
        switch_rules = f"""

SWITCH AND BRIDGE RULES:
- SWITCH tiles (S) are special tiles. When ANY part of the block lands on a switch after a move, all bridges in that switch's group TOGGLE between passable and impassable.{hard_note}
- BRIDGE tiles start as IMPASSABLE (shown as ~). When toggled ON by a switch, they become PASSABLE (shown as =). The block falls if it steps on an impassable bridge.
- Landing on the same switch again toggles bridges back to impassable.
- The block can safely rest on switch tiles (they behave like normal tiles for movement).
- Switches and bridges are grouped: switch group 0 controls bridge group 0, etc."""

    fragile_rules = ""
    if has_fragile:
        fragile_rules = """

FRAGILE TILE RULES:
- FRAGILE tiles (F) are cracked tiles that collapse under concentrated weight.
- If the block is STANDING (upright) on a fragile tile, it falls through and you lose.
- The block can safely cross fragile tiles when LYING (horizontally or vertically), because the weight is distributed across two cells.
- You must plan your orientation carefully: approach fragile tiles while LYING, not STANDING."""

    crumble_rules = ""
    if has_crumble:
        crumble_rules = """

CRUMBLE TILE RULES:
- CRUMBLE tiles (C) are single-use support tiles.
- A CRUMBLE tile is safe while the block is currently occupying it.
- Once the block fully leaves a CRUMBLE tile, that tile collapses permanently into a hole.
- A collapsed CRUMBLE tile behaves like VOID for the rest of the puzzle.
- You must plan around irreversible changes: crossing a CRUMBLE tile can remove that route for future backtracking."""

    elevator_rules = ""
    if has_elevators:
        elevator_rules = """

ELEVATOR RULES:
- The board has TWO height levels: LOW (0) and HIGH (1). Some tiles are ELEVATED (at height 1).
- The block can only move between tiles at the SAME height level. Elevated tiles are like cliffs -- you cannot roll onto them from ground level or vice versa.
- ELEVATOR tiles (E) are special: they are accessible from BOTH height levels. When the block is STANDING (upright) on an elevator tile, it automatically toggles the block's height (LOW->HIGH or HIGH->LOW). The block must be STANDING -- lying on an elevator does NOT activate it.
- After an elevator toggles your height, you can now access tiles at the new height level.
- To WIN, the block must be STANDING on the GOAL at the SAME height as the goal tile.
- The board state is shown as TWO grids: one for LOW level and one for HIGH level. Tiles at the other height appear as '#' (inaccessible). Elevator tiles appear on BOTH grids.
- Strategy: plan which elevators to use to reach the goal at the correct height."""

    symbols = """

GRID SYMBOLS:
  . = Normal tile (safe)
  # = Void (hole - block falls if it touches this)
  G = Goal tile (stand upright here to win)"""

    if has_fragile:
        symbols += """
  F = Fragile tile (safe when LYING, collapses if STANDING)"""

    if has_crumble:
        symbols += """
  C = Crumble tile (safe while occupied, collapses permanently after you leave it)"""

    if has_elevators:
        symbols += """
  E = Elevator tile (toggles block between LOW and HIGH height levels)"""

    if has_switches:
        if has_hard_switches:
            symbols += """
  S* = Hard switch tile (ONLY activates when block is STANDING upright on it; toggles associated bridges)
  ~ = Bridge (OFF/impassable - block falls)
  = = Bridge (ON/passable - safe)"""
        else:
            symbols += """
  S = Switch tile (safe to stand on; toggles associated bridges)
  ~ = Bridge (OFF/impassable - block falls)
  = = Bridge (ON/passable - safe)"""

    symbols += """
  B = Block (standing)
  H = Block (lying horizontally, both cells)
  V = Block (lying vertically, both cells)

Coordinates: (x, y) where x is column index (left->right), y is row index (top->bottom).
Directions: UP = toward y=0 (top of grid), DOWN = toward last y (bottom), LEFT = toward x=0, RIGHT = toward last x.

MOVEMENT EXAMPLES (showing cell positions after each move):
  STANDING at (3,2):
    RIGHT -> LYING_H occupying (4,2) and (5,2)
    LEFT  -> LYING_H occupying (1,2) and (2,2)
    DOWN  -> LYING_V occupying (3,3) and (3,4)
    UP    -> LYING_V occupying (3,0) and (3,1)
  LYING_H at (4,2)-(5,2):
    RIGHT -> STANDING at (6,2)
    LEFT  -> STANDING at (3,2)
    DOWN  -> LYING_H at (4,3)-(5,3)  (slides, stays horizontal)
    UP    -> LYING_H at (4,1)-(5,1)  (slides, stays horizontal)
  LYING_V at (3,3)-(3,4):
    DOWN  -> STANDING at (3,5)
    UP    -> STANDING at (3,2)
    RIGHT -> LYING_V at (4,3)-(4,4)  (slides, stays vertical)
    LEFT  -> LYING_V at (2,3)-(2,4)  (slides, stays vertical)
"""

    prompt = base
    if has_fragile:
        prompt += fragile_rules
    if has_crumble:
        prompt += crumble_rules
    if has_elevators:
        prompt += elevator_rules
    if has_switches:
        prompt += switch_rules
    prompt += symbols
    return prompt


def _mode_instruction_block(
    mode: str,
    move_budget: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> str:
    if mode == "single_turn_offline":
        mode = "single_shot"
    elif mode == "multi_turn_conversation":
        mode = "multi_turn"

    if mode == "single_shot":
        prompt = """
TASK
Solve the full puzzle from the current state. Return a complete sequence of moves that reaches the GOAL.

OUTPUT
Think step by step about the block's orientation after each move.
Your response should end with:
```
UP, DOWN, LEFT, RIGHT, ...
```"""
        if move_budget is not None:
            prompt += f"\n\nIMPORTANT: Your solution must reach the goal within {move_budget} moves. Excess moves will be truncated."
        return prompt

    if mode == "multi_turn":
        step_line = f"Solve this Blox puzzle in at most {max_steps} moves." if max_steps is not None else "Solve this Blox puzzle."
        prompt = f"""
TASK
{step_line}
You will receive the updated board state after each move.
Choose exactly one next move for this turn.

OUTPUT
Think step by step about the block's orientation after each move.
Your response should end with:
"""
        return prompt + _single_move_block()

    raise ValueError(f"Unknown Blox prompt mode: {mode}")


def build_instruction_prompt(
    has_switches: bool = False,
    has_mixed_bridges: bool = False,
    has_fragile: bool = False,
    has_crumble: bool = False,
    has_hard_switches: bool = False,
    has_elevators: bool = False,
    move_budget: Optional[int] = None,
    max_steps: Optional[int] = None,
    mode: str = "single_shot",
) -> str:
    return (
        _shared_instruction_block(
            has_switches=has_switches,
            has_mixed_bridges=has_mixed_bridges,
            has_fragile=has_fragile,
            has_crumble=has_crumble,
            has_hard_switches=has_hard_switches,
            has_elevators=has_elevators,
        )
        + "\n\n"
        + _mode_instruction_block(mode=mode, move_budget=move_budget, max_steps=max_steps)
    )


def build_observation_prompt(
    state: GameState,
    include_image: bool = True,
    include_symbolic: bool = True,
    minimal_obs: bool = False,
) -> str:
    parts = []

    if include_symbolic:
        parts.append("Current board state:")
        parts.append(
            board_to_text(
                state.board,
                state.block,
                state.goal_r,
                state.goal_c,
                active_bridges=state.active_bridges,
                current_height=state.current_height,
                collapsed_crumble=state.collapsed_crumble,
            )
        )
        if not minimal_obs:
            parts.append("")
            parts.append(state_description(state))

    if include_image:
        if parts:
            parts.append("")
        parts.append("(See the attached image for a visual representation of the board.)")

    return "\n".join(parts).strip()
