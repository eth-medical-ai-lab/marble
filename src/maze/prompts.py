"""MAZE instructions, feedback, and symbolic observations."""
from __future__ import annotations
from typing import List


def board_to_tokens(board) -> List[List[str]]:
    return [[f"{board.grid[y][x].shape}{board.grid[y][x].rot}" for x in range(board.w)] for y in range(board.h)]


def objects_to_grid(board) -> List[List[str]]:
    grid = [["-" for _ in range(board.w)] for __ in range(board.h)]
    for lab, (ox, oy) in board.objects.items():
        if 0 <= ox < board.w and 0 <= oy < board.h:
            grid[oy][ox] = str(lab)
    return grid


def extra_token(board) -> str:
    return f"{board.extra_tile.shape}{board.extra_tile.rot}"


def _maze_action_output_block(size: int) -> str:
    return (
        "```\n"
        "Action: INSERT(side='<left|right|top|bottom>', slot=<1|3|5>, rotation=<0|1|2|3>)\n"
        "```\n"
        "or\n"
        "```\n"
        f"Action: MOVE(x=<0..{size-1}>, y=<0..{size-1}>)\n"
        "```"
    )


def _input_block(size: int, include_symbolic_obs: bool, with_image: bool) -> str:
    if include_symbolic_obs:
        return f"""INPUT
You will be given the current game observation:
- Board ({size}x{size}): each tile has corridors/walls (shape in {{I, L, T}} plus rotation in {{0,1,2,3}}), indexed 0..6 on both axes.
- Objects grid ({size}x{size}): a Python list of lists of strings (letters or '-' for empty) indicating item placement.
- Extra tile: the tile currently out of the board (with shape/rotation), encoded by a token (see conventions below).
- Player position: (x,y) of our player, 0-based.
- Target: a treasure object represented by its name mapped to a letter (A-X) in symbolic representation.
- Visual encoding: the player is a green circular pawn labeled "P1"; the target is a black icon of the object.
- Right-side panel on the image: shows "Extra Tile" and "Target" sections with the current extra tile and the target.
- last_insert: the previous insertion, if any, as (side, slot); used to enforce the "no immediate reversal" rule.
"""

    extra_image_line = ""
    if with_image:
        extra_image_line = (
            '\n- an image of the current board, which includes a right-side panel labeled "Extra Tile" '
            'and "Target" showing the current extra tile and the target.'
        )
    return f"""INPUT
You will be given the current game observation:
- Target: a treasure object represented by its name mapped to a letter (A-X) in symbolic representation.
- Visual encoding: the player is a green circular pawn labeled "P1"; the target is a black icon of the object.
- last_insert: the previous insertion, if any, as (side, slot); used to enforce the "no immediate reversal" rule.{extra_image_line}
"""


def _shared_instruction_block(size: int, include_symbolic_obs: bool, with_image: bool) -> str:
    return f"""You are a spatial-planning assistant that plays Labyrinth from arbitrary board states.

{_input_block(size, include_symbolic_obs, with_image)}
RULES
INSERT action:
  - Set the loose tile to the specified rotation (see tile conventions below), then insert it from side into slot.
  - This shifts that entire row/column by one; the opposite edge tile is ejected and becomes the new loose tile.
  - Direction of ejection: Left inserts eject Right edge; Right ejects Left; Top ejects Bottom; Bottom ejects Top.
  - Any pawns on that line shift with the tiles; if pushed off, they wrap to the newly inserted tile on the opposite edge.
  - Objects behavior: treasures/items stay attached to their tiles; they shift with the row/column and if on the ejected tile, they leave the board (no wrap) and only re-enter if that tile is later re-inserted.
  - No immediate reversal: you cannot insert from the opposite side into the same slot as last_insert.

MOVE action:
  - Move your pawn along open corridors to any reachable cell (unlimited distance along connected paths).
  - Adjacency rule: a step to an orthogonal neighbor is legal only if both tiles open toward each other (e.g. W/E).
  - To stay, output MOVE(); the keyword is required (not optional) when using plain-text action formats.
  - Success is declared if the Target is reached after moving.

At each step, freely choose INSERT or MOVE -- there is no mandatory ordering between them.

Tile/token convention for Extra
- Shapes: I (straight corridor), L (corner/bend), T (three-way junction).
- Openings are on North (up), East (right), South (down), West (left).
- Rotation is clockwise; board top is North.

Visual Descriptions:
- I (straight): A corridor connecting two opposite sides
  * I0 = vertical corridor (connects top-bottom, North-South)
  * I1 = horizontal corridor (connects left-right, East-West)
  * I2 == I0, I3 == I1 (same visual appearance). Prefer I0/I1.

- L (corner): A bent corridor connecting two adjacent sides
  * L0 = connects top+right (North+East)
  * L1 = connects right+bottom (East+South)
  * L2 = connects bottom+left (South+West)
  * L3 = connects left+top (West+North)

- T (junction): A corridor with three openings
  * T0 = opens to top+left+right (North+East+West)
  * T1 = opens to right+bottom+top (East+South+North)
  * T2 = opens to bottom+left+right (South+West+East)
  * T3 = opens to left+top+bottom (West+North+South)

NOTE
- Grid coordinates: x,y in [0..6]; x is column index (left-right), y is row index (top-bottom).
- Inserts:
  - side in {{left, right, top, bottom}}.
  - slot in {{1,3,5}} (the only shove-able rows/columns).
  - For left/right, slot = row index (y). For top/bottom, slot = column index (x).
- rotation in {{0,1,2,3}} = absolute rotation (0, 90, 180, 270 degrees clockwise) to set the loose tile before insertion following the convention above.
  Do not apply relative rotations; ignore the current extra tile orientation.
- The target is a single uppercase letter (A-X).
- The extra tile token encodes shape+rotation: I0/I1, L0..L3, T0..T3.
- Do not ask for more information; use only the provided image/text to decide.
"""


def build_instruction_prompt(
    size: int,
    include_symbolic_obs: bool,
    with_image: bool,
    mode: str = "multi_turn",
) -> str:
    base = _shared_instruction_block(size, include_symbolic_obs, with_image)

    if mode == "single_shot":
        return (
            base
            + f"""

TASK
Propose a complete plan as a sequence of steps. Each step is either INSERT or MOVE -- freely chosen in any order. Aim to reach the Target as soon as possible.
Include only the steps necessary to reach the target; stop at success (no extra steps).

Step budget: MAXIMUM of 20 steps. Produce the shortest valid plan and stop immediately when the target is reached.

PLAN FORMAT
Plan:
Step 1: INSERT(side='<left|right|top|bottom>', slot=<1|3|5>, rotation=<0|1|2|3>)
Step 2: MOVE(x=<0..{size-1}>, y=<0..{size-1}>) or MOVE()
Step N: INSERT(...) or MOVE(...)

OUTPUT
Your response should end with:
```
Step 1: INSERT(side='<left|right|top|bottom>', slot=<1|3|5>, rotation=<0|1|2|3>)
Step 2: MOVE(x=<0..{size-1}>, y=<0..{size-1}>) (use MOVE() to stay)
...
Step N: INSERT(...) or MOVE(...)
```
"""
        )

    if mode != "multi_turn":
        raise ValueError(f"Unknown Maze prompt mode: {mode}")

    return (
        base
        + f"""

TASK
Choose exactly one action -- either INSERT or MOVE -- to progress toward the Target (reach it if possible; otherwise improve your position for future steps).

ACTION SPACE (exact output strings)
- INSERT:
  Action: INSERT(side='<left|right|top|bottom>', slot=<1|3|5>, rotation=<0|1|2|3>)

- MOVE:
  Action: MOVE(x=<0..{size-1}>, y=<0..{size-1}>) (use MOVE() to stay)

OUTPUT
- Your chosen action must be legal and obey the rules above.
- Your response should end with:
"""
        + _maze_action_output_block(size)
        + """
"""
    )


def build_followup_prompt(
    obs_text: str,
    size: int,
    mode: str = "multi_turn",
    remaining_budget: int | None = None,
) -> str:
    """Build the non-initial Maze prompt shown after each environment step."""
    budget_line = ""
    if remaining_budget is not None:
        budget_line = f"\n\nRemaining budget: {remaining_budget} turns."
    if mode == "multi_turn":
        return obs_text + budget_line
    return obs_text


def build_invalid_llm_action_feedback(kind: str, reason: str, remaining_budget: int | None = None) -> str:
    """Build feedback when the Maze parser/legality checks reject the action text."""
    budget_text = f" Remaining budget: {remaining_budget} turns." if remaining_budget is not None else ""
    if kind == "insert":
        return (
            f"Invalid insertion: {reason}. Your proposed action was not legal. "
            f"Please try again with a valid INSERT action.{budget_text}"
        )
    return (
        f"Invalid move: {reason}. The target cell is not reachable. "
        f"Please try again with a valid MOVE action.{budget_text}"
    )


def build_invalid_env_action_feedback(kind: str, remaining_budget: int | None = None) -> str:
    """Build feedback when the Maze environment rejects an action."""
    budget_text = f" Remaining budget: {remaining_budget} turns." if remaining_budget is not None else ""
    if kind == "insert":
        return (
            "Invalid insertion: this insertion reverses the previous one (forbidden). "
            f"Please choose a different INSERT action.{budget_text}"
        )
    return (
        "Invalid move: target cell not reachable from your current position. "
        f"Please choose a different MOVE action.{budget_text}"
    )
