"""
Blox perception task: prompts and response parsing for board state extraction.
"""
from __future__ import annotations

import re
import ast
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def normalize_height_map(grid: List[List[int]], height_map: List[List[int]]) -> List[List[int]]:
    """Return tile heights with every void cell fixed to the documented value 0."""
    return [
        [
            int(height_map[row][col]) if int(grid[row][col]) != 0 else 0
            for col in range(len(grid[row]))
        ]
        for row in range(len(grid))
    ]


def board_state_prompt(rows: int, cols: int) -> str:
    """Build the prompt for the board state parsing perception task."""
    view_desc = f"an isometric 3D rendering of a Blox puzzle board ({rows} rows x {cols} columns)"
    visual_guide = """\
- **Normal tiles**: Tan/beige raised cuboid tiles — these are safe walkable cells.
- **Void cells**: Dark recessed areas (gaps in the platform) — the block falls if it touches these.
- **Goal tile**: A red raised tile with a small diamond marker — the block must stand upright here to win.
- **Fragile tiles**: Light sand raised tiles with crack/X markers — unsafe if the block stands upright on them.
- **Crumble tiles**: Light gray raised tiles with chipped edges and crack markers — safe while occupied, then collapse after being fully left.
- **Elevator tiles**: Purple raised tiles with up/down arrow markers — transport the block between LOW and HIGH height levels.
- **Elevated tiles**: Tiles floating above the main platform (shown in exploded view, shifted up and right) — these are on the upper (HIGH) level.
- **Block**: A blue cuboid sitting on the board.
  * STANDING: Tall and upright, occupying exactly 1 cell (1x1 footprint).
  * LYING_H: Flat, occupying 2 cells side-by-side in the same row (horizontal).
  * LYING_V: Flat, occupying 2 cells stacked in the same column (vertical).
  * The block has a height level (LOW or HIGH). When HIGH, the block and elevated tiles shift in the exploded view."""
    coord_desc = """\
- Row labels appear along the right edge; column labels appear along the bottom edge.
- Rows increase mostly downward and slightly leftward, starting at row 0.
- Columns increase mostly rightward and slightly downward, starting at column 0."""

    return f"""You are given {view_desc}.

TASK
Extract the complete board state from the image.

VISUAL GUIDE
{visual_guide}

COORDINATE SYSTEM
{coord_desc}

CELL SYMBOLS
- `.` = Normal tile
- `#` = Void (empty/gap)
- `G` = Goal tile
- `C` = Crumble tile (cracked appearance, collapses after you leave it)
- `S` = Switch tile
- `~` = Bridge (OFF/impassable)
- `=` = Bridge (ON/passable)
- `E` = Elevator tile (purple, transports block between height levels)

OUTPUT FORMAT (use exactly this format):
Grid:
[[row0_col0, row0_col1, ..., row0_col{cols - 1}], [row1_col0, ...], ..., [row{rows - 1}_col0, ...]]
Block Cells: (row, col) if STANDING, or (row1, col1), (row2, col2) if LYING
Block Orientation: STANDING|LYING_H|LYING_V
Goal Position: (row, col)
Block Height: LOW|HIGH
Height Map:
[[row0_col0, row0_col1, ..., row0_col{cols - 1}], [row1_col0, ...], ..., [row{rows - 1}_col0, ...]]

Replace each cell value with ".", "#", "G", "C", "S", "~", "=", or "E". The grid should be a {rows}x{cols} 2D array with {rows} inner lists of {cols} elements each.
For block cells, list ALL cells occupied by the block: one cell if STANDING, two cells if LYING (any order).
Block Height is LOW (ground level) or HIGH (upper level).
Height Map is a {rows}x{cols} 2D array of 0 (LOW) or 1 (HIGH) for each cell, indicating which height level that tile is on. Void cells should be 0.

IMPORTANT: You must provide your best answer using the exact output format above. Do not ask clarifying questions or refuse to answer. The image contains all the information you need.
"""


def extract_board_state(response: str, rows: int, cols: int) -> Dict[str, Any]:
    """Parse LLM response to extract board state components.

    Returns dict with keys: grid, block_cells, block_orient, goal_pos,
    block_height, and optionally height_map.
    Raises ValueError if parsing fails for any required component.
    """
    # Strip thinking blocks (e.g. Qwen3.6 via vLLM doesn't separate reasoning_content)
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
    if "```" in response:
        response = response.replace("```python", "```").replace("```json", "```").replace("```", "")

    result: Dict[str, Any] = {}

    # --- Grid extraction (bracket-balancing) ---
    grid_label = re.search(r"Grid\s*:", response, re.IGNORECASE)
    start_pos = grid_label.end() if grid_label else 0
    try:
        start_idx = response.index("[", start_pos)
    except ValueError:
        start_idx = response.index("[")

    depth = 0
    end_idx = None
    for i, ch in enumerate(response[start_idx:], start=start_idx):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    if end_idx is None:
        raise ValueError("Unbalanced brackets while parsing grid.")

    grid_literal = response[start_idx:end_idx]

    def _parse_array(text: str):
        for parser in (ast.literal_eval, json.loads):
            try:
                return parser(text)
            except Exception:
                pass
        # Models often output unquoted symbols like [., #, G, F, C, S, ~, =, E] — quote them
        quoted = re.sub(r'(?<=[\[,])\s*([.#GFCS~=E])\s*(?=[,\]])', r' "\1"', text, flags=re.IGNORECASE)
        for parser in (ast.literal_eval, json.loads):
            try:
                return parser(quoted)
            except Exception:
                pass
        raise ValueError(f"Cannot parse grid array: {text[:200]}")

    grid_py = _parse_array(grid_literal)
    normalized_grid = []
    for row in grid_py:
        normalized_row = []
        for cell in row:
            token = str(cell).strip()
            normalized_row.append(token.upper() if token.isalpha() else token)
        normalized_grid.append(normalized_row)
    result["grid"] = np.array(normalized_grid, dtype=str)

    # --- Block cells (1 or 2 positions) ---
    block_label = re.search(r"Block\s+(?:Cells|Position)\s*:", response, re.IGNORECASE)
    if not block_label:
        raise ValueError("Block cells/position not found in response.")
    # Extract all (row, col) pairs on the same line
    block_line_start = block_label.start()
    block_line_end = response.find("\n", block_line_start)
    if block_line_end == -1:
        block_line_end = len(response)
    block_line = response[block_line_start:block_line_end]
    coord_pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", block_line)
    if not coord_pairs:
        raise ValueError("No coordinate pairs found for block cells.")
    result["block_cells"] = {(int(r), int(c)) for r, c in coord_pairs}

    # --- Block orientation ---
    orient_match = re.search(
        r"Block\s+Orientation\s*:\s*(STANDING|LYING_H|LYING_V)", response, re.IGNORECASE
    )
    if not orient_match:
        raise ValueError("Block orientation not found in response.")
    result["block_orient"] = orient_match.group(1).upper()

    # --- Goal position ---
    goal_match = re.search(
        r"Goal\s+Position\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", response, re.IGNORECASE
    )
    if not goal_match:
        raise ValueError("Goal position not found in response.")
    result["goal_pos"] = (int(goal_match.group(1)), int(goal_match.group(2)))

    # --- Block height (optional) ---
    height_match = re.search(
        r"Block\s+Height\s*:\s*(LOW|HIGH)", response, re.IGNORECASE
    )
    if height_match:
        result["block_height"] = 1 if height_match.group(1).upper() == "HIGH" else 0
    else:
        result["block_height"] = 0

    # --- Height map (optional) ---
    height_label = re.search(r"Height\s+Map\s*:", response, re.IGNORECASE)
    if height_label:
        hm_start_pos = height_label.end()
        try:
            hm_start_idx = response.index("[", hm_start_pos)
            hm_depth = 0
            hm_end_idx = None
            for i, ch in enumerate(response[hm_start_idx:], start=hm_start_idx):
                if ch == "[":
                    hm_depth += 1
                elif ch == "]":
                    hm_depth -= 1
                    if hm_depth == 0:
                        hm_end_idx = i + 1
                        break
            if hm_end_idx is not None:
                hm_literal = response[hm_start_idx:hm_end_idx]
                hm_py = _parse_array(hm_literal)
                result["height_map"] = np.array(hm_py, dtype=int)
        except (ValueError, Exception):
            pass  # height map is optional, fail gracefully

    return result
