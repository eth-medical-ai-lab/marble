from __future__ import annotations

import numpy as np
import ast
import re
import json

def extract_2d_array(response: str):
    # Strip thinking blocks (e.g. Qwen3.6 via vLLM doesn't separate reasoning_content)
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
    if "```" in response:
        response = response.replace("```python", "```").replace("```json", "```").replace("```", "")

    phrase_pat = re.compile(r"Here is the converted 2D array:", re.IGNORECASE)
    match = phrase_pat.search(response)
    start: int | None = None
    
    if match:
        try:
            start = response.index("[", match.end())
        except ValueError:
            start = None

    if start is None:
        dbl = response.find("[[")
        if dbl != -1:
            start = dbl

    if start is None:
        raise ValueError("Opening '[' for array not found in response (after phrase or '[[').")

    depth = 0
    array_literal = None
    for i, ch in enumerate(response[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                array_literal = response[start: i + 1]
                break
    if array_literal is None:
        raise ValueError("Unbalanced brackets while searching for the array.")

    def _parse_array(text: str):
        for parser in (ast.literal_eval, json.loads):
            try:
                return parser(text)
            except Exception:
                pass
        sanitized = text.replace('\\"', '"').replace("\\'", "'")
        return ast.literal_eval(sanitized)

    python_array = _parse_array(array_literal)
    return np.array(python_array, dtype=str)


def board_parsing_prompt(size: int) -> str:
    return f"""You are given an image of a {size}x{size} Labyrinth board.

TASK
Identify the tile type and rotation for every cell by examining the light-colored corridors (open pathways) visible on each tile. Output a {size}x{size} grid of tokens.

HOW TO READ THE IMAGE
- Each cell contains a tile with light/white corridors (passable paths) and darker textured walls.
- A corridor "opens" to a side (up/down/left/right) when the light pathway reaches the edge of that tile on that side.
- For each tile, determine which sides have corridor openings, then match to one of the tokens below.

TILE TYPES AND ROTATIONS
Each rotation adds 90° clockwise. Openings list the sides where corridors exit the tile.

- I-tile (straight corridor, 2 openings on opposite sides):
  * I0 = opens up/down (vertical)
  * I1 = opens left/right (horizontal)
  Note: I2≡I0, I3≡I1. Use I0 or I1 only.

- L-tile (corner, 2 openings on adjacent sides):
  * L0 = opens up/right
  * L1 = opens down/right
  * L2 = opens down/left
  * L3 = opens up/left

- T-tile (T-junction, 3 openings):
  * T0 = opens up/left/right (closed down)
  * T1 = opens up/down/right (closed left)
  * T2 = opens down/left/right (closed up)
  * T3 = opens up/down/left (closed right)

GRID LAYOUT
- Rows top→bottom (row 0 at top), columns left→right (col 0 at left).

OUTPUT
Examine the image carefully, tile by tile, and provide your best answer. Do not ask for clarification or refuse to answer.
You must answer with "Here is the converted 2D array: [array]" where [array] is a {size}x{size} Python list of lists.
Example row: ["L0", "T1", "I0", "L2", "T0", "I1", "L3"]
"""
