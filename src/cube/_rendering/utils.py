from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


Face = Sequence[Sequence[int]]

# Conventions
EDGE_INDEX = {"Top": 0, "Right": 1, "Bottom": 2, "Left": 3}
INDEX_EDGE = {v: k for k, v in EDGE_INDEX.items()}
VALID_CUBE_FACES = {"Top", "Bottom", "Left", "Right", "Front", "Back"}


def normalize_solution_mapping(data: Dict) -> Dict[str, Tuple[str, int, int]]:
    out: Dict[str, Tuple[str, int, int]] = {}
    for k, v in data.items():
        k_up = str(k).upper()
        if isinstance(v, dict):
            cube_face = str(v["cube_face"]).capitalize()
            left_num = int(v["left"])
            top_num = int(v["top"])
        else:
            cube_face = str(v[0]).capitalize()
            left_num = int(v[1])
            top_num = int(v[2])
        if cube_face not in VALID_CUBE_FACES:
            raise ValueError(f"Invalid cube face '{cube_face}' for {k_up}.")
        out[k_up] = (cube_face, left_num, top_num)
    return out


def rotate_face_from_edges(face_np: np.ndarray, left_edge_num: int, top_edge_num: int) -> np.ndarray:
    """
    Edge numbering on the ORIGINAL unrotated face:
      1 = first column (left edge)
      2 = first row    (top edge)
      3 = last column  (right edge)
      4 = last row     (bottom edge)

    Allow rotation plus planar flips to realize any (left, top) edge mapping.
    """
    left_edge_num = int(left_edge_num)
    top_edge_num = int(top_edge_num)
    if left_edge_num not in (1, 2, 3, 4) or top_edge_num not in (1, 2, 3, 4):
        raise ValueError("Edge numbers must be in {1,2,3,4}.")

    for k in range(4):
        arr = np.rot90(face_np, k)
        e_left = (1 + k - 1) % 4 + 1
        e_top = (2 + k - 1) % 4 + 1
        e_right = (3 + k - 1) % 4 + 1
        e_bottom = (4 + k - 1) % 4 + 1

        if e_left == left_edge_num and e_top == top_edge_num:
            return arr
        if e_right == left_edge_num and e_top == top_edge_num:
            return np.fliplr(arr)
        if e_left == left_edge_num and e_bottom == top_edge_num:
            return np.flipud(arr)
        if e_right == left_edge_num and e_bottom == top_edge_num:
            return np.flipud(np.fliplr(arr))

    raise ValueError(
        f"No rotation/flip can place original left={left_edge_num} and top={top_edge_num}."
    )


def parse_solution_string(sol_str: str) -> Dict[str, Tuple[str, int, int]]:
    """
    Parse string like:
      "A: (Top, 3, 2) B: (Back, 4, 3) ..."
    """
    pattern = re.compile(
        r"\b([A-F])\s*:\s*\(\s*(Top|Bottom|Left|Right|Front|Back)\s*,\s*([1-4])\s*,\s*([1-4])\s*\)",
        re.IGNORECASE,
    )
    out: Dict[str, Tuple[str, int, int]] = {}
    for m in pattern.finditer(sol_str):
        face_letter = m.group(1).upper()
        cube_face = m.group(2).capitalize()
        left_num = int(m.group(3))
        top_num = int(m.group(4))
        out[face_letter] = (cube_face, left_num, top_num)
    if not out:
        raise ValueError("Could not parse solution string. Expected format like 'A: (Top, 3, 2) ...'")
    return out


def load_solution(solution_arg: str) -> Dict[str, Tuple[str, int, int]]:
    """
    Load solution mapping either from a JSON file or parse it directly from a string.
    """
    p = Path(solution_arg)
    if p.exists():
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Solution JSON must be an object mapping face letters to tuples/dicts.")
        return normalize_solution_mapping(data)
    return parse_solution_string(solution_arg)


def compute_oriented_placements(
    faces_np: List[np.ndarray],
    face_order: Sequence[str],
    solution: Dict[str, Tuple[str, int, int]] = None,
) -> List[Tuple[str, np.ndarray, Tuple[str, Tuple[int, int]]]]:
    """
    Compute oriented placements for cube faces.
    Returns a list of (letter, face_array, (cube_face, (top,left_cell_offset))).
    """
    n = faces_np[0].shape[0]
    layout_by_cube_face: Dict[str, Tuple[int, int]] = {
        "Top": (0 * n, 1 * n),
        "Front": (1 * n, 1 * n),
        "Bottom": (2 * n, 1 * n),
        "Left": (1 * n, 0 * n),
        "Right": (1 * n, 2 * n),
        "Back": (1 * n, 3 * n),
    }
    placements: List[Tuple[str, np.ndarray, Tuple[str, Tuple[int, int]]]] = []
    letter_to_index = {name: idx for idx, name in enumerate(face_order)}
    if solution is not None:
        used_cube_faces = set()
        for letter, (cube_face, left_num, top_num) in solution.items():
            letter = str(letter).upper()
            cube_face_norm = str(cube_face).capitalize()
            if letter not in letter_to_index:
                raise ValueError(f"Solution references unknown face '{letter}'.")
            if cube_face_norm not in layout_by_cube_face:
                raise ValueError(f"Invalid cube face '{cube_face}' for {letter}.")
            if cube_face_norm in used_cube_faces:
                raise ValueError(f"Duplicate assignment of cube face '{cube_face}'.")
            used_cube_faces.add(cube_face_norm)
            face_idx = letter_to_index[letter]
            rotated = rotate_face_from_edges(faces_np[face_idx], left_num, top_num)
            placements.append((letter, rotated, (cube_face_norm, layout_by_cube_face[cube_face_norm])))
    else:
        default_map = {"C": "Top", "F": "Front", "B": "Bottom", "E": "Left", "D": "Right", "A": "Back"}
        for letter, cube_face in default_map.items():
            face_idx = letter_to_index[letter]
            placements.append((letter, faces_np[face_idx], (cube_face, layout_by_cube_face[cube_face])))
    return placements



