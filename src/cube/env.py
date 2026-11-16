"""CUBE puzzle rules, validation, and environment."""
from __future__ import annotations
import numpy as np
import random
import json
import gymnasium as gym
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
from gymnasium import spaces
from data_paths import repo_data_path
from cube._rendering.utils import parse_solution_string


# Edges are numbered left=1, top=2, right=3, bottom=4. Rotate counterclockwise,
# then mirror horizontally; each entry gives the resulting left/top edge pair.
rotation_flips_to_edges = {
    (0, False): (1, 2),
    (1, False): (2, 3),
    (2, False): (3, 4),
    (3, False): (4, 1),
    (0, True): (3, 2),
    (1, True): (4, 3),
    (2, True): (1, 4),
    (3, True): (2, 1),
}


edges_to_rotation_flips = {v: k for k, v in rotation_flips_to_edges.items()}


LETTERS = ["A", "B", "C", "D", "E", "F"]


def normalize_cube_solution(
    solution: Optional[Dict[str, object]],
) -> Dict[str, Tuple[str, int, int]]:
    """Normalize a JSON-style Cube solution mapping."""
    normalized: Dict[str, Tuple[str, int, int]] = {}
    if not isinstance(solution, dict):
        return normalized

    for letter in LETTERS:
        value = solution.get(letter)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            continue
        face, left_edge, top_edge = value
        normalized[letter] = (str(face).lower(), int(left_edge), int(top_edge))
    return normalized


def select_locked_placements(
    solution: Dict[str, Tuple[str, int, int]],
    desired_missing: int,
    seed: int = 20,
) -> Dict[str, Tuple[str, int, int]]:
    """Select the non-missing placements that should start locked."""
    rng = random.Random(seed)
    target = max(0, min(len(LETTERS), int(desired_missing)))
    shuffled = list(LETTERS)
    rng.shuffle(shuffled)
    missing = set(shuffled[:target])
    return {
        letter: solution[letter]
        for letter in LETTERS
        if letter in solution and letter not in missing
    }


class SolutionValidator():
    """
    Helper class to validate if a candidate solution is correct.
    A tensor is populated with 1s (minicubes) and 0s (gaps).

    """
    def __init__(self, size=5):

        self.size = size
        self.cube = self._initialize_cube(size=size)

        self.face_to_coordinate_mapping = {
            'front': {'axis': 0, 'value': size-1},
            'top': {'axis': 1, 'value': 0},
            'right': {'axis': 2, 'value': size-1},
            'left': {'axis': 2, 'value': 0},
            'back': {'axis': 0, 'value': 0},
            'bottom': {'axis': 1, 'value': size-1}
        }


    def _initialize_cube(self,size):
        cube = np.zeros([size, size, size])
        cube[1:size-1, 1:size-1, 1:size-1] = 1
        return cube

    def refresh_cube(self):
        self.cube = self._initialize_cube(self.size)

    def _input_validate(self, x, s):
        """
        Check that input (x: face, s: destination string) has proper format and shape.
        - x (face) needs to be a an 2d array (if a list of list of provided -> converted to np array)
        - s (destination string) needs to be in the set of face keys (top, bottom, etc.)
        """
        if type(x) != np.ndarray:
            # check if nested list:
            if type(x) == list and type(x[0]) == list:
                x = np.array(x) # convert to np array
            else:
                raise ValueError(f'face / puzzle piece has unexpected input format: {x}')
        size = self.size
        assert x.shape == (size, size)

        assert s in self.face_to_coordinate_mapping.keys()

        return x

    def _add_oriented_piece(self,
                            face: np.array,
                            destination: str):
        """
        Add single piece (i.e. face) to the tensor. Expects already oriented
        matrix, such that edge a is on top and edge b is to the right.
        Inputs:
        - face: [ p.array ] representing the face grid, where gaps are expected
            as None, for example:
            array([['f', 'f2e', 'f3e', None, None],
                   [None, 'F', 'F', 'F', None],
                   [None, 'F', 'F', 'F', None],
                   ['f4a', 'F', 'F', 'F', None],
                   [None, 'b4a', None, None, 'b']], dtype=object)
            "face" input may also be passed as List[List] whereas the input validator will convert to np array
        - destination: [ str ] string representation of destination face, where a
            piece has to placed in the simulated solution, e.g. "front", or "top" etc.
        """
        # check proper format
        face = self._input_validate(face, destination) # maybe reformats face object, if necessary

        # # convert string rep. of face piece to binary mask:
        # face = self._convert_string_to_binary_mask(face)

        # actually update the cube tensor with current piece at desired location
        self._update_cube(face, destination)

    def _update_cube(self,
                     face: np.array,
                     destination: str):
        """
        Helper routine for populating cube with single piece at the desired location destination.
        """
        # Get current info on coordinates to use
        info = self.face_to_coordinate_mapping[destination]
        axis = info['axis']; value = info['value'] # extract dict
        # print(axis); print(value)

        # Adjust orientation depending on destination. This is a fix because np arrays do not
        # render matrices in an intuitive manner.
        if destination == 'bottom':
            face = np.rot90(np.rot90(np.flip(face, axis=1))) #mirror it #np.rot90(np.rot90(face)) # rotate 180 degrees)
        if destination == 'back':
            face = np.flip(np.rot90(np.rot90(face)), axis=0) #, axis=1)
        if destination == 'left':
            face = np.flip(np.rot90(face), axis=0) #, axis=1)
        if destination == 'right':
            face = np.rot90(face)

        # Dynamically slice the cube depending on which axis we need to "freeze" in the piece addition
        if axis == 0:
            self.cube[int(value), :,:] += face
        elif axis == 1:
            self.cube[:, int(value), :] += face
        elif axis == 2:
            self.cube[:, :, int(value)] += face
        else:
            raise ValueError(f"Unexepcted value for axis {axis}, should be between 0 and 2.")

    def _convert_string_to_binary_mask(self,
                                       face: np.array):
        """
        Helper func to convert face (2d array) with string and None entries
        to binary mask.
        Inputs:
        - face: np array, see example:
                array([['f', 'f2e', 'f3e', None, None],
                           [None, 'F', 'F', 'F', None],
                           [None, 'F', 'F', 'F', None],
                           ['f4a', 'F', 'F', 'F', None],
                           [None, 'b4a', None, None, 'b']], dtype=object)
        Outputs:
        - face as mask (same shaped np array with ints: 1s and 0s)
        """
        return (face != None).astype(int)

    def add_piece(self, face, destination):
        """
        Public method for adding a piece to cube emulation.
        Inputs: > check _add_oriented_piece()
        # TODO: we may need to add a way to handle differently oriented pieces, i.e.
        orient them canonically (e.g. edge a on top, b on the right), create a mapping from
        displayed edge label to secret edge label (e.g. always a for top, etc)
        -
        """
        return self._add_oriented_piece(
                            face,
                            destination
        )

    def _is_valid_solution(self, solution):
        # TODO: assert keys are Face A-F
        # assert all sides are covered
        all_sides = ['top', 'bottom', 'front', 'back', 'left', 'right']
        for (side, edge1, edge2) in solution.values():
            if side not in all_sides:
                return False
            elif (edge1, edge2) not in edges_to_rotation_flips.keys():
                return False
            else:
                all_sides.pop(all_sides.index(side))

        return True

    def validate(self, face_arrays, solution):
        """
        Solution should be a dictionary of 'Face': ('side', edge1, edge2)
        {'A': ('top', 2, 1),
        'B': ('bottom', 4, 3),
        'C': ('front', 2, 3),
        'D': ('back', 1, 2),
        'E': ('left', 2, 3),
        'F': ('right', 3, 2)}

        First virtually ensmble the 6 pieces into a 5x5x5 cube according to the solution,
        then check if there is exactly 1 square in each place.
        """
        if not self._is_valid_solution(solution):
            return False

        def rotate_and_flip(arr, rot_k, flip):
            arr = np.rot90(arr, k=rot_k)
            arr = np.flip(arr, axis=1) if flip else arr
            return arr

        self.refresh_cube()
        solution_per_piece = [solution[piece] for piece in ['A', 'B', 'C', 'D', 'E', 'F']]
        for arr, (side, edge1, edge2) in zip(face_arrays, solution_per_piece):
            arr = rotate_and_flip(arr, *edges_to_rotation_flips[(edge1, edge2)])
            self.add_piece(arr, side)

        correct = (self.cube.max() == 1) and (self.cube.min() == 1)
        return bool(correct)

    def validate_with_text_feedback(self, face_arrays, solution, binary=False):
        """
        Examine the solution as validate(), and then provide text feedback to help MLLM to refine the solution
        """
        # first part, same as self.validate()
        all_sides = ['top', 'bottom', 'front', 'back', 'left', 'right']
        for (side, edge1, edge2) in solution.values():
            if side not in all_sides:
                correct = False
                feedback = "Solution is not legal, all faces should present once."
                return correct, feedback
            elif (edge1, edge2) not in edges_to_rotation_flips.keys():
                correct = False
                feedback = "Solution is not legal, left and top edges should be adjacent."
                return correct, feedback
            else:
                all_sides.pop(all_sides.index(side))

        # valid solution
        def rotate_and_flip(arr, rot_k, flip):
            arr = np.rot90(arr, k=rot_k)
            arr = np.flip(arr, axis=1) if flip else arr
            return arr

        self.refresh_cube()
        oriented = {}   # side → (piece_letter, oriented_array)
        for piece in ['A', 'B', 'C', 'D', 'E', 'F']:
            side, e1, e2 = solution[piece]
            arr          = rotate_and_flip(face_arrays[['A','B','C','D','E','F'].index(piece)],
                                    *edges_to_rotation_flips[(e1, e2)])
            oriented[side] = (piece, arr)          # keep for patterns later
            self.add_piece(arr, side)

        # quick outcome
        is_ok = (self.cube.min() == 1 and self.cube.max() == 1)
        if is_ok:
            return True, "Solution is correct!"
        elif binary:
            return False, "Solution is incorrect!"

        # start to gather edge conflicts for incorrect solution
        # ── 1 · helpers ──────────────────────────────────────────────────────
        size  = self.size
        s     = size - 1
        edge_no = {"west":1, "north":2, "east":3, "south":4}
        #         left-edge, top-edge, right-edge, bottom-edge

        EDGE_NAME = {
            frozenset(("top","front" )): "top-front",
            frozenset(("top","right" )): "top-right",
            frozenset(("top","back"  )): "top-back",
            frozenset(("top","left"  )): "top-left",
            frozenset(("bottom","front")): "bottom-front",
            frozenset(("bottom","right")): "bottom-right",
            frozenset(("bottom","back" )): "bottom-back",
            frozenset(("bottom","left" )): "bottom-left",
            frozenset(("front","left" )): "front-left",
            frozenset(("front","right")): "front-right",
            frozenset(("back","left"  )): "back-left",
            frozenset(("back","right" )): "back-right",
        }

        def edge_bits(arr, compass):
            if   compass == "north": return ''.join('1' if v else '0' for v in arr[0, :])
            elif compass == "south": return ''.join('1' if v else '0' for v in arr[-1, :])
            elif compass == "east":  return ''.join('1' if v else '0' for v in arr[:, -1])
            else:                    return ''.join('1' if v else '0' for v in arr[:, 0])

        # ── 2 · build lists of all edge & corner slices ──────────────────────
        EDGES = [
            (("top","back"), ("north","north"),  self.cube[0, 0, :]),
            (("top","front"), ("south","north"), self.cube[s, 0, :]),
            (("top","left" ), ("west" ,"north"), self.cube[:, 0, 0]),
            (("top","right"), ("east" ,"north"), self.cube[:, 0, s]),

            (("bottom","back" ), ("south","south"), self.cube[0, s, :]),
            (("bottom","front"), ("north","south"), self.cube[s, s, :]),
            (("bottom","left" ), ("west" ,"south"), self.cube[:, s, 0]),
            (("bottom","right"), ("east" ,"south"), self.cube[:, s, s]),

            (("front","left" ), ("west","east"),  self.cube[s, :, 0]),
            (("front","right"), ("east","west"),  self.cube[s, :, s]),
            (("back","left"  ), ("east","east"),  self.cube[0, :, 0]),
            (("back","right" ), ("west","east"),  self.cube[0, :, s]),
        ]

        CORNERS = {
            (0,0,0): ("top","back","left"),
            (s,0,0): ("top","back","right"),
            (0,s,0): ("top","front","left"),
            (s,s,0): ("top","front","right"),
            (0,0,s): ("bottom","back","left"),
            (s,0,s): ("bottom","back","right"),
            (0,s,s): ("bottom","front","left"),
            (s,s,s): ("bottom","front","right"),
        }

        # ── 3 · collect problems ─────────────────────────────────────────────
        edge_problems   = []
        corner_problems = []

        # A · edge clashes / gaps
        for (side1, side2), (d1, d2), vec in EDGES:
            if np.all(vec[1:size-1] == 1):
                continue  # edge is fine

            piece1, arr1 = oriented[side1]
            piece2, arr2 = oriented[side2]
            bits1  = edge_bits(arr1, d1)
            bits2  = edge_bits(arr2, d2)

            edge_problems.append(
                f"Edge at {EDGE_NAME[frozenset((side1, side2))]}: "
                f"{side1.capitalize()} face (piece {piece1} – edge {edge_no[d1]}) "
                f"is not compatible with {side2.capitalize()} face (piece {piece2} - edge {edge_no[d2]})"
            )

        # B · corner clashes / gaps
        for (x,y,z), faces in CORNERS.items():
            if self.cube[x,y,z] == 1:
                continue
            pieces = ', '.join(oriented[f][0] for f in faces)
            kind   = "overlap" if self.cube[x,y,z] > 1 else "gap"
            corner_problems.append(
                f"Corner at {'-'.join(faces)}: pieces {pieces} ({kind})."
            )

        # ── 4 · assemble human report ────────────────────────────────────────
        if not edge_problems and not corner_problems:
            return False, "Solution is incorrect, but no edge or corner mismatch was isolated."

        lines = ["Solution is incorrect!"]
        if edge_problems:
            lines.append(f"\nEdge conflicts ({len(edge_problems)}):")
            lines.extend(f"• {msg}" for msg in edge_problems)
        # if corner_problems:
        #     lines.append(f"\nCorner conflicts ({len(corner_problems)}):")
        #     lines.extend(f"• {msg}" for msg in corner_problems)

        feedback = '\n'.join(lines)
        return False, feedback


FACE_TO_COORD_MAPPING = {
    "front": (0, lambda size: size - 1),
    "back": (0, lambda size: 0),
    "top": (1, lambda size: 0),
    "bottom": (1, lambda size: size - 1),
    "left": (2, lambda size: 0),
    "right": (2, lambda size: size - 1),
}


def orient_face_for_destination(face: np.ndarray, destination: str) -> np.ndarray:
    """Apply the destination-face transform used by the validator/cube tensor."""
    destination = str(destination).lower()
    arr = np.asarray(face)
    if destination == "bottom":
        arr = np.rot90(np.rot90(np.flip(arr, axis=1)))
    elif destination == "back":
        arr = np.flip(np.rot90(np.rot90(arr)), axis=0)
    elif destination == "left":
        arr = np.flip(np.rot90(arr), axis=0)
    elif destination == "right":
        arr = np.rot90(arr)
    return arr


def build_oriented_face_map(
    face_arrays,
    solution: Dict[str, Tuple[str, int, int]],
) -> Dict[str, Tuple[str, np.ndarray]]:
    """Return {cube_face: (piece_letter, oriented_piece_array)} for placed pieces."""
    oriented: Dict[str, Tuple[str, np.ndarray]] = {}
    arrays = np.asarray(face_arrays)
    for piece in LETTERS:
        if piece not in solution:
            continue
        side, e1, e2 = solution[piece]
        rot_flip = edges_to_rotation_flips.get((int(e1), int(e2)))
        if rot_flip is None:
            continue
        rot_k, flip = rot_flip
        arr = np.rot90(arrays[LETTERS.index(piece)], k=rot_k)
        if flip:
            arr = np.flip(arr, axis=1)
        oriented[str(side).lower()] = (piece, arr.astype(np.int32))
    return oriented


def detect_cube_conflicts(
    oriented_faces: Dict[str, Tuple[str, np.ndarray]],
) -> List[Dict[str, object]]:
    """Detect seam and corner overlaps for a partial cube state."""
    if not oriented_faces:
        return []

    size = int(next(iter(oriented_faces.values()))[1].shape[0])
    s = size - 1
    cube = np.zeros((size, size, size), dtype=np.int32)
    placed_letters: Dict[str, str] = {}

    for face_name, (piece, arr) in oriented_faces.items():
        axis, value_fn = FACE_TO_COORD_MAPPING[face_name]
        value = int(value_fn(size))
        placed_letters[face_name] = piece
        transformed = orient_face_for_destination(arr.astype(np.int32), face_name)
        if axis == 0:
            cube[value, :, :] += transformed
        elif axis == 1:
            cube[:, value, :] += transformed
        else:
            cube[:, :, value] += transformed

    conflicts: List[Dict[str, object]] = []
    seams = [
        ("top", "back", cube[0, 0, :]),
        ("top", "front", cube[s, 0, :]),
        ("top", "left", cube[:, 0, 0]),
        ("top", "right", cube[:, 0, s]),
        ("bottom", "back", cube[0, s, :]),
        ("bottom", "front", cube[s, s, :]),
        ("bottom", "left", cube[:, s, 0]),
        ("bottom", "right", cube[:, s, s]),
        ("front", "left", cube[s, :, 0]),
        ("front", "right", cube[s, :, s]),
        ("back", "left", cube[0, :, 0]),
        ("back", "right", cube[0, :, s]),
    ]
    for face1, face2, seam in seams:
        if face1 not in placed_letters or face2 not in placed_letters:
            continue
        inner = seam[1:s]
        if np.any(inner > 1):
            conflicts.append(
                {
                    "faces": (face1, face2),
                    "pieces": (placed_letters[face1], placed_letters[face2]),
                    "type": "overlap",
                    "location": "seam",
                }
            )

    corners = {
        (0, 0, 0): ("top", "back", "left"),
        (s, 0, 0): ("top", "back", "right"),
        (0, s, 0): ("top", "front", "left"),
        (s, s, 0): ("top", "front", "right"),
        (0, 0, s): ("bottom", "back", "left"),
        (s, 0, s): ("bottom", "back", "right"),
        (0, s, s): ("bottom", "front", "left"),
        (s, s, s): ("bottom", "front", "right"),
    }
    for coord, faces in corners.items():
        present_faces = [face for face in faces if face in placed_letters]
        if len(present_faces) < 2:
            continue
        value = int(cube[coord])
        if value > 1:
            conflicts.append(
                {
                    "faces": tuple(present_faces),
                    "pieces": tuple(placed_letters[face] for face in present_faces),
                    "type": "overlap",
                    "location": "corner",
                    "corner": "-".join(faces),
                    "value": value,
                }
            )

    return conflicts


def count_required_flips(solution: Dict[str, Tuple[str, int, int]]) -> Tuple[Dict[str, int], int]:
    """
    Returns (per_piece_flips, total_flips) where each flip is 1 if the piece
    requires a mirror flip under (Left-Edge#, Top-Edge#), else 0.
    solution like: {'A': ('top', 2, 1), ...}
    """
    per_piece: Dict[str, int] = {}
    total = 0
    for piece in ['A','B','C','D','E','F']:
        if piece not in solution:
            continue
        _, left_edge, top_edge = solution[piece]
        rot_k, flip = edges_to_rotation_flips[(left_edge, top_edge)]
        per_piece[piece] = int(bool(flip))
        total += per_piece[piece]
    return per_piece, total


def _letter_to_index(letter: str) -> int:
    order = ['A','B','C','D','E','F']
    return order.index(letter)


def apply_mirror_to_pieces(face_arrays: List[np.ndarray], pieces_to_mirror: List[str]) -> List[np.ndarray]:
    """
    Return a new list of face arrays where the given piece letters have been mirrored (axis=1).
    """
    new_arrays = list(face_arrays)
    for p in pieces_to_mirror or []:
        try:
            idx = _letter_to_index(p)
        except ValueError:
            continue
        arr = new_arrays[idx]
        arr_np = arr if isinstance(arr, np.ndarray) else np.array(arr)
        new_arrays[idx] = np.flip(arr_np, axis=1)
    return new_arrays


def modulate_required_flips(face_arrays: List[np.ndarray],
                            solution: Dict[str, Tuple[str, int, int]],
                            desired_total_flips: int,
                            seed: Optional[int] = None) -> Tuple[List[np.ndarray], Dict[str, List[str]]]:
    """
    Adjust the input face arrays by mirroring selected pieces (axis=1) so that the
    number of pieces that require a mirror flip (under the given solution) equals desired_total_flips.
    Mirroring toggles the parity (flip bit) for any target orientation because rotations are det=+1.

    Returns: (new_face_arrays, {'mirrored': [letters...]})
    """
    rng = random.Random(seed)
    per_piece, current = count_required_flips(solution)
    desired = max(0, min(6, int(desired_total_flips)))
    if current == desired:
        return face_arrays, {'mirrored': []}

    # Choose pieces to toggle parity by mirroring
    need = desired - current
    mirror_letters: List[str] = []
    if need > 0:
        # Increase flips: mirror pieces that currently do not require flip
        candidates = [p for p, f in per_piece.items() if f == 0]
        mirror_letters = rng.sample(candidates, k=min(len(candidates), need))
    else:
        # Decrease flips: mirror pieces that currently require flip
        candidates = [p for p, f in per_piece.items() if f == 1]
        mirror_letters = rng.sample(candidates, k=min(len(candidates), -need))

    new_arrays = apply_mirror_to_pieces(face_arrays, mirror_letters)
    return new_arrays, {'mirrored': mirror_letters}


def modulate_required_flips_restricted(face_arrays: List[np.ndarray],
                                       solution: Dict[str, Tuple[str, int, int]],
                                       desired_total_flips: int,
                                       allowed_letters: List[str],
                                       seed: Optional[int] = None) -> Tuple[List[np.ndarray], Dict[str, List[str]]]:
    """
    Same as modulate_required_flips but only mirrors pieces in 'allowed_letters'
    (e.g., propositions – the ones not already placed in the partial solution).
    If the exact target cannot be reached due to restrictions, mirrors as close as possible.
    """
    rng = random.Random(seed)
    per_piece, _ = count_required_flips(solution)
    desired = max(0, min(6, int(desired_total_flips)))
    if not allowed_letters:
        return face_arrays, {'mirrored': []}
    current_allowed = sum(int(per_piece.get(p, 0)) for p in allowed_letters)
    if current_allowed == desired:
        return face_arrays, {'mirrored': []}

    # Choose candidates within allowed letters
    need = desired - current_allowed
    mirrored: List[str] = []
    if need > 0:
        # Increase flips: choose letters currently not requiring flip within allowed set
        candidates = [p for p, f in per_piece.items() if f == 0 and p in allowed_letters]
        k = min(len(candidates), need)
        mirrored = rng.sample(candidates, k=k)
    else:
        # Decrease flips: choose letters currently requiring flip within allowed set
        candidates = [p for p, f in per_piece.items() if f == 1 and p in allowed_letters]
        k = min(len(candidates), -need)
        mirrored = rng.sample(candidates, k=k)

    new_arrays = apply_mirror_to_pieces(face_arrays, mirrored)
    return new_arrays, {'mirrored': mirrored}


def is_mirror_symmetric(arr: np.ndarray) -> bool:
    """
    Check if face array is symmetric w.r.t. vertical mirror (axis=1).
    """
    try:
        return bool(np.array_equal(arr, np.flip(arr, axis=1)))
    except Exception:
        return False


def toggle_solution_flip_for_letters(solution: Dict[str, Tuple[str, int, int]],
                                     letters: List[str]) -> Dict[str, Tuple[str, int, int]]:
    """
    Return a new solution where for the given letters, the (left_edge, top_edge) pair
    is remapped to the conjugated rotation (−rot_k mod 4) and flip bit toggled.
    This corresponds to pre-mirroring the face array (reflection before applying solution),
    which changes parity and inverts rotation in D4.
    """
    new_sol: Dict[str, Tuple[str, int, int]] = dict(solution)
    for letter in letters or []:
        if letter not in new_sol:
            continue
        face, le, te = new_sol[letter]
        rot_k, flip = edges_to_rotation_flips[(le, te)]
        rot_k2 = (-rot_k) % 4
        le2, te2 = rotation_flips_to_edges[(rot_k2, (not flip))]
        new_sol[letter] = (face, le2, te2)
    return new_sol


def modulate_flips_for_allowed_letters_with_solution_adjustment(
    face_arrays: List[np.ndarray],
    solution: Dict[str, Tuple[str, int, int]],
    allowed_letters: List[str],
    desired_total_flips: int,
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], Dict[str, Tuple[str, int, int]], Dict[str, List[str]]]:
    """Adjust flip difficulty using an explicit set of editable letters."""
    rng = random.Random(seed)
    allowed = [letter for letter in LETTERS if letter in set(allowed_letters or [])]
    per_piece, _ = count_required_flips(solution)
    desired = max(0, min(len(allowed), int(desired_total_flips)))
    info: Dict[str, List[str]] = {"mirrored": [], "solution_toggled": []}

    current_allowed = sum(int(per_piece.get(piece, 0)) for piece in allowed)

    if desired == 0 and current_allowed > 0:
        symmetric_letters: List[str] = []
        for piece, requires_flip in per_piece.items():
            if requires_flip != 1 or piece not in allowed:
                continue
            try:
                idx = _letter_to_index(piece)
                if is_mirror_symmetric(face_arrays[idx]):
                    symmetric_letters.append(piece)
            except Exception:
                continue
        if symmetric_letters:
            k = min(len(symmetric_letters), current_allowed - desired)
            chosen = rng.sample(symmetric_letters, k=k)
            solution = toggle_solution_flip_for_letters(solution, chosen)
            info["solution_toggled"].extend(chosen)
            per_piece, _ = count_required_flips(solution)
            current_allowed = sum(int(per_piece.get(piece, 0)) for piece in allowed)
            if current_allowed == desired:
                return face_arrays, solution, info

    new_arrays, mirror_info = modulate_required_flips_restricted(
        face_arrays=face_arrays,
        solution=solution,
        desired_total_flips=desired,
        allowed_letters=allowed,
        seed=seed,
    )
    mirrored_letters = mirror_info.get("mirrored", []) or []
    info["mirrored"].extend(mirrored_letters)
    if mirrored_letters:
        solution = toggle_solution_flip_for_letters(solution, mirrored_letters)
        info["solution_toggled"].extend(mirrored_letters)
    return new_arrays, solution, info


def count_required_flips(solution: dict) -> tuple[dict[str, int], int]:
    """
    Returns (per_piece_flips, total_flips) where each flip is 1 if the piece
    requires a mirror flip under (Left-Edge#, Top-Edge#), else 0.

    solution must be like:
    {'A': ('top', 2, 1), 'B': ('bottom', 4, 3), ...}
    """
    per_piece = {}
    total = 0
    for piece in ['A','B','C','D','E','F']:
        if piece not in solution:
            continue
        _, left_edge, top_edge = solution[piece]
        rot_k, flip = edges_to_rotation_flips[(left_edge, top_edge)]
        per_piece[piece] = int(bool(flip))
        total += int(bool(flip))
    return per_piece, total


CANON_FACES = ["top", "bottom", "front", "back", "left", "right"]


LETTERS = ["A", "B", "C", "D", "E", "F"]


FACE_ADJACENCY = {
	"top": [
		("north", "back", "north"),
		("south", "front", "north"),
		("west", "left", "north"),
		("east", "right", "north"),
	],
	"bottom": [
		("north", "front", "south"),
		("south", "back", "south"),
		("west", "left", "south"),
		("east", "right", "south"),
	],
	"front": [
		("north", "top", "south"),
		("south", "bottom", "north"),
		("west", "left", "east"),
		("east", "right", "west"),
	],
	"back": [
		("north", "top", "north"),
		("south", "bottom", "south"),
		("west", "right", "east"),
		("east", "left", "west"),
	],
	"left": [
		("north", "top", "west"),
		("south", "bottom", "west"),
		("west", "back", "east"),
		("east", "front", "west"),
	],
	"right": [
		("north", "top", "east"),
		("south", "bottom", "east"),
		("west", "front", "east"),
		("east", "back", "west"),
	],
}


def _load_faces_and_solution(path: str | Path) -> Tuple[np.ndarray, Optional[Dict[str, Tuple[str, int, int]]]]:
	"""
	Load combined JSON: {"faces": [...], "solution": {...}} or plain list-of-faces.
	Returns faces as numpy int array of shape (6, N, N) and optional solution dict.
	"""
	p = Path(path)
	with p.open("r") as f:
		data = json.load(f)
	if isinstance(data, dict):
		faces = np.asarray(data["faces"], dtype=np.uint8)
		solution = data.get("solution", None)
	else:
		faces = np.asarray(data, dtype=np.uint8)
		solution = None
	assert faces.shape[0] == 6 and faces.ndim == 3, "faces must be 6xNxN"
	return faces, solution


class CubeEnv(gym.Env):
	"""
	A small discrete environment for assembling a cube by specifying the mapping:
	  piece_letter -> (cube_face, left_edge, top_edge)

	Action: MultiDiscrete([6, 6, 4, 4])
	  - piece index ∈ [0..5]   for A..F
	  - cube_face   ∈ [0..5]   for ["top","bottom","front","back","left","right"]
	  - left_edge   ∈ [0..3]   encoded as 1..4
	  - top_edge    ∈ [0..3]   encoded as 1..4

Observation: Dict
  - faces:    (6, N, N) uint8 mask grids
  - placed:   (6, 3) int32; each row: [cube_face_idx or -1, left_edge(0..4), top_edge(0..4)]
  - used:     (6,) bool mask of cube faces already used

Termination: All six pieces placed and the `SolutionValidator` validates True.
Episode length limited by `max_steps`.
	"""

	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

	def __init__(
		self,
		data_path: str | Path = None,
		faces=None,
		max_steps: int = 30,
		seed: int = 0,
		render_mode: Optional[str] = None,
		preload_file_solution: bool = False,
		report_conflicts: bool = False,
	):
		super().__init__()
		self.rng = np.random.default_rng(seed)
		self.render_mode = render_mode
		self.max_steps = int(max_steps)
		self.preload_file_solution = bool(preload_file_solution)
		self.report_conflicts = bool(report_conflicts)

		if faces is not None:
			self.faces = np.asarray(faces, dtype=np.uint8)
			self.initial_solution = None
		else:
			if data_path is None:
				data_path = repo_data_path("cube", "cube.json")
			self.faces, self.initial_solution = _load_faces_and_solution(data_path)
		self.N = int(self.faces.shape[1])

		# Spaces
		self.action_space = spaces.MultiDiscrete([6, 6, 4, 4])
		self.observation_space = spaces.Dict(
			{
				"faces": spaces.Box(low=0, high=1, shape=(6, self.N, self.N), dtype=np.uint8),
				"placed": spaces.Box(low=-1, high=6, shape=(6, 3), dtype=np.int32),
				"used": spaces.MultiBinary(6),
			}
		)

		# Internal state
		self.validator = SolutionValidator(size=self.N)
		self._placements: Dict[str, Optional[Tuple[str, int, int]]] = {}
		self._t: int = 0
		self._render_seq: int = 0

		self.reset()

	def _obs(self) -> Dict[str, Any]:
		# placed matrix: each row [cube_face_idx or -1, left(0..4), top(0..4)]
		placed = np.full((6, 3), -1, dtype=np.int32)
		used = np.zeros((6,), dtype=np.int8)
		for i, letter in enumerate(LETTERS):
			val = self._placements.get(letter)
			if val is None:
				placed[i] = np.array([-1, 0, 0], dtype=np.int32)
			else:
				face_name, left, top = val
				face_idx = CANON_FACES.index(face_name)
				placed[i] = np.array([face_idx, left, top], dtype=np.int32)
				used[face_idx] = 1
		return {"faces": self.faces.copy(), "placed": placed, "used": used}

	def _solution_dict(self) -> Optional[Dict[str, Tuple[str, int, int]]]:
		# Only return solution if all pieces placed (no None)
		if any(self._placements.get(letter) is None for letter in LETTERS):
			return None
		return {letter: self._placements[letter] for letter in LETTERS}  # type: ignore

	def _get_oriented_piece(self, letter: str) -> Optional[np.ndarray]:
		"""Get the oriented piece array for a placed piece."""
		placement = self._placements.get(letter)
		if placement is None:
			return None
		face_name, left_edge, top_edge = placement
		piece_idx = LETTERS.index(letter)
		arr = self.faces[piece_idx].copy()
		# Apply rotation and flip based on edge orientation
		rot_flip = edges_to_rotation_flips.get((left_edge, top_edge))
		if rot_flip is None:
			return arr  # Invalid orientation, return as-is
		rot_k, flip = rot_flip
		arr = np.rot90(arr, k=rot_k)
		if flip:
			arr = np.flip(arr, axis=1)
		return arr

	def _get_edge(self, arr: np.ndarray, compass: str) -> np.ndarray:
		"""Extract edge cells from an oriented piece array."""
		if compass == "north":
			return arr[0, :]
		elif compass == "south":
			return arr[-1, :]
		elif compass == "west":
			return arr[:, 0]
		else:  # east
			return arr[:, -1]

	def _check_conflicts(self) -> List[Dict[str, Any]]:
		"""
		Check for seam and corner overlaps using the shared cube-tensor
		approach. Returns a list of conflict
		dicts with keys:
		  - faces: tuple of (face1, face2) that conflict
		  - pieces: tuple of (letter1, letter2)
		  - type: "overlap"
		"""
		solution = {
			letter: placement
			for letter, placement in self._placements.items()
			if placement is not None
		}
		oriented_faces = build_oriented_face_map(self.faces, solution)
		return [dict(item) for item in detect_cube_conflicts(oriented_faces)]

		conflicts: List[Dict[str, Any]] = []
		size = self.N
		s = size - 1

		# Coordinate mapping: face → (axis, slice_value)
		face_to_coord = {
			"front": (0, s), "back": (0, 0),
			"top": (1, 0),   "bottom": (1, s),
			"left": (2, 0),  "right": (2, s),
		}

		cube = np.zeros((size, size, size), dtype=np.int32)
		placed_faces: set = set()
		placed_letters: Dict[str, str] = {}  # face_name → letter

		# 1. Project each placed piece into the 3D tensor
		for letter, placement in self._placements.items():
			if placement is None:
				continue
			face_name, left_edge, top_edge = placement
			piece_idx = LETTERS.index(letter)
			arr = self.faces[piece_idx].copy().astype(np.int32)

			# User rotation + flip
			rot_flip = edges_to_rotation_flips.get((left_edge, top_edge))
			if rot_flip is None:
				continue
			rot_k, flip = rot_flip
			arr = np.rot90(arr, k=rot_k)
			if flip:
				arr = np.flip(arr, axis=1)

			# Destination transform (mirrors SolutionValidator._update_cube)
			if face_name == "bottom":
				arr = np.rot90(np.rot90(np.flip(arr, axis=1)))
			elif face_name == "back":
				arr = np.flip(np.rot90(np.rot90(arr)), axis=0)
			elif face_name == "left":
				arr = np.flip(np.rot90(arr), axis=0)
			elif face_name == "right":
				arr = np.rot90(arr)

			axis, value = face_to_coord[face_name]
			if axis == 0:
				cube[value, :, :] += arr
			elif axis == 1:
				cube[:, value, :] += arr
			else:
				cube[:, :, value] += arr

			placed_faces.add(face_name)
			placed_letters[face_name] = letter

		# 2. Check all 12 seams; only report those between two placed faces
		SEAMS = [
			("top", "back",    cube[0, 0, :]),
			("top", "front",   cube[s, 0, :]),
			("top", "left",    cube[:, 0, 0]),
			("top", "right",   cube[:, 0, s]),
			("bottom", "back",  cube[0, s, :]),
			("bottom", "front", cube[s, s, :]),
			("bottom", "left",  cube[:, s, 0]),
			("bottom", "right", cube[:, s, s]),
			("front", "left",   cube[s, :, 0]),
			("front", "right",  cube[s, :, s]),
			("back", "left",    cube[0, :, 0]),
			("back", "right",   cube[0, :, s]),
		]

		for face1, face2, seam in SEAMS:
			if face1 not in placed_faces or face2 not in placed_faces:
				continue
			# Exclude corner cells (governed by 3-face rule, not pairwise)
			inner = seam[1:s]
			if np.any(inner > 1):
				conflicts.append({
					"faces": (face1, face2),
					"pieces": (placed_letters[face1], placed_letters[face2]),
					"type": "overlap",
				})

		return conflicts

	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
		if seed is not None:
			self.rng = np.random.default_rng(seed)
		self._t = 0
		self._render_seq = 0
		self._last_placed_face = None
		self._locked_faces = set()
		# Clear placements
		self._placements = {letter: None for letter in LETTERS}
		# Optionally preload a partial solution from the file (disabled by default)
		use_file_solution = bool(isinstance(self.initial_solution, dict) and self.preload_file_solution)
		if use_file_solution:
			for k, v in self.initial_solution.items():
				kk = k.upper()
				if kk in LETTERS:
					face, le, te = v
					face = face.lower()
					if face in CANON_FACES and 1 <= int(le) <= 4 and 1 <= int(te) <= 4:
						self._placements[kk] = (face, int(le), int(te))
		# Or from a solution string passed via reset(options={"solution_str": "A: (Top, 3, 2) ..." })
		if isinstance(options, dict) and options.get("solution_str"):
			try:
				smap = parse_solution_string(str(options["solution_str"]))
				self.apply_solution_mapping(smap, clear=False)
			except Exception:
				pass
		return self._obs(), {}

	def step(self, action):
		self._t += 1
		piece_i, face_i, left_i, top_i = map(int, action)
		letter = LETTERS[piece_i]
		face_name = CANON_FACES[face_i]
		self._last_placed_face = face_name.capitalize()
		left_edge = left_i + 1
		top_edge = top_i + 1

		# If the target cube-face is already occupied by another letter,
		# treat this as an Edit: evict the occupant and place the new piece.
		occupied_by: Optional[str] = None
		for k, v in self._placements.items():
			if v is None:
				continue
			if v[0] == face_name and k != letter:
				occupied_by = k
				break

		reward = 0.0
		info: Dict[str, Any] = {}
		terminated = False
		truncated = False

		if occupied_by is not None:
			# Edit: free the slot then place the new mapping
			self._placements[occupied_by] = None
			self._placements[letter] = (face_name, left_edge, top_edge)
			info["replaced"] = occupied_by
		else:
			self._placements[letter] = (face_name, left_edge, top_edge)

			sol = self._solution_dict()
			if sol is not None:
				# Evaluate only when all are set
				ok, msg = self.validator.validate_with_text_feedback(self.faces, sol, binary=False)
				info["feedback"] = msg
				if ok:
					terminated = True

		if not terminated and self._t >= self.max_steps:
			truncated = True

		# Optionally report conflicts
		if self.report_conflicts:
			conflicts = self._check_conflicts()
			info["conflicts"] = conflicts
			info["num_conflicts"] = len(conflicts)
			info["valid_partial"] = len(conflicts) == 0

		return self._obs(), reward, terminated, truncated, info

	def apply_solution_mapping(self, mapping: Dict[str, Tuple[str, int, int]], clear: bool = False) -> None:
		"""
		Apply a (possibly partial) solution mapping to current placements.
		mapping: {"A": ("Top", 3, 2), ...}
		- If clear is True, all previous placements are wiped before applying.
		- Face names are case-insensitive; edges must be 1..4.
		"""
		if clear:
			self._placements = {letter: None for letter in LETTERS}
		for k, (face, le, te) in mapping.items():
			kk = k.upper()
			if kk not in LETTERS:
				continue
			face_l = str(face).lower()
			if face_l not in CANON_FACES:
				continue
			le = int(le); te = int(te)
			if not (1 <= le <= 4 and 1 <= te <= 4):
				continue
			# editing semantics: evict existing occupant of this face
			for other, v in list(self._placements.items()):
				if v is not None and v[0] == face_l and other != kk:
					self._placements[other] = None
			self._placements[kk] = (face_l, le, te)

	def set_solution_string(self, solution_str: str, clear: bool = False) -> None:
		"""
		Convenience: parse 'A: (Top, 3, 2) B: (Back, 4, 3)' and apply.
		"""
		mapping = parse_solution_string(solution_str)
		self.apply_solution_mapping(mapping, clear=clear)

	def render(self, save: bool = False, save_dir: Optional[str] = None,
			   formats: Optional[List[str]] = None) -> Dict[str, Any]:
		"""
		Render current state as RGB arrays. Optionally save to disk.

		Args:
			formats: list of formats to render, subset of ('net', 'view', 'views').
			         Defaults to all three when None.

		Returns:
			dict with requested format keys containing numpy RGB arrays (H, W, 3).
			If save=True, also includes '<format>_path' keys with file paths.
		"""
		from cube.rendering import plot_cube_net, plot_cube_3d, plot_cube_3d_views

		_all = ('net', 'view', 'views')
		active = set(formats) if formats is not None else set(_all)

		# Build partial solution mapping for placed pieces only
		sol: Dict[str, Tuple[str, int, int]] = {}
		for letter, triple in self._placements.items():
			if triple is None:
				continue
			face, le, te = triple
			sol[letter] = (face, int(le), int(te))

		# Prepare save paths if needed
		save_paths: Dict[str, str] = {}
		if save:
			if save_dir is not None:
				out_dir = Path(save_dir)
			else:
				out_dir = Path(__file__).resolve().parent / "outputs"
			out_dir.mkdir(parents=True, exist_ok=True)
			step_tag = f"{self._render_seq:03d}"
			self._render_seq += 1
			save_paths = {
				"net":   str(out_dir / f"cube_{step_tag}.png"),
				"view":  str(out_dir / f"cube3d_{step_tag}.png"),
				"views": str(out_dir / f"cube3d_views_{step_tag}.png"),
			}

		result: Dict[str, Any] = {}

		# Pre-placed (locked) faces stay flush; model-placed faces can be offset if they conflict
		locked = getattr(self, '_locked_faces', set())

		if 'net' in active:
			result["net"] = plot_cube_net(
				faces=self.faces.tolist(),
				save_path=save_paths.get("net"),
				show=False,
				title=None,
				solution=sol,
				show_labels=True,
				show_face_names=True,
			)
			if save:
				result["net_path"] = save_paths["net"]

		if 'view' in active:
			result["view"] = plot_cube_3d(
				faces=self.faces.tolist(),
				save_path=save_paths.get("view"),
				show=False,
				solution=sol,
				tile_depth=1.0,
				flush_faces=locked,
			)
			if save:
				result["view_path"] = save_paths["view"]

		if 'views' in active:
			result["views"] = plot_cube_3d_views(
				faces=self.faces.tolist(),
				save_path=save_paths.get("views"),
				solution=sol,
				tile_depth=1.0,
				flush_faces=locked,
			)
			if save:
				result["views_path"] = save_paths["views"]

		return result

	def close(self):
		pass


def make_env(**kwargs) -> CubeEnv:
	return CubeEnv(**kwargs)
