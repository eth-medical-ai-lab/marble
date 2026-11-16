"""Construct solved cubes, then permute and orient the observed pieces."""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from cube.env import SolutionValidator, rotation_flips_to_edges, select_locked_placements

RECIPE = {"size": 5, "missing": {"easy": 2, "hard": 4},
          "permutation": True, "flips": True, "rotations": True}
FACES = ["top", "bottom", "front", "back", "left", "right"]
LETTERS = ["A", "B", "C", "D", "E", "F"]

CORNERS = {
    "a": ["top", "front", "left"],
    "b": ["top", "front", "right"],
    "c": ["bottom", "front", "right"],
    "d": ["bottom", "front", "left"],
    "e": ["top", "back", "right"],
    "f": ["top", "back", "left"],
    "g": ["bottom", "back", "left"],
    "h": ["bottom", "back", "right"],
}

EDGES_FACES = {
    "ab": ["front", "top"],
    "ad": ["left", "front"],
    "fa": ["left", "top"],
    "cb": ["right", "front"],
    "be": ["right", "top"],
    "cd": ["front", "bottom"],
    "ch": ["right", "bottom"],
    "gd": ["left", "bottom"],
    "fe": ["back", "top"],
    "he": ["right", "back"],
    "fg": ["left", "back"],
    "gh": ["back", "bottom"],
}

FACE_CORNERS = {
    "top": ("f", "e", "b", "a"),
    "bottom": ("d", "c", "h", "g"),
    "front": ("a", "b", "c", "d"),
    "back": ("e", "f", "g", "h"),
    "left": ("f", "a", "d", "g"),
    "right": ("b", "e", "h", "c"),
}

def _capitalize_face(face_name: str) -> str:
    return face_name[:1].upper() + face_name[1:].lower()


def _solution_json(solution: Dict[str, Tuple[str, int, int]]) -> Dict[str, List[object]]:
    return {
        letter: [_capitalize_face(face_name), int(left_edge), int(top_edge)]
        for letter, (face_name, left_edge, top_edge) in solution.items()
    }


def generate(num_samples, seed, output_dir, config):
    """Yield validated 5x5 cubes with two or four pieces left to place."""
    for index in range(num_samples):
        sample_seed = seed + index
        sample = generate_instance(seed=sample_seed, subset=config, missing=RECIPE["missing"][config])
        yield {"id": f"sample_{index:04d}", "seed": sample_seed, **sample}


def _get_2d_coordinate(
    mini_cube: str,
    corner_coordinates: Dict[str, np.ndarray],
    n: int,
) -> np.ndarray:
    if len(mini_cube) == 1:
        if mini_cube not in corner_coordinates:
            raise ValueError(f"{mini_cube=} not present on the target face")
        return corner_coordinates[mini_cube]

    if len(mini_cube) != 3:
        raise ValueError(f"Unexpected mini-cube name: {mini_cube}")

    start_corner = mini_cube[0]
    index = int(mini_cube[1])
    end_corner = mini_cube[-1]
    direction = corner_coordinates[end_corner] - corner_coordinates[start_corner]
    coords = corner_coordinates[start_corner] + (index - 1) / (n - 1) * direction
    return coords.astype(int)


def _build_face_grid(face_name: str, mini_cubes: Sequence[str], n: int) -> List[List[Optional[str]]]:
    grid: List[List[Optional[str]]] = [[None for _ in range(n)] for _ in range(n)]

    for row in range(1, n - 1):
        for col in range(1, n - 1):
            grid[row][col] = "F"

    corners = FACE_CORNERS[face_name]
    corner_xys = np.array([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]])
    corner_coordinates = dict(zip(corners, corner_xys))

    for mini_cube in mini_cubes:
        x, y = _get_2d_coordinate(mini_cube, corner_coordinates, n=n)
        grid[int(y)][int(x)] = mini_cube

    return grid


def _assign_mini_cubes(n: int, rng: random.Random) -> Tuple[Dict[str, str], Dict[str, List[List[Optional[str]]]]]:
    mini_cube_to_face: Dict[str, str] = {}

    for corner, face_options in CORNERS.items():
        mini_cube_to_face[corner] = rng.choice(face_options)

    edge_positions = list(range(2, n))
    for edge_name, face_options in EDGES_FACES.items():
        start_corner = edge_name[0]
        end_corner = edge_name[-1]
        counts = {face_options[0]: 0, face_options[1]: 0}

        for corner_name in (start_corner, end_corner):
            corner_face = mini_cube_to_face[corner_name]
            if corner_face in counts:
                counts[corner_face] += 1

        shuffled_positions = list(edge_positions)
        rng.shuffle(shuffled_positions)
        total_positions = len(shuffled_positions)
        for idx, cell_idx in enumerate(shuffled_positions):
            remaining_after = total_positions - idx - 1
            missing_faces = [face_name for face_name, count in counts.items() if count == 0]
            if missing_faces and remaining_after < len(missing_faces):
                chosen_face = missing_faces[0]
            else:
                chosen_face = rng.choice(face_options)
            mini_cube_to_face[f"{start_corner}{cell_idx}{end_corner}"] = chosen_face
            counts[chosen_face] += 1

    pieces = {
        face_name: sorted(
            [mini_cube for mini_cube, assigned_face in mini_cube_to_face.items() if assigned_face == face_name]
        )
        for face_name in FACES
    }
    face_grids = {
        face_name: _build_face_grid(face_name, pieces[face_name], n=n)
        for face_name in FACES
    }
    return mini_cube_to_face, face_grids


def _grid_to_mask(grid: List[List[Optional[str]]]) -> np.ndarray:
    return (np.array(grid, dtype=object) != None).astype(np.uint8)  # noqa: E711


def _apply_piece_transform(face_array: np.ndarray, flip: bool, rot_k: int) -> np.ndarray:
    transformed = np.flip(face_array, axis=1) if flip else face_array
    return np.rot90(transformed, k=rot_k)


def _randomize_face_arrays(
    canonical_faces: Sequence[np.ndarray],
    rng: random.Random,
    *,
    allow_permutation: bool,
    allow_flip: bool,
    allow_rotation: bool,
) -> Tuple[np.ndarray, List[int], List[Tuple[bool, int]]]:
    transformed_faces: List[np.ndarray] = []
    flip_rotations: List[Tuple[bool, int]] = []

    for face_array in canonical_faces:
        flip = bool(rng.getrandbits(1)) if allow_flip else False
        rot_k = rng.randrange(4) if allow_rotation else 0
        transformed_faces.append(_apply_piece_transform(face_array, flip, rot_k))
        flip_rotations.append((flip, rot_k))

    permutation = list(range(len(canonical_faces)))
    if allow_permutation:
        rng.shuffle(permutation)

    observed_faces = np.array([transformed_faces[idx] for idx in permutation], dtype=np.uint8)
    return observed_faces, permutation, flip_rotations


def _derive_solution(
    permutation: Sequence[int],
    flip_rotations: Sequence[Tuple[bool, int]],
) -> Dict[str, Tuple[str, int, int]]:
    observed_transforms = [flip_rotations[idx] for idx in permutation]
    inverse_rotations = [((4 - rot_k) % 4, flip) for flip, rot_k in observed_transforms]
    cube_faces = [FACES[idx] for idx in permutation]
    edges = [rotation_flips_to_edges[(rot_k, flip)] for rot_k, flip in inverse_rotations]

    return {
        letter: (cube_face, int(left_edge), int(top_edge))
        for letter, cube_face, (left_edge, top_edge) in zip(LETTERS, cube_faces, edges)
    }


def generate_instance(
    *,
    seed: int,
    size: int = 5,
    subset: str = "easy",
    missing: int = 2,
    allow_permutation: bool = True,
    allow_flip: bool = True,
    allow_rotation: bool = True,
) -> Dict[str, object]:
    if size < 3:
        raise ValueError("Happy Cube faces must be at least 3x3.")

    rng = random.Random(seed)
    mini_cube_to_face, face_grids = _assign_mini_cubes(size, rng)
    canonical_face_arrays = [_grid_to_mask(face_grids[face_name]) for face_name in FACES]
    observed_face_arrays, permutation, flip_rotations = _randomize_face_arrays(
        canonical_face_arrays,
        rng,
        allow_permutation=allow_permutation,
        allow_flip=allow_flip,
        allow_rotation=allow_rotation,
    )

    solution = _derive_solution(permutation, flip_rotations)
    validator_solution = {
        letter: (face_name.lower(), left_edge, top_edge)
        for letter, (face_name, left_edge, top_edge) in solution.items()
    }
    validator = SolutionValidator(size=size)
    if not validator.validate(observed_face_arrays, validator_solution):
        raise RuntimeError("Generated solution did not validate; this should not happen.")

    locked_placements = select_locked_placements(solution, desired_missing=missing, seed=seed)

    return {
        "faces": observed_face_arrays.astype(int).tolist(),
        "solution": _solution_json(solution),
        "locked_placements": _solution_json(locked_placements),
        "meta_info": {
            "seed": seed,
            "subset": subset,
            "missing": missing,
            "size": size,
            "canonical_face_order": list(FACES),
            "permutation": list(permutation),
            "flip_rotation": [
                {"flip": bool(flip), "rot_k": int(rot_k)}
                for flip, rot_k in flip_rotations
            ],
            "mini_cube_assignment": mini_cube_to_face,
        },
    }


