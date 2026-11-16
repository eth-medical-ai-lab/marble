"""MAZE puzzle rules, search helpers, and environment."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set, Iterable, Any
import random
import os
import contextlib
import re
import copy
import numpy as np
import io
import gymnasium as gym
from collections import deque
from maze.constants import (
    CORNER_ORIENTATIONS,
    STRATEGIC_ANCHOR_ORIENTATIONS,
    STRATEGIC_ANCHOR_POSITIONS,
    STRATEGIC_PIECE_LETTERS,
    STRATEGIC_ANCHOR_SHAPE_CHOICES,
)
from enum import Enum
from gymnasium import spaces


class Direction(Enum):
    """Compass directions for movement/connectivity; exposes dx/dy, opposite(), rotate()."""
    N = (0, -1)
    E = (1, 0)
    S = (0, 1)
    W = (-1, 0)

    @property
    def dx(self): return self.value[0]

    @property
    def dy(self): return self.value[1]

    def opposite(self):
        return {
            Direction.N: Direction.S,
            Direction.E: Direction.W,
            Direction.S: Direction.N,
            Direction.W: Direction.E,
        }[self]

    def rotate(self, quarter_turns: int):
        order = [Direction.N, Direction.E, Direction.S, Direction.W]
        i = order.index(self)
        return order[(i + (quarter_turns % 4)) % 4]


class Side(Enum):
    """Board insertion edges; provides opposite() and mapping to Direction."""
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"

    def opposite(self) -> "Side":
        return {
            Side.LEFT: Side.RIGHT,
            Side.RIGHT: Side.LEFT,
            Side.TOP: Side.BOTTOM,
            Side.BOTTOM: Side.TOP,
        }[self]

    def to_direction(self) -> Direction:
        return {
            Side.LEFT: Direction.W,
            Side.RIGHT: Direction.E,
            Side.TOP: Direction.N,
            Side.BOTTOM: Direction.S,
        }[self]

    @classmethod
    def from_str(cls, s: str) -> "Side":
        return cls(s.lower())


ODD_IDXS = (1, 3, 5)


ROTATIONS = (0, 1, 2, 3)


SIDE_ORDER: Tuple[Side, Side, Side, Side] = (Side.LEFT, Side.RIGHT, Side.TOP, Side.BOTTOM)


def decode_insertion(a: int) -> Tuple[str, int, int]:
    side_i = a // 12
    rem = a % 12
    idx_i = rem // 4
    rot = rem % 4
    side = (Side.LEFT, Side.RIGHT, Side.TOP, Side.BOTTOM)[side_i].value
    return side, ODD_IDXS[idx_i], rot


def encode_insertion(side: str, idx: int, rot: int) -> int:
    side_i = {s.value: i for i, s in enumerate((Side.LEFT, Side.RIGHT, Side.TOP, Side.BOTTOM))}[side]
    idx_i = ODD_IDXS.index(idx)
    rot_i = rot % 4
    return side_i * 12 + idx_i * 4 + rot_i


def get_opposite_side(side: str) -> str:
    opposites = {'left':'right','right':'left','top':'bottom','bottom':'top'}
    return opposites.get(side, side)


def is_forbidden_lane(prev_edit: "Edit", side: str, idx: int) -> bool:
    return side == Side.from_str(prev_edit.side).opposite().value and idx == prev_edit.idx


def bfs_path(board: "Board", start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    if start == goal:
        return [start]
    q = deque([start])
    prev: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {start: None}
    while q:
        x,y = q.popleft()
        for nx,ny in board.neighbors(x,y):
            if (nx,ny) not in prev:
                prev[(nx,ny)] = (x,y)
                if (nx,ny) == goal:
                    path = [(nx,ny)]
                    cur = (x,y)
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    path.reverse()
                    return path
                q.append((nx,ny))
    return None


@dataclass
class Tile:
    """A single tile with a shape (I, L, T) and rotation (0..3 quarter-turns clockwise)."""
    shape: str  # 'I','L','T'
    rot: int = 0  # 0..3

    def connections(self) -> Set[str]:
        base = {
            'I': {'N','S'},
            'L': {'N','E'},
            'T': {'N','E','W'},
        }[self.shape]
        return { Direction[d].rotate(self.rot).name for d in base }

    def rotate(self, steps: int=1) -> "Tile":
        self.rot = (self.rot + steps) % 4
        return self


class Board:
    """7x7 board with tiles, an extra tile, and optional objects."""
    def __init__(self, grid: List[List[Tile]], extra_tile: Tile, width:int=7, height:int=7, objects: Optional[Dict[str, Tuple[int,int]]]=None):
        self.w = width
        self.h = height
        self.grid = grid
        self.extra_tile = extra_tile
        self.objects = objects or {}

    def clone(self) -> "Board":
        grid = [[Tile(tile.shape, tile.rot) for tile in row] for row in self.grid]
        extra_tile = Tile(self.extra_tile.shape, self.extra_tile.rot)
        return Board(grid, extra_tile, self.w, self.h, dict(self.objects))

    def signature(self) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[str, int], Tuple[Tuple[str, Tuple[int, int]], ...]]:
        tiles = tuple(
            (self.grid[y][x].shape, self.grid[y][x].rot)
            for y in range(self.h)
            for x in range(self.w)
        )
        extra = (self.extra_tile.shape, self.extra_tile.rot)
        objects = tuple(sorted(self.objects.items()))
        return (tiles, extra, objects)

    def to_dict(self) -> Dict[str, object]:
        grid = [
            [
                {"shape": self.grid[y][x].shape, "rot": self.grid[y][x].rot}
                for x in range(self.w)
            ]
            for y in range(self.h)
        ]
        extra = {"shape": self.extra_tile.shape, "rot": self.extra_tile.rot}
        objects = {k: [v[0], v[1]] for k, v in self.objects.items()}
        return {"grid": grid, "extra": extra, "objects": objects}

    @staticmethod
    def random_tile(rng: random.Random) -> Tile:
        shape = rng.choices(['I','L','T'], weights=[1,2,1.5], k=1)[0]
        rot = rng.randrange(4)
        return Tile(shape, rot)

    @classmethod
    def random_board(cls, seed: Optional[int]=None, object_count: int=1, size: int = 7) -> "Board":
        rng = random.Random(seed)
        grid = [[cls.random_tile(rng) for _ in range(size)] for __ in range(size)]
        extra = cls.random_tile(rng)
        objects: Dict[str, Tuple[int,int]] = {}
        coords = [(x,y) for x in range(size) for y in range(size)]
        rng.shuffle(coords)
        labels = [chr(ord('A')+i) for i in range(max(0, min(object_count, 26)))]
        for lab, (x,y) in zip(labels, coords):
            objects[lab] = (x,y)
        return cls(grid, extra, size, size, objects)

    @classmethod
    def strategic_board(
        cls,
        seed: Optional[int]=None,
        size: int = 7,
        object_count: int = 1,
        strategic_anchor_shapes: Tuple[str, ...] = ("T",),
    ) -> "Board":
        rng = random.Random(seed)
        rotation_map = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        strategic_anchor_positions = set(STRATEGIC_ANCHOR_POSITIONS)
        anchor_shapes = tuple(dict.fromkeys(strategic_anchor_shapes))
        if not anchor_shapes:
            raise ValueError("strategic_anchor_shapes must be non-empty")
        invalid_shapes = set(anchor_shapes) - set(STRATEGIC_ANCHOR_SHAPE_CHOICES)
        if invalid_shapes:
            raise ValueError(f"invalid strategic anchor shapes: {sorted(invalid_shapes)}")
        grid = []
        for y in range(size):
            row = []
            for x in range(size):
                if (y, x) in CORNER_ORIENTATIONS:
                    rotation = rotation_map[CORNER_ORIENTATIONS[(y, x)]]
                    row.append(Tile('L', rotation))
                elif size == 7 and (y, x) in strategic_anchor_positions:
                    shape = rng.choice(anchor_shapes)
                    orientation = rng.choice(STRATEGIC_ANCHOR_ORIENTATIONS)
                    rotation = rotation_map[orientation]
                    row.append(Tile(shape, rotation))
                else:
                    row.append(None)
            grid.append(row)

        loose_pieces = []
        loose_pieces.extend(['I'] * 12)
        loose_pieces.extend(['L'] * 16)
        loose_pieces.extend(['T'] * 6)
        rng.shuffle(loose_pieces)

        loose_index = 0
        for y in range(size):
            for x in range(size):
                if grid[y][x] is None:
                    shape = loose_pieces[loose_index]
                    rotation = rng.randint(0, 3)
                    grid[y][x] = Tile(shape, rotation)
                    loose_index += 1

        leftover_count = len(loose_pieces) - loose_index
        if leftover_count >= 1:
            extra_shape = loose_pieces[loose_index]
        else:
            extra_shape = rng.choice(['I', 'L', 'T'])
        extra_rotation = rng.randint(0, 3)
        extra = Tile(extra_shape, extra_rotation)

        board = cls(grid, extra, size, size, {})

        if size == 7:
            all_objects: Dict[str, Tuple[int, int]] = {}
            strategic_positions = list(STRATEGIC_ANCHOR_POSITIONS)
            rng.shuffle(strategic_positions)
            for letter, (y, x) in zip(STRATEGIC_PIECE_LETTERS, strategic_positions):
                all_objects[letter] = (x, y)

            available_positions = []
            for y in range(7):
                for x in range(7):
                    if (y, x) not in strategic_anchor_positions:
                        available_positions.append((x, y))
            player_start = (0, 0)
            if player_start in available_positions:
                available_positions.remove(player_start)
            rng.shuffle(available_positions)
            additional_objects = ['M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
            for i, (x, y) in enumerate(available_positions[:12]):
                all_objects[additional_objects[i]] = (x, y)

            for letter in sorted(all_objects.keys())[:object_count]:
                board.objects[letter] = all_objects[letter]

        return board

    def rotate_extra(self, absolute_rot:int) -> None:
        self.extra_tile.rot = absolute_rot % 4

    def allowed_indices(self) -> List[int]:
        max_dim = max(self.w, self.h)
        return [i for i in range(max_dim) if i % 2 == 1]

    def insert(self, side: str, idx: int) -> None:
        if side not in ('left','right','top','bottom'):
            raise ValueError("side must be one of: left, right, top, bottom")
        if idx not in self.allowed_indices():
            raise ValueError("idx must be one of 1,3,5")
        if side in ('left','right') and not (0 <= idx < self.h):
            raise ValueError("row idx out of range")
        if side in ('top','bottom') and not (0 <= idx < self.w):
            raise ValueError("col idx out of range")

        ejected_objects = {}

        if side == 'left':
            row = self.grid[idx]
            ejected = row[-1]
            self.grid[idx] = [self.extra_tile] + row[:-1]
            for k, (x, y) in list(self.objects.items()):
                if y == idx and x == self.w - 1:
                    ejected_objects[k] = (x, y)
                    del self.objects[k]
            self.objects = {k: (((x+1)%7 if y==idx else x), y) for k,(x,y) in self.objects.items()}

        elif side == 'right':
            row = self.grid[idx]
            ejected = row[0]
            self.grid[idx] = row[1:] + [self.extra_tile]
            for k, (x, y) in list(self.objects.items()):
                if y == idx and x == 0:
                    ejected_objects[k] = (x, y)
                    del self.objects[k]
            self.objects = {k: (((x-1)%7 if y==idx else x), y) for k,(x,y) in self.objects.items()}

        elif side == 'top':
            col = [self.grid[y][idx] for y in range(self.h)]
            ejected = col[-1]
            newcol = [self.extra_tile] + col[:-1]
            for y in range(self.h):
                self.grid[y][idx] = newcol[y]
            for k, (x, y) in list(self.objects.items()):
                if x == idx and y == self.h - 1:
                    ejected_objects[k] = (x, y)
                    del self.objects[k]
            self.objects = {k: (x, ((y+1)%7 if x==idx else y)) for k,(x,y) in self.objects.items()}

        else:  # bottom
            col = [self.grid[y][idx] for y in range(self.h)]
            ejected = col[0]
            newcol = col[1:] + [self.extra_tile]
            for y in range(self.h):
                self.grid[y][idx] = newcol[y]
            for k, (x, y) in list(self.objects.items()):
                if x == idx and y == 0:
                    ejected_objects[k] = (x, y)
                    del self.objects[k]
            self.objects = {k: (x, ((y-1)%7 if x==idx else y)) for k,(x,y) in self.objects.items()}

        self.extra_tile = ejected

        extra_tile_objects = {}
        for k, (x, y) in list(self.objects.items()):
            if x == -1 and y == -1:
                extra_tile_objects[k] = (x, y)
                del self.objects[k]

        if side == 'left':
            for k in extra_tile_objects:
                self.objects[k] = (0, idx)
        elif side == 'right':
            for k in extra_tile_objects:
                self.objects[k] = (self.w - 1, idx)
        elif side == 'top':
            for k in extra_tile_objects:
                self.objects[k] = (idx, 0)
        else:  # bottom
            for k in extra_tile_objects:
                self.objects[k] = (idx, self.h - 1)

        for k, (x, y) in ejected_objects.items():
            self.objects[k] = (-1, -1)

    def neighbors(self, x:int, y:int) -> Iterable[Tuple[int,int]]:
        conns = self.grid[y][x].connections()
        for d_name in conns:
            d = Direction[d_name]
            nx = x + d.dx
            ny = y + d.dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                if d.opposite().name in self.grid[ny][nx].connections():
                    yield nx, ny


@dataclass
class Edit:
    side: str
    idx: int
    rotate_extra: int = 0

    def to_dict(self) -> Dict[str, int | str]:
        return {"side": self.side, "index": self.idx, "rotation": self.rotate_extra}

    @classmethod
    def from_dict(cls, data: Dict[str, int | str]) -> "Edit":
        side = data.get("side")  # type: ignore[assignment]
        idx = data.get("index", data.get("idx"))  # type: ignore[assignment]
        rot = data.get("rotation", data.get("rotate_extra", 0))  # type: ignore[assignment]
        return cls(str(side), int(idx), int(rot))


class GameState:
    def __init__(self, board: Board, players: Dict[str, Tuple[int,int]], targets: Dict[str, Tuple[int,int]]):
        self.board = board
        self.players = dict(players)
        self.targets = dict(targets)
        self.last_insert: Optional[Tuple[str,int]] = None

    def to_dict(self) -> Dict[str, object]:
        state = {
            'players': dict(self.players),
            'targets': dict(self.targets),
            'board_grid': [
                [
                    {'shape': self.board.grid[y][x].shape, 'rotation': self.board.grid[y][x].rot}
                    for x in range(self.board.w)
                ]
                for y in range(self.board.h)
            ],
            'extra_tile': {
                'shape': self.board.extra_tile.shape,
                'rotation': self.board.extra_tile.rot,
            },
            'objects': {k: {'x': v[0], 'y': v[1]} for k, v in self.board.objects.items()},
        }
        return state

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "GameState":
        if 'initial_state' in data:  # type: ignore[operator]
            data = data['initial_state']  # type: ignore[assignment]

        if 'board' in data:  # type: ignore[operator]
            b = data['board']  # type: ignore[index]
            grid = [[Tile(cell['shape'], cell['rot']) for cell in row] for row in b['grid']]  # type: ignore[index]
            extra = Tile(b['extra']['shape'], b['extra']['rot'])  # type: ignore[index]
            objects = {k: (v[0], v[1]) for k, v in b.get('objects', {}).items()}  # type: ignore[union-attr]
            board = Board(grid, extra, 7, 7, objects)
            players = {k: tuple(v) for k, v in data.get('players', {}).items()}  # type: ignore[union-attr]
            targets = {k: tuple(v) for k, v in data.get('targets', players).items()}  # type: ignore[union-attr]
            return cls(board, players, targets)

        grid = [[Tile(cell['shape'], cell['rotation']) for cell in row] for row in data['board_grid']]  # type: ignore[index]
        extra = Tile(data['extra_tile']['shape'], data['extra_tile']['rotation'])  # type: ignore[index]
        objects = {lab: (obj['x'], obj['y']) for lab, obj in data.get('objects', {}).items()}  # type: ignore[union-attr]
        board = Board(grid, extra, 7, 7, objects)
        players = {k: tuple(v) for k, v in data['players'].items()}  # type: ignore[index]
        targets = {k: tuple(v) for k, v in data.get('targets', players).items()}  # type: ignore[union-attr]
        return cls(board, players, targets)

    @classmethod
    def from_json(cls, path: str) -> "GameState":
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def _shift_player_positions_for_insert(self, side: str, idx: int) -> None:
        if side in ('left','right'):
            row = idx
            if side == 'left':
                self.players = {p: (((x+1)%7 if y==row else x), y) for p,(x,y) in self.players.items()}
            else:
                self.players = {p: (((x-1)%7 if y==row else x), y) for p,(x,y) in self.players.items()}
        else:
            col = idx
            if side == 'top':
                self.players = {p: (x, ((y+1)%7 if x==col else y)) for p,(x,y) in self.players.items()}
            else:
                self.players = {p: (x, ((y-1)%7 if x==col else y)) for p,(x,y) in self.players.items()}

    def _shift_target_positions_for_insert(self, side: str, idx: int) -> None:
        if side in ('left','right'):
            row = idx
            if side == 'left':
                self.targets = {p: (((x+1)%7 if y==row else x), y) for p,(x,y) in self.targets.items()}
            else:
                self.targets = {p: (((x-1)%7 if y==row else x), y) for p,(x,y) in self.targets.items()}
        else:
            col = idx
            if side == 'top':
                self.targets = {p: (x, ((y+1)%7 if x==col else y)) for p,(x,y) in self.targets.items()}
            else:
                self.targets = {p: (x, ((y-1)%7 if x==col else y)) for p,(x,y) in self.targets.items()}

    def apply_insert(self, side: str, idx: int, rotate_extra: int=0) -> None:
        self.board.rotate_extra(rotate_extra)
        self.board.insert(side, idx)
        self._shift_player_positions_for_insert(side, idx)
        self._shift_target_positions_for_insert(side, idx)
        self.last_insert = (side, idx)

    def single_step_move(self, player: str, insert_side: str, insert_idx: int, rotate_extra: int=0) -> Optional[List[Tuple[int,int]]]:
        self.apply_insert(insert_side, insert_idx, rotate_extra)
        start = self.players[player]
        goal = self.targets.get(player, start)
        path = bfs_path(self.board, start, goal)
        if path:
            self.players[player] = path[-1]
        return path

    def multi_edit_then_move(self, player: str, edits: List[Edit]) -> Optional[List[Tuple[int,int]]]:
        for e in edits:
            self.apply_insert(e.side, e.idx, e.rotate_extra)
        start = self.players[player]
        goal = self.targets.get(player, start)
        path = bfs_path(self.board, start, goal)
        if path:
            self.players[player] = path[-1]
        return path


def _encode_tokens(board: Board) -> np.ndarray:
    shape_to_base = {"I": 0, "L": 4, "T": 8}
    h, w = board.h, board.w
    arr = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            t = board.grid[y][x]
            arr[y, x] = shape_to_base.get(t.shape, 0) + (t.rot % 4)
    return arr


class LabyrinthEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        seed: int = 900,
        size: int = 7,
        max_turns: int = 20,
        start_pos: Tuple[int, int] = (0, 0),
        target_piece: Optional[str] = None,
        render_mode: Optional[str] = None,
        num_pieces: Optional[int] = None,
        show_harness: bool = False,
        show_guides: bool = True,
        texture_opacity: float = 0.2,
        show_coordinates: bool = False,
        random_board: bool = False,
    ) -> None:
        super().__init__()
        self.seed_val = seed
        self.size = size
        self.max_turns = max_turns
        self.start_pos = start_pos
        self.target_piece = target_piece
        self.render_mode = render_mode
        self.num_pieces = num_pieces
        self.show_harness = show_harness
        self.show_guides = show_guides
        self.texture_opacity = texture_opacity
        self.show_coordinates = show_coordinates
        self.random_board_mode = random_board

        self._move_dim = size * size + 1
        # Action: [action_type, param]
        #   action_type 0 = INSERT (param = insertion id, 0-47)
        #   action_type 1 = MOVE   (param = move target index, 0 to size*size)
        self.action_space = spaces.MultiDiscrete([2, max(48, self._move_dim)])

        self.observation_space = spaces.Dict(
            {
                "tokens": spaces.Box(low=0, high=11, shape=(size, size), dtype=np.int32),
                "player": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
                "target": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
                "extra": spaces.Box(low=0, high=11, shape=(1,), dtype=np.int32),
                "last_lane": spaces.Box(low=0, high=6, shape=(2,), dtype=np.int32),
            }
        )

        self._board: Optional[Board] = None
        self._gs: Optional[GameState] = None
        self._step_count: int = 0
        self._last_lane: Optional[Tuple[str, int]] = None
        self._target_label: Optional[str] = None

    def _build(self, seed: Optional[int]) -> None:
        board_seed = seed if seed is not None else self.seed_val
        if self.random_board_mode:
            self._board = Board.random_board(seed=board_seed, size=self.size)
        else:
            self._board = Board.strategic_board(seed=board_seed, size=self.size)

        if self.target_piece and self.target_piece in self._board.objects:
            self._target_label = self.target_piece
        else:
            self._target_label = sorted(self._board.objects.keys())[0]

        self._gs = GameState(
            self._board,
            {"P1": (int(self.start_pos[0]), int(self.start_pos[1]))},
            {"P1": self._board.objects.get(self._target_label, self.start_pos)},
        )

        self._step_count = 0
        self._last_lane = None

    def _obs(self) -> Dict[str, Any]:
        tokens = _encode_tokens(self._board)
        px, py = self._gs.players["P1"]
        tx, ty = self._board.objects.get(self._target_label, self._gs.targets["P1"])
        extra_tile = self._board.extra_tile
        shape_to_base = {"I": 0, "L": 4, "T": 8}
        extra_code = np.array([shape_to_base.get(extra_tile.shape, 0) + (extra_tile.rot % 4)], dtype=np.int32)
        if self._last_lane is None:
            last = np.array([7, 7], dtype=np.int32)
        else:
            side, idx = self._last_lane
            side_i = {s.value: i for i, s in enumerate((Side.LEFT, Side.RIGHT, Side.TOP, Side.BOTTOM))}[side]
            last = np.array([side_i, idx], dtype=np.int32)
        return {
            "tokens": tokens,
            "player": np.array([px, py], dtype=np.int32),
            "target": np.array([tx, ty], dtype=np.int32),
            "extra": extra_code,
            "last_lane": last,
        }

    def _action_mask(self) -> np.ndarray:
        mask = np.ones((48,), dtype=np.bool_)
        if self._last_lane is None:
            return mask
        prev_side, prev_idx = self._last_lane
        forbid_side = Side(prev_side).opposite().value
        for side_i, side_enum in enumerate((Side.LEFT, Side.RIGHT, Side.TOP, Side.BOTTOM)):
            side = side_enum.value
            for idx_i, idx in enumerate(ODD_IDXS):
                for rot_i in range(4):
                    a = side_i * 12 + idx_i * 4 + rot_i
                    if side == forbid_side and idx == prev_idx:
                        mask[a] = False
        return mask

    def _movement_mask(self) -> np.ndarray:
        """Compute mask of reachable positions from player's current position."""
        if self._gs is None or self._board is None:
            return np.zeros((self._move_dim,), dtype=np.bool_)
        start = self._gs.players["P1"]
        seen = {start}; q = [start]
        while q:
            x, y = q.pop(0)
            for nx, ny in self._board.neighbors(x, y):
                if (nx, ny) not in seen:
                    seen.add((nx, ny)); q.append((nx, ny))
        mask = np.zeros((self._move_dim,), dtype=np.bool_)
        for (xv, yv) in seen:
            mask[yv * self.size + xv] = True
        mask[-1] = True  # stay is always valid
        return mask

    def set_state_from_gs(self, gs: GameState, target_label: Optional[str]=None):
        self._board = copy.deepcopy(gs.board)
        self._gs = copy.deepcopy(gs)

        self._target_label = target_label or (sorted(self._board.objects.keys())[0] if self._board.objects else None)

        # Keep only the target object on the board
        if self._target_label and self._target_label in self._board.objects:
            self._board.objects = {self._target_label: self._board.objects[self._target_label]}
        else:
            self._board.objects = {}
        self._gs.board = self._board

        self._step_count = 0
        self._last_lane = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        super().reset(seed=seed)
        self._build(seed)
        return self._obs(), {
            "insertion_mask": self._action_mask(),
            "movement_mask": self._movement_mask(),
        }

    def step(self, action):
        """Execute one action: either INSERT or MOVE.

        Action format: [action_type, param]
          action_type 0 = INSERT, param = insertion id (0-47)
          action_type 1 = MOVE,   param = target position index (0..size*size-1) or size*size to stay
        Legacy: a single int is treated as INSERT with that insertion id.
        """
        assert self._gs is not None and self._board is not None

        # Parse action
        if isinstance(action, (list, tuple)) and len(action) == 2:
            action_type, param = int(action[0]), int(action[1])
        else:
            action_type, param = 0, int(action)  # legacy: single int = INSERT

        if action_type == 0:
            # ── INSERT ──
            ins_id = param
            side, idx, rot = decode_insertion(ins_id)

            # Check forbidden reversal
            if self._last_lane is not None:
                prev_side, prev_idx = self._last_lane
                forbid_side = Side(prev_side).opposite().value
                if side == forbid_side and idx == prev_idx:
                    info = {
                        "invalid": True,
                        "action_type": "insert",
                        "insertion_mask": self._action_mask(),
                        "movement_mask": self._movement_mask(),
                        "insertion": {"side": side, "idx": idx, "rot": rot},
                        "feedback": (
                            f"This is an invalid action. You cannot insert from the opposite side "
                            f"(side={side}, slot={idx}) of the previous insertion. "
                            f"The board remains unchanged. Please try a different action."
                        ),
                    }
                    truncated = self._step_count >= self.max_turns
                    return self._obs(), 0, False, truncated, info

            # Execute insertion
            with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
                self._gs.apply_insert(side, idx, rot)
            self._last_lane = (side, idx)
            self._step_count += 1

            truncated = self._step_count >= self.max_turns
            info = {
                "action_type": "insert",
                "insertion_mask": self._action_mask(),
                "movement_mask": self._movement_mask(),
                "insertion": {"side": side, "idx": idx, "rot": rot},
            }
            return self._obs(), 0, False, truncated, info

        else:
            # ── MOVE ──
            move_idx = param

            # Compute reachable positions
            move_mask_now = self._movement_mask()

            if move_idx == self._move_dim - 1:
                endpoint = None
            else:
                x = move_idx % self.size; y = move_idx // self.size
                endpoint = (x, y)

            # Validate movement target
            if not move_mask_now[move_idx]:
                info = {
                    "invalid": True,
                    "action_type": "move",
                    "insertion_mask": self._action_mask(),
                    "movement_mask": move_mask_now,
                    "movement": {"moved": False, "path_len": 0, "end": None, "invalid_movement": True},
                    "feedback": (
                        "This is an invalid action. The target position is not reachable "
                        "from the current position. The player remains in place. "
                        "Please try a different action."
                    ),
                }
                truncated = self._step_count >= self.max_turns
                return self._obs(), 0, False, truncated, info

            # Execute movement
            start = self._gs.players["P1"]
            path = [start]
            invalid_movement = False
            if endpoint is not None:
                path_candidate = bfs_path(self._board, start, endpoint)
                if path_candidate and len(path_candidate) > 1:
                    path = path_candidate
                    self._gs.players["P1"] = path[-1]
                else:
                    invalid_movement = True

            self._step_count += 1

            # Check goal
            goal = self._board.objects.get(self._target_label, self._gs.targets["P1"])
            reached = (self._gs.players["P1"] == goal)
            terminated = bool(reached)
            truncated = bool(self._step_count >= self.max_turns and not terminated)
            reward = 1.0 if reached else 0.0

            info = {
                "action_type": "move",
                "insertion_mask": self._action_mask(),
                "movement_mask": self._movement_mask(),
                "movement": {
                    "moved": (endpoint is not None and len(path) > 1),
                    "path_len": len(path) if path else 0,
                    "end": (path[-1] if path else None),
                    "invalid_movement": invalid_movement,
                },
            }
            return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self._board is None or self._gs is None:
            return None
        # Lazy import rendering and optional PNG conversion deps
        from maze.rendering import render_board
        svg = render_board(self._board, players=self._gs.players, show_arrows=True, cell=80, margin=30,
                           target_piece=self._target_label, show_harness=self.show_harness,
                           show_guides=self.show_guides, texture_opacity=self.texture_opacity,
                           show_coordinates=self.show_coordinates)

        if self.render_mode == "human":
            out = os.path.join("visualizations", "env_render.svg")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.write(svg)
            return out

        if self.render_mode == "rgb_array":
            try:
                import cairosvg
                from PIL import Image
            except Exception:
                return svg
            vb = re.search(r'viewBox="\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)"', svg)
            if vb:
                wpx = float(vb.group(1)); hpx = float(vb.group(2))
                tgt_min = float(os.getenv("PNG_MIN_DIM", "896"))
                base_min = min(wpx, hpx) if (wpx > 0 and hpx > 0) else 1.0
                s = max(1e-6, tgt_min / base_min)
                out_w = int(round(wpx * s))
                out_h = int(round(hpx * s))
                png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=out_w, output_height=out_h)
            else:
                scale = float(os.getenv("PNG_SCALE", "1"))
                png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"), scale=scale)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            return np.array(img)

        return svg

    def step_edit(self, edit: Edit):
        a = encode_insertion(edit.side, edit.idx, edit.rotate_extra)
        return self.step([0, a])  # INSERT action

    def replay_edits(self, edits: list[Edit]):
        last = None
        for e in edits:
            obs, reward, terminated, truncated, info = self.step_edit(e)
            last = (obs, reward, terminated, truncated, info)
            if terminated or truncated:
                break
        return last

    def set_target_piece(self, label: str) -> bool:
        if self._board is None or self._gs is None:
            return False
        if label in self._board.objects:
            self._target_label = label
            self._gs.targets["P1"] = self._board.objects[label]
            return True
        return False

    def shortest_path(self, start: Optional[Tuple[int, int]] = None, goal: Optional[Tuple[int, int]] = None):
        if self._board is None or self._gs is None:
            return None
        s = start or self._gs.players.get("P1")
        if s is None:
            return None
        if goal is None:
            goal = self._board.objects.get(self._target_label, self._gs.targets.get("P1", s))
        return bfs_path(self._board, s, goal)

    def reachable_positions(self, from_pos: Optional[Tuple[int, int]] = None) -> set:
        if self._board is None or self._gs is None:
            return set()
        start = from_pos or self._gs.players.get("P1")
        if start is None:
            return set()
        seen = {start}
        queue = [start]
        while queue:
            x, y = queue.pop(0)
            for nx, ny in self._board.neighbors(x, y):
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    queue.append((nx, ny))
        return seen

    def preview_action_outcomes(self, masked: bool = True) -> list:
        if self._board is None or self._gs is None:
            return []
        results = []
        mask = self._action_mask() if masked else None
        for a in range(48):
            if mask is not None and not mask[a]:
                continue
            side, idx, rot = decode_insertion(a)
            bcopy = copy.deepcopy(self._board)
            gsc = GameState(bcopy, dict(self._gs.players), dict(self._gs.targets))
            with open(os.devnull, "w") as _dn, contextlib.redirect_stdout(_dn):
                gsc.apply_insert(side, idx, rot)
            goal = gsc.board.objects.get(self._target_label, gsc.targets.get("P1", gsc.players.get("P1")))
            path = bfs_path(gsc.board, gsc.players["P1"], goal)
            results.append({
                "action": a,
                "side": side,
                "idx": idx,
                "rot": rot,
                "endpoint": (path[-1] if path else None),
                "path_len": (len(path) if path else None),
            })
        return results

    def close(self):
        pass


def make_env(**kwargs) -> LabyrinthEnv:
    return LabyrinthEnv(**kwargs)
