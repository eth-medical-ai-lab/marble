"""BLOX puzzle rules, validation, and environment."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from blox.constants import Orientation, Action, TileType, MOVEMENT_TABLE, ACTION_NAMES
import json
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass(frozen=True)
class Block:
    """A 1x1x2 block on the grid. Anchor is the top-left occupied cell."""
    r: int
    c: int
    orientation: Orientation

    def cells(self) -> List[Tuple[int, int]]:
        """Return list of grid cells occupied by this block."""
        if self.orientation == Orientation.STANDING:
            return [(self.r, self.c)]
        elif self.orientation == Orientation.LYING_H:
            return [(self.r, self.c), (self.r, self.c + 1)]
        else:  # LYING_V
            return [(self.r, self.c), (self.r + 1, self.c)]

    def move(self, action: Action) -> Block:
        """Return a new Block after applying the given action."""
        new_orient, dr, dc = MOVEMENT_TABLE[(self.orientation, action)]
        return Block(self.r + dr, self.c + dc, new_orient)

    def signature(self) -> Tuple[int, int, int]:
        return (self.r, self.c, int(self.orientation))


@dataclass
class Board:
    """Grid of tiles. grid[r][c] is a TileType."""
    rows: int
    cols: int
    grid: List[List[int]] = field(default_factory=list)
    switches: Dict[Tuple[int, int], int] = field(default_factory=dict)  # (r,c) -> group_id
    bridges: Dict[Tuple[int, int], int] = field(default_factory=dict)   # (r,c) -> group_id
    hard_switches: set = field(default_factory=set)  # set of (r,c) that require STANDING to activate
    height_map: List[List[int]] = field(default_factory=list)  # 0=LOW, 1=HIGH per cell
    elevators: set = field(default_factory=set)  # set of (r,c) that are elevator tiles

    def tile_at(self, r: int, c: int) -> int:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c]
        return TileType.VOID

    def effective_tile_at(
        self,
        r: int,
        c: int,
        collapsed_crumble: frozenset = frozenset(),
    ) -> int:
        """Return the walkable tile at a cell after dynamic crumble collapse."""
        tile = self.tile_at(r, c)
        if tile == TileType.CRUMBLE and (r, c) in collapsed_crumble:
            return TileType.VOID
        return tile

    def effective_grid(self, collapsed_crumble: frozenset = frozenset()) -> List[List[int]]:
        """Return a grid snapshot with collapsed CRUMBLE tiles converted to VOID."""
        return [
            [self.effective_tile_at(r, c, collapsed_crumble) for c in range(self.cols)]
            for r in range(self.rows)
        ]

    def has_tile_type(self, tile_type: TileType) -> bool:
        return any(
            self.grid[row][col] == tile_type
            for row in range(self.rows)
            for col in range(self.cols)
        )

    def cell_height(self, r: int, c: int) -> int:
        """Return the height level of a cell (0=LOW, 1=HIGH). Default 0."""
        if self.height_map and 0 <= r < self.rows and 0 <= c < self.cols:
            return self.height_map[r][c]
        return 0

    def is_valid_position(
        self,
        block: Block,
        active_bridges: frozenset = frozenset(),
        current_height: int = 0,
        collapsed_crumble: frozenset = frozenset(),
    ) -> bool:
        """Check if all cells occupied by block are on walkable tiles.

        BRIDGE tiles are only walkable when their group is in active_bridges.
        FRAGILE tiles collapse under a STANDING block (only safe when LYING).
        CRUMBLE tiles are safe while occupied, but become VOID after the block leaves.
        Height check: all occupied cells must be at current_height, unless
        the cell is an ELEVATOR tile (accessible at both heights).
        """
        for r, c in block.cells():
            tile = self.effective_tile_at(r, c, collapsed_crumble)
            if tile == TileType.VOID:
                return False
            if tile == TileType.BRIDGE:
                group = self.bridges.get((r, c))
                if group is None or group not in active_bridges:
                    return False
            if tile == TileType.FRAGILE and block.orientation == Orientation.STANDING:
                return False
            # Height check: cell must be at block's current height,
            # unless it's an elevator tile (accessible from both levels)
            if self.height_map and (r, c) not in self.elevators:
                if self.cell_height(r, c) != current_height:
                    return False
        return True

    def to_dict(self) -> Dict:
        d = {
            "rows": self.rows,
            "cols": self.cols,
            "grid": [row[:] for row in self.grid],
        }
        if self.switches:
            d["switches"] = [
                {"r": r, "c": c, "group": g, **({"hard": True} if (r, c) in self.hard_switches else {})}
                for (r, c), g in self.switches.items()
            ]
        if self.bridges:
            d["bridges"] = [{"r": r, "c": c, "group": g} for (r, c), g in self.bridges.items()]
        if self.height_map:
            d["height_map"] = [row[:] for row in self.height_map]
        if self.elevators:
            d["elevators"] = [{"r": r, "c": c} for r, c in sorted(self.elevators)]
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> Board:
        rows = data["rows"]
        cols = data["cols"]
        grid = [row[:] for row in data["grid"]]
        switches = {}
        hard_switches = set()
        for s in data.get("switches", []):
            switches[(s["r"], s["c"])] = s["group"]
            if s.get("hard", False):
                hard_switches.add((s["r"], s["c"]))
        bridges = {}
        for b in data.get("bridges", []):
            bridges[(b["r"], b["c"])] = b["group"]
        height_map = [row[:] for row in data["height_map"]] if "height_map" in data else []
        elevators = set()
        for e in data.get("elevators", []):
            elevators.add((e["r"], e["c"]))
        return cls(rows, cols, grid, switches, bridges, hard_switches, height_map, elevators)

    @classmethod
    def full_board(cls, rows: int, cols: int) -> Board:
        """Create a board where all cells are NORMAL."""
        grid = [[TileType.NORMAL] * cols for _ in range(rows)]
        return cls(rows, cols, grid)


@dataclass
class GameState:
    """Full game state: board + block + goal + bridge configuration."""
    board: Board
    block: Block
    goal_r: int
    goal_c: int
    active_bridges: frozenset = field(default_factory=frozenset)
    current_height: int = 0  # block's current height level (0=LOW, 1=HIGH)
    collapsed_crumble: frozenset[Tuple[int, int]] = field(default_factory=frozenset)
    last_collapsed_crumble: Tuple[Tuple[int, int], ...] = field(
        default_factory=tuple,
        init=False,
        repr=False,
        compare=False,
    )

    def clone(self) -> "GameState":
        """Copy dynamic state while reusing the immutable board description."""
        cloned = GameState(
            board=self.board,
            block=self.block,
            goal_r=self.goal_r,
            goal_c=self.goal_c,
            active_bridges=self.active_bridges,
            current_height=self.current_height,
            collapsed_crumble=self.collapsed_crumble,
        )
        cloned.last_collapsed_crumble = self.last_collapsed_crumble
        return cloned

    def apply_action(self, action: Action) -> Tuple[bool, bool]:
        """
        Apply an action. Returns (valid, won).
        If invalid (block falls off), state is NOT mutated.
        """
        old_cells = set(self.block.cells())
        self.last_collapsed_crumble = ()
        new_block = self.block.move(action)
        if not self.board.is_valid_position(
            new_block,
            self.active_bridges,
            self.current_height,
            self.collapsed_crumble,
        ):
            return False, False
        self.block = new_block
        self._collapse_crumble(old_cells, set(new_block.cells()))
        self._check_switches()
        self._check_elevators()
        won = self.is_won()
        return True, won

    def _collapse_crumble(
        self,
        previous_cells: set[Tuple[int, int]],
        current_cells: set[Tuple[int, int]],
    ) -> None:
        """Collapse CRUMBLE tiles that were fully vacated by the latest move."""
        collapsed = tuple(sorted(
            (r, c)
            for r, c in previous_cells - current_cells
            if self.board.tile_at(r, c) == TileType.CRUMBLE
            and (r, c) not in self.collapsed_crumble
        ))
        self.last_collapsed_crumble = collapsed
        if collapsed:
            self.collapsed_crumble = self.collapsed_crumble.union(collapsed)

    def _check_switches(self):
        """Toggle bridge groups for any switches the block is touching.

        Hard switches only activate when the block is STANDING on them.
        Soft switches activate on any contact (standing or lying).
        """
        if not self.board.switches:
            return
        groups_to_toggle = set()
        for r, c in self.block.cells():
            if (r, c) in self.board.switches:
                if (r, c) in self.board.hard_switches and self.block.orientation != Orientation.STANDING:
                    continue  # hard switch requires standing
                groups_to_toggle.add(self.board.switches[(r, c)])
        if groups_to_toggle:
            self.active_bridges = self.active_bridges.symmetric_difference(groups_to_toggle)

    def _check_elevators(self):
        """Toggle height if the block is STANDING on an elevator tile.

        Only activates when the block is upright (standing) on the elevator.
        Then re-validate at the new height; if invalid, revert.
        """
        if not self.board.elevators:
            return
        on_elevator = (self.block.orientation == Orientation.STANDING
                       and (self.block.r, self.block.c) in self.board.elevators)
        if on_elevator:
            new_height = 1 - self.current_height
            # Validate position at new height
            if self.board.is_valid_position(
                self.block,
                self.active_bridges,
                new_height,
                self.collapsed_crumble,
            ):
                self.current_height = new_height
            # If invalid at new height, block stays at old height (elevator doesn't activate)

    def is_won(self) -> bool:
        if not (self.block.orientation == Orientation.STANDING
                and self.block.r == self.goal_r
                and self.block.c == self.goal_c):
            return False
        # Height check: block must be at the same height as the goal cell
        if self.board.height_map:
            goal_height = self.board.cell_height(self.goal_r, self.goal_c)
            if self.current_height != goal_height:
                return False
        return True

    def is_valid(self) -> bool:
        return self.board.is_valid_position(
            self.block,
            self.active_bridges,
            self.current_height,
            self.collapsed_crumble,
        )

    def signature(self) -> tuple:
        """Hashable signature for BFS dedup. Includes bridge state and height when relevant."""
        base = self.block.signature()
        has_extra = (
            self.board.switches
            or self.board.elevators
            or self.board.has_tile_type(TileType.CRUMBLE)
        )
        if has_extra:
            parts = list(base)
            if self.board.switches:
                parts.append(self.active_bridges)
            if self.board.elevators:
                parts.append(self.current_height)
            if self.board.has_tile_type(TileType.CRUMBLE):
                parts.append(self.collapsed_crumble)
            return tuple(parts)
        return base

    def visible_grid(self) -> List[List[int]]:
        """Return the current grid as seen by the agent after dynamic updates."""
        return self.board.effective_grid(self.collapsed_crumble)

    def to_dict(self) -> Dict:
        d = {
            "board": self.board.to_dict(),
            "block": {"r": self.block.r, "c": self.block.c, "orientation": int(self.block.orientation)},
            "goal": {"r": self.goal_r, "c": self.goal_c},
        }
        if self.active_bridges:
            d["active_bridges"] = sorted(self.active_bridges)
        if self.current_height != 0:
            d["current_height"] = self.current_height
        if self.collapsed_crumble:
            d["collapsed_crumble"] = [
                {"r": r, "c": c} for r, c in sorted(self.collapsed_crumble)
            ]
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> GameState:
        board = Board.from_dict(data["board"])
        b = data["block"]
        block = Block(b["r"], b["c"], Orientation(b["orientation"]))
        g = data["goal"]
        active = frozenset(data.get("active_bridges", []))
        current_height = data.get("current_height", 0)
        collapsed_crumble = frozenset(
            (cell["r"], cell["c"]) for cell in data.get("collapsed_crumble", [])
        )
        return cls(board, block, g["r"], g["c"], active, current_height, collapsed_crumble)


def bfs_solve(state: GameState) -> Optional[List[Action]]:
    """
    BFS over (r, c, orientation[, active_bridges][, current_height][, collapsed_crumble]) states.
    Returns shortest action sequence to win, or None if unsolvable.
    """
    if state.is_won():
        return []

    board = state.board
    switches = board.switches
    hard_switches = board.hard_switches
    elevators = board.elevators

    has_switches = bool(switches)
    has_elevators = bool(elevators)
    has_height_map = bool(board.height_map)
    has_crumble = board.has_tile_type(TileType.CRUMBLE)

    def _make_sig(block, active_bridges, current_height, collapsed_crumble):
        sig = block.signature()
        if has_switches:
            sig += (active_bridges,)
        if has_elevators:
            sig += (current_height,)
        if has_crumble:
            sig += (collapsed_crumble,)
        return sig

    def _check_win(block, current_height):
        if not (block.orientation == Orientation.STANDING
                and block.r == state.goal_r
                and block.c == state.goal_c):
            return False
        if has_height_map:
            goal_height = board.cell_height(state.goal_r, state.goal_c)
            if current_height != goal_height:
                return False
        return True

    def _reconstruct_path(parents, end_sig):
        path: List[Action] = []
        cur_sig = end_sig
        while True:
            parent_sig, action = parents[cur_sig]
            if parent_sig is None or action is None:
                path.reverse()
                return path
            path.append(action)
            cur_sig = parent_sig

    start_sig = _make_sig(
        state.block,
        state.active_bridges,
        state.current_height,
        state.collapsed_crumble,
    )
    visited = {start_sig}
    parents: Dict[tuple, Tuple[Optional[tuple], Optional[Action]]] = {
        start_sig: (None, None)
    }
    # Queue entries: (sig, block, active_bridges, current_height, collapsed_crumble)
    queue = deque([
        (
            start_sig,
            state.block,
            state.active_bridges,
            state.current_height,
            state.collapsed_crumble,
        )
    ])

    while queue:
        parent_sig, block, active_bridges, current_height, collapsed_crumble = queue.popleft()
        for action in Action:
            new_block = block.move(action)
            new_cells = new_block.cells()
            if not board.is_valid_position(
                new_block,
                active_bridges,
                current_height,
                collapsed_crumble,
            ):
                continue

            new_collapsed = collapsed_crumble
            if has_crumble:
                vacated_crumble = [
                    (r, c)
                    for r, c in block.cells()
                    if (r, c) not in new_cells
                    and board.tile_at(r, c) == TileType.CRUMBLE
                    and (r, c) not in collapsed_crumble
                ]
                if vacated_crumble:
                    new_collapsed = collapsed_crumble.union(vacated_crumble)

            # Compute switch toggles
            new_active = active_bridges
            if has_switches:
                groups_to_toggle = set()
                for r, c in new_cells:
                    if (r, c) in switches:
                        if (r, c) in hard_switches and new_block.orientation != Orientation.STANDING:
                            continue  # hard switch requires standing
                        groups_to_toggle.add(switches[(r, c)])
                if groups_to_toggle:
                    new_active = active_bridges.symmetric_difference(groups_to_toggle)

            # Compute elevator toggle (must be STANDING on elevator)
            new_height = current_height
            if has_elevators:
                on_elevator = (new_block.orientation == Orientation.STANDING
                               and (new_block.r, new_block.c) in elevators)
                if on_elevator:
                    candidate_height = 1 - current_height
                    if board.is_valid_position(
                        new_block,
                        new_active,
                        candidate_height,
                        new_collapsed,
                    ):
                        new_height = candidate_height

            sig = _make_sig(new_block, new_active, new_height, new_collapsed)
            if sig in visited:
                continue
            visited.add(sig)
            parents[sig] = (parent_sig, action)
            # Check win
            if _check_win(new_block, new_height):
                return _reconstruct_path(parents, sig)
            queue.append((sig, new_block, new_active, new_height, new_collapsed))

    return None


class BloxEnv(gym.Env):
    """
    Blox rolling-block puzzle environment.

    Action space: Discrete(4) — UP, DOWN, LEFT, RIGHT
    Observation space: Dict with grid, block_pos, goal
    """
    metadata = {"render_modes": ["human", "rgb_array", "svg"], "render_fps": 4}

    def __init__(
        self,
        level: Optional[Dict] = None,
        level_path: Optional[str] = None,
        max_steps: int = 50,
        render_mode: Optional[str] = None,
        show_coordinates: bool = True,
        show_legend: bool = True,
    ) -> None:
        super().__init__()
        self._level_data = level
        self._level_path = level_path
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.show_coordinates = show_coordinates
        self.show_legend = show_legend

        self.action_space = spaces.Discrete(4)

        # Will be set on reset
        self._state: Optional[GameState] = None
        self._step_count: int = 0
        self._rows: int = 0
        self._cols: int = 0

        # Set up observation space with placeholder dims; updated on reset
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=int(max(TileType)), shape=(1, 1), dtype=np.int32),
            "block_pos": spaces.Box(low=0, high=255, shape=(3,), dtype=np.int32),
            "goal": spaces.Box(low=0, high=255, shape=(2,), dtype=np.int32),
        })

    def _load_level(self) -> Dict:
        if self._level_data is not None:
            return self._level_data
        if self._level_path is not None:
            with open(self._level_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError("No level data or path provided")

    def _build_state(self, data: Dict) -> GameState:
        return GameState.from_dict(data)

    def _obs(self) -> Dict[str, np.ndarray]:
        assert self._state is not None
        grid = np.array(self._state.visible_grid(), dtype=np.int32)
        block_pos = np.array(
            [self._state.block.r, self._state.block.c, int(self._state.block.orientation)],
            dtype=np.int32,
        )
        goal = np.array([self._state.goal_r, self._state.goal_c], dtype=np.int32)
        obs = {"grid": grid, "block_pos": block_pos, "goal": goal}
        if self._state.board.elevators:
            obs["current_height"] = np.array([self._state.current_height], dtype=np.int32)
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        data = self._load_level()
        self._state = self._build_state(data)
        self._rows = self._state.board.rows
        self._cols = self._state.board.cols
        self._step_count = 0

        # Update observation space to match actual grid
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0,
                high=int(max(TileType)),
                shape=(self._rows, self._cols),
                dtype=np.int32,
            ),
            "block_pos": spaces.Box(low=0, high=max(self._rows, self._cols), shape=(3,), dtype=np.int32),
            "goal": spaces.Box(low=0, high=max(self._rows, self._cols), shape=(2,), dtype=np.int32),
        })

        info = {
            "target_depth": data.get("target_depth"),
            "solution": data.get("solution"),
            "step_count": 0,
        }
        return self._obs(), info

    def set_state(self, state: GameState) -> None:
        """Set the environment state directly."""
        self._state = state
        self._rows = state.board.rows
        self._cols = state.board.cols
        self._step_count = 0

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        assert self._state is not None
        action_enum = Action(int(action))

        valid, won = self._state.apply_action(action_enum)
        collapsed_crumble = [list(cell) for cell in self._state.last_collapsed_crumble]
        self._step_count += 1

        reward = 0.0
        terminated = False
        truncated = False

        if not valid:
            # Block would fall off — invalid action, state unchanged, episode continues
            truncated = self._step_count >= self.max_steps
            info = {
                "action": ACTION_NAMES[action_enum],
                "valid": False,
                "won": False,
                "fell": True,
                "step_count": self._step_count,
                "active_bridges": sorted(self._state.active_bridges),
                "collapsed_crumble": collapsed_crumble,
                "feedback": f"This is an invalid action. Moving {ACTION_NAMES[action_enum]} would cause the block to fall off the board. The block remains in its previous position. Please try a different move.",
            }
        elif won:
            reward = 1.0
            terminated = True
            info = {
                "action": ACTION_NAMES[action_enum],
                "valid": True,
                "won": True,
                "fell": False,
                "step_count": self._step_count,
                "active_bridges": sorted(self._state.active_bridges),
                "collapsed_crumble": collapsed_crumble,
            }
        else:
            dead_end = False
            if collapsed_crumble:
                # Exact oracle check used only after irreversible board damage.
                dead_end = bfs_solve(self._state.clone()) is None
            terminated = dead_end
            truncated = not dead_end and self._step_count >= self.max_steps
            info = {
                "action": ACTION_NAMES[action_enum],
                "valid": True,
                "won": False,
                "fell": False,
                "dead_end": dead_end,
                "step_count": self._step_count,
                "active_bridges": sorted(self._state.active_bridges),
                "collapsed_crumble": collapsed_crumble,
            }
            if dead_end:
                info["failure_reason"] = "unreachable_after_crumble"
                info["feedback"] = (
                    "A crumble tile collapsed and the goal is no longer reachable "
                    "from the current state. The episode stops as failed."
                )

        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self._state is None:
            return None

        from blox.rendering import render_board_svg, render_to_png_array

        svg = render_board_svg(
            self._state.board,
            block=self._state.block,
            goal_r=self._state.goal_r,
            goal_c=self._state.goal_c,
            active_bridges=self._state.active_bridges,
            show_coordinates=self.show_coordinates,
            show_legend=self.show_legend,
            current_height=self._state.current_height,
            collapsed_crumble=self._state.collapsed_crumble,
        )

        if self.render_mode == "human":
            out = os.path.join("visualizations", "blox_render.svg")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.write(svg)
            return out

        if self.render_mode == "rgb_array":
            return render_to_png_array(svg)

        # "svg" or default
        return svg

    def close(self):
        pass


def make_env(**kwargs) -> BloxEnv:
    return BloxEnv(**kwargs)


def has_fragile(board: Board) -> bool:
    return any(board.grid[row][col] == TileType.FRAGILE for row in range(board.rows) for col in range(board.cols))


def has_crumble(board: Board) -> bool:
    return any(board.grid[row][col] == TileType.CRUMBLE for row in range(board.rows) for col in range(board.cols))


def validate_solution(state: GameState, actions: List[Action]) -> Tuple[bool, int, str]:
    """
    Validate a sequence of actions against a game state.

    Returns (success, steps_completed, message).
    Does NOT mutate the input state.
    """
    board = state.board
    block = state.block
    goal_r, goal_c = state.goal_r, state.goal_c
    active_bridges = state.active_bridges
    current_height = state.current_height
    collapsed_crumble = state.collapsed_crumble
    has_elevators = bool(board.elevators)
    has_height_map = bool(board.height_map)

    for i, action in enumerate(actions):
        new_block = block.move(action)
        if not board.is_valid_position(new_block, active_bridges, current_height, collapsed_crumble):
            return False, i, f"Step {i + 1}: {ACTION_NAMES[action]} causes block to fall off the board"

        vacated = frozenset(
            (r, c)
            for r, c in set(block.cells()) - set(new_block.cells())
            if board.tile_at(r, c) == TileType.CRUMBLE and (r, c) not in collapsed_crumble
        )
        if vacated:
            collapsed_crumble = collapsed_crumble.union(vacated)
        block = new_block

        if board.switches:
            groups_to_toggle = set()
            for r, c in block.cells():
                if (r, c) in board.switches:
                    if (r, c) in board.hard_switches and block.orientation != Orientation.STANDING:
                        continue
                    groups_to_toggle.add(board.switches[(r, c)])
            if groups_to_toggle:
                active_bridges = active_bridges.symmetric_difference(groups_to_toggle)

        if has_elevators:
            on_elevator = block.orientation == Orientation.STANDING and (block.r, block.c) in board.elevators
            if on_elevator:
                new_height = 1 - current_height
                if board.is_valid_position(block, active_bridges, new_height, collapsed_crumble):
                    current_height = new_height

        won = block.orientation == Orientation.STANDING and block.r == goal_r and block.c == goal_c
        if won and has_height_map:
            goal_height = board.cell_height(goal_r, goal_c)
            if current_height != goal_height:
                won = False
        if won:
            return True, i + 1, f"Goal reached in {i + 1} steps"

    won = block.orientation == Orientation.STANDING and block.r == goal_r and block.c == goal_c
    if won and has_height_map:
        goal_height = board.cell_height(goal_r, goal_c)
        if current_height != goal_height:
            won = False
    if won:
        return True, len(actions), f"Goal reached in {len(actions)} steps"

    return False, len(actions), f"Goal not reached after {len(actions)} steps. Block at ({block.r},{block.c}) orientation={block.orientation.name}"
