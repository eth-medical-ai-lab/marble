"""One-elevator crumble generation and candidate ranking for the benchmark."""
from __future__ import annotations
import copy
import json
import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple
from blox.constants import Action, Orientation, TileType
from blox.env import Block, Board, GameState, bfs_solve

_SIMPLE_CRUMBLE_MODES = [
    "path",
    "mixed",
    "random",
    "pre_elevator",
    "post_elevator",
    "goal_approach",
    "elevator_buffer",
    "mid_path",
]
_SIMPLE_CRUMBLE_BASE_ATTEMPTS = 4
_SIMPLE_CRUMBLE_MODE_ATTEMPTS = 2
_SIMPLE_CRUMBLE_FORCED_MODE_ATTEMPTS = 4


def _trace_solution_states(level: Dict, solution: List[int]) -> Optional[List[GameState]]:
    """Replay a solution and return the state after each step, including start."""
    state = GameState.from_dict(level)
    states = [state.clone()]
    for raw_action in solution:
        valid, _ = state.apply_action(Action(raw_action))
        if not valid:
            return None
        states.append(state.clone())
    return states


def _eligible_crumble_cells(state: GameState) -> set[Tuple[int, int]]:
    """Return walkable NORMAL cells that are safe to replace with CRUMBLE."""
    start_cells = set(state.block.cells())
    goal_cell = (state.goal_r, state.goal_c)
    return {
        (r, c)
        for r in range(state.board.rows)
        for c in range(state.board.cols)
        if state.board.grid[r][c] == TileType.NORMAL
        and (r, c) not in start_cells
        and (r, c) != goal_cell
        and (r, c) not in state.board.elevators
    }


def _neighbors4(r: int, c: int) -> List[Tuple[int, int]]:
    """Return the 4-neighborhood of a cell."""
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _union_cell_sets(cell_sets: List[set[Tuple[int, int]]]) -> set[Tuple[int, int]]:
    """Union helper that tolerates empty input."""
    merged: set[Tuple[int, int]] = set()
    for cells in cell_sets:
        merged.update(cells)
    return merged


def _first_elevator_toggle_step(states: List[GameState]) -> Optional[int]:
    """Return the first solution step index that changes height."""
    for step_idx in range(1, len(states)):
        if states[step_idx].current_height != states[step_idx - 1].current_height:
            return step_idx
    return None


def _reachable_cells_from_state(initial_state: GameState) -> set[Tuple[int, int]]:
    """Collect all board cells reachable from a state under full gameplay rules."""
    start_state = initial_state.clone()
    visited = {start_state.signature()}
    queue = deque([start_state])
    reachable = set(start_state.block.cells())

    while queue:
        state = queue.popleft()
        for action in Action:
            next_state = state.clone()
            valid, _ = next_state.apply_action(action)
            if not valid:
                continue
            sig = next_state.signature()
            if sig in visited:
                continue
            visited.add(sig)
            reachable.update(next_state.block.cells())
            queue.append(next_state)

    return reachable


def _build_simple_crumble_context(base_level: Dict) -> Optional[Dict]:
    """Pre-compute path- and geometry-derived cell pools for crumble placement."""
    base_state = GameState.from_dict(base_level)
    solution = base_level.get("solution", [])
    states = _trace_solution_states(base_level, solution)
    if states is None:
        return None

    eligible = _eligible_crumble_cells(base_state)
    if not eligible:
        return None

    step_cells = [set(state.block.cells()) for state in states]
    path_cells = eligible.intersection(_union_cell_sets(step_cells))
    toggle_step = _first_elevator_toggle_step(states)
    if toggle_step is None:
        toggle_step = max(1, (len(step_cells) - 1) // 2)

    pre_cells = eligible.intersection(_union_cell_sets(step_cells[:max(1, toggle_step)]))
    post_cells = eligible.intersection(_union_cell_sets(step_cells[toggle_step:]))
    goal_cells = eligible.intersection(_union_cell_sets(step_cells[max(0, len(step_cells) - 5):]))

    mid_lo = max(0, len(step_cells) // 3 - 1)
    mid_hi = min(len(step_cells), (2 * len(step_cells)) // 3 + 1)
    mid_cells = eligible.intersection(_union_cell_sets(step_cells[mid_lo:mid_hi]))
    off_path_cells = eligible - path_cells
    fringe_off_path = {
        cell
        for cell in off_path_cells
        if any(neighbor in path_cells for neighbor in _neighbors4(*cell))
    }

    elevator_buffer = {
        cell
        for cell in eligible
        if any(abs(cell[0] - er) + abs(cell[1] - ec) <= 2 for er, ec in base_state.board.elevators)
    }

    mode_pools = {
        "path": (path_cells, path_cells),
        "mixed": (eligible, path_cells or eligible),
        "random": (eligible, eligible),
        "pre_elevator": (pre_cells or path_cells, pre_cells or path_cells),
        "post_elevator": (post_cells or path_cells, post_cells or path_cells),
        "goal_approach": (goal_cells or path_cells, goal_cells or path_cells),
        "elevator_buffer": (elevator_buffer or path_cells, elevator_buffer or path_cells),
        "mid_path": (mid_cells or path_cells, mid_cells or path_cells),
    }

    return {
        "eligible": eligible,
        "path_cells": path_cells,
        "off_path_cells": off_path_cells,
        "fringe_off_path_cells": fringe_off_path,
        "board_area": base_state.board.rows * base_state.board.cols,
        "num_elevators": len(base_state.board.elevators),
        "states": states,
        "step_cells": step_cells,
        "mode_pools": mode_pools,
    }


def _sample_connected_region(
    allowed_cells: set[Tuple[int, int]],
    rng: random.Random,
    size: int,
    seed_cells: Optional[set[Tuple[int, int]]] = None,
    max_attempts: int = 32,
) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Sample a connected region of a given size from *allowed_cells*."""
    if size <= 0 or not allowed_cells:
        return None

    seed_pool = list(seed_cells or allowed_cells)
    if not seed_pool:
        return None

    for _ in range(max_attempts):
        region = {rng.choice(seed_pool)}
        while len(region) < size:
            frontier = []
            for r, c in region:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in allowed_cells and (nr, nc) not in region:
                        frontier.append((nr, nc))
            if not frontier:
                break
            region.add(rng.choice(frontier))
        if len(region) == size:
            return tuple(sorted(region))
    return None


def _score_simple_crumble_impact(
    states: List[GameState],
    step_cells: List[set[Tuple[int, int]]],
    crumble_set: set[Tuple[int, int]],
) -> Optional[Dict[str, int | float]]:
    """Measure how strongly a crumble region creates one-way commitment."""
    first_touch_step = None
    full_collapse_step = None
    for step_idx, (state, cells) in enumerate(zip(states, step_cells)):
        if first_touch_step is None and cells.intersection(crumble_set):
            first_touch_step = step_idx
        if full_collapse_step is None and crumble_set.issubset(state.collapsed_crumble):
            full_collapse_step = step_idx
            break

    if first_touch_step is None or full_collapse_step is None:
        return None

    pre_collapse_visited = _union_cell_sets(step_cells[:full_collapse_step])
    collapsed_state = states[full_collapse_step].clone()
    hypothetical_uncollapsed = GameState(
        board=collapsed_state.board,
        block=collapsed_state.block,
        goal_r=collapsed_state.goal_r,
        goal_c=collapsed_state.goal_c,
        active_bridges=collapsed_state.active_bridges,
        current_height=collapsed_state.current_height,
        collapsed_crumble=collapsed_state.collapsed_crumble.difference(crumble_set),
    )
    if not hypothetical_uncollapsed.is_valid():
        return None

    reachable_collapsed = _reachable_cells_from_state(collapsed_state)
    reachable_uncollapsed = _reachable_cells_from_state(hypothetical_uncollapsed)
    lost_reachable = reachable_uncollapsed - reachable_collapsed
    lost_previsited = (pre_collapse_visited & reachable_uncollapsed) - reachable_collapsed
    post_collapse_remaining = len(states) - 1 - full_collapse_step

    return {
        "crumble_first_touch_step": first_touch_step,
        "crumble_full_collapse_step": full_collapse_step,
        "crumble_post_collapse_remaining": post_collapse_remaining,
        "crumble_collapse_delay": full_collapse_step - first_touch_step,
        "crumble_lost_reachable_cells": len(lost_reachable),
        "crumble_lost_previsited_cells": len(lost_previsited),
        "crumble_return_blocked": int(bool(lost_previsited)),
    }


def _sample_simple_crumble_ambient_cells(
    context: Dict,
    core_set: set[Tuple[int, int]],
    rng: random.Random,
) -> set[Tuple[int, int]]:
    """Add optional extra CRUMBLE cells, preferring off-path scatter."""
    ambient_pool = set(context.get("off_path_cells", set())) - core_set
    if not ambient_pool:
        ambient_pool = set(context.get("eligible", set())) - core_set
    if not ambient_pool:
        return set()

    fringe_pool = set(context.get("fringe_off_path_cells", set())) - core_set
    board_area = int(context.get("board_area", 0))
    num_elevators = int(context.get("num_elevators", 1))
    if board_area <= 40:
        min_extra = 3
        max_extra = min(6, max(4, len(ambient_pool) // 4))
    else:
        min_extra = 4 + max(0, num_elevators - 1)
        max_extra = min(9, max(min_extra + 1, len(ambient_pool) // 5))
    max_extra = min(max_extra, len(ambient_pool))
    if max_extra <= 0:
        return set()
    min_extra = min(min_extra, max_extra)
    desired = rng.randint(min_extra, max_extra) if max_extra > min_extra else max_extra

    extras: set[Tuple[int, int]] = set()
    while len(extras) < desired:
        remaining = desired - len(extras)
        candidate_pool = ambient_pool - extras
        if not candidate_pool:
            break

        seed_pool = fringe_pool - extras if fringe_pool - extras and rng.random() < 0.65 else candidate_pool
        if remaining >= 2 and rng.random() < 0.55:
            region = _sample_connected_region(
                candidate_pool,
                rng,
                min(remaining, rng.choice([2, 2, 3, 3, 4])),
                seed_cells=seed_pool,
                max_attempts=10,
            )
            if region is not None:
                extras.update(region)
                continue

        extras.add(rng.choice(list(seed_pool)))

    return extras


def _combine_simple_crumble_quality(
    level: Dict,
    crumble_metrics: Dict[str, int | float],
) -> Dict[str, int | float]:
    """Merge branchiness metrics with crumble-impact metrics into one score."""
    base_metrics = _score_simple_elevator_level(level)
    base_quality_score = float(base_metrics["quality_score"])
    crumble_bonus = (
        5.0 * int(crumble_metrics["crumble_return_blocked"])
        + 1.5 * int(crumble_metrics["crumble_lost_previsited_cells"])
        + 0.75 * int(crumble_metrics["crumble_lost_reachable_cells"])
        + 1.0 * int(crumble_metrics["crumble_post_collapse_remaining"])
        + 0.5 * int(level.get("num_crumble", 0))
        + 0.25 * int(crumble_metrics["crumble_collapse_delay"])
        + 0.35 * int(crumble_metrics.get("ambient_crumble_count", 0))
        + 0.20 * int(crumble_metrics.get("untouched_crumble_count", 0))
        + 0.15 * int(crumble_metrics.get("off_path_crumble_count", 0))
    )

    merged = dict(base_metrics)
    merged.update(crumble_metrics)
    merged["base_quality_score"] = round(base_quality_score, 3)
    merged["crumble_bonus_score"] = round(crumble_bonus, 3)
    merged["quality_score"] = round(base_quality_score + crumble_bonus, 3)
    return merged


def _simple_crumble_level_sort_key(level: Dict) -> Tuple[float, int, int, int, int, int, int, int, int, int]:
    """Prefer crumble layouts with real irreversible impact plus base search quality."""
    metrics = level.get("quality_metrics", {})
    return (
        float(metrics.get("quality_score", float("-inf"))),
        int(metrics.get("crumble_return_blocked", 0)),
        int(metrics.get("crumble_lost_previsited_cells", 0)),
        int(metrics.get("crumble_post_collapse_remaining", 0)),
        int(metrics.get("ambient_crumble_count", 0)),
        int(metrics.get("deceptive_steps", 0)),
        int(metrics.get("deceptive_choices", 0)),
        int(metrics.get("untouched_crumble_count", 0)),
        int(level.get("num_crumble", 0)),
        int(metrics.get("solution_turns", 0)),
        -int(level.get("generation_subseed", 0)),
    )


def _inject_crumble_region_candidate(
    base_level: Dict,
    context: Dict,
    rng: random.Random,
    target_depth: int,
    mode: str,
    max_attempts: int = 8,
    second_region_prob: float = 0.25,
) -> Optional[Dict]:
    """Try several CRUMBLE placements for a fixed mode and keep the strongest."""
    if mode not in context["mode_pools"]:
        return None
    allowed_cells, seed_cells = context["mode_pools"][mode]
    if not allowed_cells:
        return None

    best_level: Optional[Dict] = None
    best_key = None

    for _ in range(max_attempts):
        core_region = _sample_connected_region(
            allowed_cells,
            rng,
            rng.choice([1, 2, 2, 3, 3]),
            seed_cells=seed_cells,
        )
        if core_region is None:
            continue

        core_regions = [set(core_region)]
        core_set = set(core_region)

        # add second crumble region with probs
        if rng.random() < second_region_prob:
            blocked = set(core_set)
            blocked.update(
                (nr, nc)
                for r, c in core_set
                for nr, nc in _neighbors4(r, c)
            )
            second_allowed = set(allowed_cells) - blocked
            second_seed = (set(seed_cells) & second_allowed) if seed_cells else second_allowed

            if second_allowed:
                region2 = _sample_connected_region(
                    second_allowed,
                    rng,
                    rng.choice([1, 2, 2, 3]),
                    seed_cells=second_seed or second_allowed,
                )
                if region2 is not None:
                    region2_set = set(region2)
                    core_regions.append(region2_set)
                    core_set.update(region2_set)

        ambient_variants = [set()]
        for _ambient_attempt in range(3):
            extra_cells = _sample_simple_crumble_ambient_cells(context, core_set, rng)
            if extra_cells and extra_cells not in ambient_variants:
                ambient_variants.append(extra_cells)

        for ambient_set in ambient_variants:
            crumble_set = core_set | ambient_set
            mutated = copy.deepcopy(base_level)
            for r, c in crumble_set:
                mutated["board"]["grid"][r][c] = int(TileType.CRUMBLE)

            state = GameState.from_dict(mutated)
            if not state.is_valid():
                continue

            solution = bfs_solve(state)
            if solution is None or len(solution) != target_depth:
                continue

            mutated["target_depth"] = len(solution)
            mutated["solution"] = [int(a) for a in solution]
            mutated["has_crumble"] = True
            mutated["num_crumble"] = len(crumble_set)
            mutated["crumble_positions"] = [[r, c] for r, c in sorted(crumble_set)]
            mutated["mandatory_crumble_positions"] = [[r, c] for r, c in sorted(core_set)]
            mutated["ambient_crumble_positions"] = [[r, c] for r, c in sorted(ambient_set)]
            mutated["num_mandatory_crumble"] = len(core_set)
            mutated["num_ambient_crumble"] = len(ambient_set)
            mutated["crumble_mode"] = mode

            states = _trace_solution_states(mutated, mutated["solution"])
            if states is None:
                continue
            step_cells = [set(step_state.block.cells()) for step_state in states]
            touched_all = _union_cell_sets([cells.intersection(crumble_set) for cells in step_cells])
            touched_core = touched_all.intersection(core_set)
            if touched_core != core_set:
                continue

            crumble_metrics = _score_simple_crumble_impact(states, step_cells, core_set)
            if crumble_metrics is None:
                continue
            if int(crumble_metrics["crumble_return_blocked"]) == 0:
                continue
            if int(crumble_metrics["crumble_post_collapse_remaining"]) < 2:
                continue

            untouched_set = crumble_set - touched_all
            off_path_crumble = crumble_set - set(context.get("path_cells", set()))
            crumble_metrics["mandatory_crumble_count"] = len(core_set)
            crumble_metrics["ambient_crumble_count"] = len(ambient_set)
            crumble_metrics["off_path_crumble_count"] = len(off_path_crumble)
            crumble_metrics["untouched_crumble_count"] = len(untouched_set)
            crumble_metrics["touched_crumble_count"] = len(touched_all)

            mutated["quality_metrics"] = _combine_simple_crumble_quality(mutated, crumble_metrics)
            mutated["quality_score"] = mutated["quality_metrics"]["quality_score"]
            mutated["crumble_first_touch_step"] = int(crumble_metrics["crumble_first_touch_step"])
            mutated["crumble_full_collapse_step"] = int(crumble_metrics["crumble_full_collapse_step"])

            sort_key = _simple_crumble_level_sort_key(mutated)
            if best_key is None or sort_key > best_key:
                best_level = mutated
                best_key = sort_key

    return best_level


def _iso_corner_simple(i: float, j: float, tw: float) -> Tuple[float, float]:
    """Screen-space point for the simplified isometric occlusion check."""
    yaw = math.radians(20)
    scale = tw * 0.55
    jx = scale * math.cos(yaw)
    jy = scale * math.sin(yaw) * 0.75
    ix = -scale * math.sin(yaw)
    iy = scale * math.cos(yaw) * 0.75
    return j * jx + i * ix, j * jy + i * iy


def _lerp_point_simple(
    a: Tuple[float, float],
    b: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def _quad_point_simple(
    tl: Tuple[float, float],
    tr: Tuple[float, float],
    br: Tuple[float, float],
    bl: Tuple[float, float],
    u: float,
    v: float,
) -> Tuple[float, float]:
    left = _lerp_point_simple(tl, bl, v)
    right = _lerp_point_simple(tr, br, v)
    return _lerp_point_simple(left, right, u)


def _point_in_polygon_simple(
    point: Tuple[float, float],
    poly: List[Tuple[float, float]],
) -> bool:
    """Ray-casting point-in-polygon test for convex tile faces."""
    x, y = point
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            denom = yj - yi
            if abs(denom) >= 1e-9:
                x_at_y = (xj - xi) * (y - yi) / denom + xi
                if x < x_at_y:
                    inside = not inside
        j = i
    return inside


def _simple_tile_faces(
    r: int,
    c: int,
    tw: float,
    thickness: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> List[List[Tuple[float, float]]]:
    """Tile polygons in the same order used by the isometric renderer."""
    tl = _iso_corner_simple(r, c, tw)
    tr = _iso_corner_simple(r, c + 1, tw)
    br = _iso_corner_simple(r + 1, c + 1, tw)
    bl = _iso_corner_simple(r + 1, c, tw)
    if x_offset or y_offset:
        tl = (tl[0] + x_offset, tl[1] + y_offset)
        tr = (tr[0] + x_offset, tr[1] + y_offset)
        br = (br[0] + x_offset, br[1] + y_offset)
        bl = (bl[0] + x_offset, bl[1] + y_offset)
    return [
        [bl, br, (br[0], br[1] + thickness), (bl[0], bl[1] + thickness)],
        [tr, br, (br[0], br[1] + thickness), (tr[0], tr[1] + thickness)],
        [tl, tr, br, bl],
    ]


def _simple_back_island(
    grid: List[List[int]],
    height_map: List[List[int]],
    start_pos: Tuple[int, int],
) -> Tuple[set, float]:
    """Replicate the renderer's easy-level tiering for occlusion checks."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    tw = 60.0
    th = tw / 2
    elevation_lift = th * 1.8
    back_island = set()

    low_cells = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if height_map[r][c] == 0 and grid[r][c] != TileType.VOID
    }
    if start_pos in low_cells:
        queue = deque([start_pos])
        while queue:
            cr, cc = queue.popleft()
            if (cr, cc) in back_island:
                continue
            back_island.add((cr, cc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) in low_cells and (nr, nc) not in back_island:
                    queue.append((nr, nc))

    if len(low_cells - back_island) < 3:
        back_island = set()
        elevation_lift = th * 1.2
    return back_island, elevation_lift


def _simple_tile_offset(
    r: int,
    c: int,
    height_map: List[List[int]],
    elevators: set,
    back_island: set,
    elevation_lift: float,
) -> Tuple[float, float]:
    """Combined-view offset for the initial simple-elevator render."""
    is_elevated = height_map[r][c] == 1
    is_elevator = (r, c) in elevators
    if is_elevated and not is_elevator:
        return 0.0, -elevation_lift
    if is_elevator:
        if not back_island:
            return 0.0, 0.0
        is_back_elevator = any(
            (r + dr, c + dc) in back_island
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        )
        return (0.0, -elevation_lift * 2.5) if is_back_elevator else (0.0, -elevation_lift)
    if (r, c) in back_island:
        return 0.0, -elevation_lift * 2.5
    return 0.0, 0.0


def _simple_elevator_has_occlusion(
    grid: List[List[int]],
    height_map: List[List[int]],
    elevators: set,
    start_pos: Tuple[int, int],
) -> bool:
    """Return True when the combined isometric view hides LOW tiles behind HIGH."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if not rows or not cols:
        return False

    tw = 60.0
    th = tw / 2
    tile_thick = th * 0.20
    back_island, elevation_lift = _simple_back_island(grid, height_map, start_pos)

    faces = []
    order = 0
    for depth in range(rows + cols - 1):
        cells = [(r, depth - r) for r in range(max(0, depth - cols + 1), min(rows, depth + 1))]
        cells.sort(key=lambda rc: -height_map[rc[0]][rc[1]])
        for r, c in cells:
            if grid[r][c] == TileType.VOID:
                continue
            x_off, y_off = _simple_tile_offset(r, c, height_map, elevators, back_island, elevation_lift)
            is_high = height_map[r][c] == 1 or ((r, c) in elevators and y_off < 0)
            for poly in _simple_tile_faces(r, c, tw, tile_thick, x_offset=x_off, y_offset=y_off):
                faces.append({"order": order, "cell": (r, c), "is_high": is_high, "poly": poly})
                order += 1

    checked_cells = set()
    for face in reversed(faces):
        cell = face["cell"]
        if cell in checked_cells or face["is_high"]:
            continue
        checked_cells.add(cell)
        r, c = cell
        if height_map[r][c] != 0 or grid[r][c] == TileType.VOID:
            continue
        tl, tr, br, bl = face["poly"]
        samples = [
            _quad_point_simple(tl, tr, br, bl, 0.50, 0.50),
            _quad_point_simple(tl, tr, br, bl, 0.35, 0.50),
            _quad_point_simple(tl, tr, br, bl, 0.65, 0.50),
            _quad_point_simple(tl, tr, br, bl, 0.50, 0.35),
            _quad_point_simple(tl, tr, br, bl, 0.50, 0.65),
        ]
        later_high_faces = [entry for entry in faces if entry["order"] > face["order"] and entry["is_high"]]
        covered = 0
        for sample in samples:
            for entry in later_high_faces:
                if _point_in_polygon_simple(sample, entry["poly"]):
                    covered += 1
                    break
        if covered:
            return True
    return False


_SIMPLE_ELEVATOR_STYLES = ["iso_straight", "iso_jagged", "iso_step", "L_shape", "blob"]
_SIMPLE_ELEVATOR_STYLE_ATTEMPTS = 8
_SIMPLE_ELEVATOR_FORCED_STYLE_ATTEMPTS = 12
_SIMPLE_ELEVATOR_NEAR_OPTIMAL_SLACK = 2


def _simple_solution_turns(solution: List[Action]) -> int:
    """Count direction changes along a solution path."""
    if len(solution) < 2:
        return 0
    turns = 0
    for prev, curr in zip(solution, solution[1:]):
        if prev != curr:
            turns += 1
    return turns


def _simple_height_boundary_edges(
    grid: List[List[int]],
    height_map: List[List[int]],
) -> int:
    """Count exposed LOW/HIGH boundaries among non-VOID cells."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    edges = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == TileType.VOID:
                continue
            for dr, dc in [(1, 0), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if grid[nr][nc] == TileType.VOID:
                    continue
                if height_map[r][c] != height_map[nr][nc]:
                    edges += 1
    return edges


def _simple_step_choice_stats(state: GameState, remaining_steps: int) -> Dict[str, int]:
    """Measure how many valid actions stay optimal, become near-optimal, or fail."""
    optimal = 0
    deceptive = 0
    dead_end = 0
    valid = 0

    for action in Action:
        next_state = state.clone()
        ok, _ = next_state.apply_action(action)
        if not ok:
            continue
        valid += 1
        remainder = bfs_solve(next_state)
        if remainder is None:
            dead_end += 1
            continue
        rem_len = len(remainder)
        if rem_len == remaining_steps:
            optimal += 1
        elif rem_len <= remaining_steps + _SIMPLE_ELEVATOR_NEAR_OPTIMAL_SLACK:
            deceptive += 1

    return {
        "optimal": optimal,
        "deceptive": deceptive,
        "dead_end": dead_end,
        "valid": valid,
    }


def _score_simple_elevator_level(level: Dict) -> Dict[str, float | int]:
    """Attach lightweight search-quality metrics for generator selection."""
    state = GameState.from_dict(level)
    solution = [Action(a) for a in level["solution"]]
    grid = state.board.grid
    height_map = state.board.height_map or [[0] * state.board.cols for _ in range(state.board.rows)]
    void_tiles = sum(
        1
        for row in grid
        for tile in row
        if tile == TileType.VOID
    )
    walkable_tiles = state.board.rows * state.board.cols - void_tiles

    valid_counts: List[int] = []
    multi_shortest_steps = 0
    deceptive_steps = 0
    deceptive_choices = 0
    dead_end_steps = 0
    dead_end_choices = 0

    for step_idx, action in enumerate(solution[:-1]):
        remaining_steps = len(solution) - step_idx - 1
        stats = _simple_step_choice_stats(state, remaining_steps)
        valid_counts.append(stats["valid"])
        if stats["optimal"] > 1:
            multi_shortest_steps += 1
        if stats["deceptive"] > 0:
            deceptive_steps += 1
        deceptive_choices += stats["deceptive"]
        if stats["dead_end"] > 0:
            dead_end_steps += 1
        dead_end_choices += stats["dead_end"]

        ok, _ = state.apply_action(action)
        if not ok:
            break

    valid_action_mean = sum(valid_counts) / len(valid_counts) if valid_counts else 0.0
    quality_score = (
        7.0 * multi_shortest_steps
        + 4.0 * deceptive_steps
        + 1.5 * deceptive_choices
        + 1.0 * dead_end_steps
        + 0.5 * dead_end_choices
        + 0.75 * _simple_solution_turns(solution)
        + 0.35 * _simple_height_boundary_edges(grid, height_map)
        + 0.5 * valid_action_mean
        + 0.15 * void_tiles
    )

    return {
        "quality_score": round(quality_score, 3),
        "multi_shortest_steps": multi_shortest_steps,
        "deceptive_steps": deceptive_steps,
        "deceptive_choices": deceptive_choices,
        "dead_end_steps": dead_end_steps,
        "dead_end_choices": dead_end_choices,
        "solution_turns": _simple_solution_turns(solution),
        "valid_action_mean": round(valid_action_mean, 3),
        "valid_action_max": max(valid_counts, default=0),
        "walkable_tiles": walkable_tiles,
        "void_tiles": void_tiles,
        "height_boundary_edges": _simple_height_boundary_edges(grid, height_map),
    }


def _simple_level_sort_key(level: Dict) -> Tuple[float, int, int, int, int, int, int, int, int]:
    """Prefer branchier, more deceptive accepted levels while staying deterministic."""
    metrics = level.get("quality_metrics", {})
    walkable = int(metrics.get("walkable_tiles", 0))
    irregularity_bias = -abs(walkable - 18)
    return (
        float(metrics.get("quality_score", float("-inf"))),
        int(metrics.get("multi_shortest_steps", 0)),
        int(metrics.get("deceptive_steps", 0)),
        int(metrics.get("deceptive_choices", 0)),
        int(metrics.get("dead_end_steps", 0)),
        int(metrics.get("solution_turns", 0)),
        int(metrics.get("height_boundary_edges", 0)),
        irregularity_bias,
        -int(level.get("generation_subseed", 0)),
    )


def _simple_crumble_level_fingerprint(level: Dict) -> str:
    """Compact structural fingerprint used for dataset-level deduplication."""
    board = level.get("board", {})
    payload = {
        "grid": board.get("grid"),
        "height_map": board.get("height_map"),
        "elevators": board.get("elevators", []),
        "block": level.get("block"),
        "goal": level.get("goal"),
        "crumble_positions": level.get("crumble_positions", []),
    }
    return json.dumps(payload, sort_keys=True)


def _simple_attempt_void_fraction(base_fraction: float, rng: random.Random) -> float:
    """Vary void density slightly so repeated attempts explore different shapes."""
    lo = max(0.08, base_fraction * 0.7)
    hi = min(0.40, base_fraction * 1.2)
    if hi <= lo:
        return base_fraction
    return rng.uniform(lo, hi)


def _generate_elevator_level_simple_once(
    target_depth: int,
    grid_rows: int = 5,
    grid_cols: int = 7,
    seed: Optional[int] = None,
    void_fraction: float = 0.20,
    style: Optional[str] = None,
) -> Optional[Dict]:
    """
    Generate a compact elevator level with diverse layouts.

    Randomly chooses among several split styles (vertical, horizontal,
    diagonal, L-shaped, checkerboard blob) so that the elevated region,
    start position, goal position, and elevator placement all vary.
    Voids are carved aggressively to create erratic, organic shapes.
    """
    rng = random.Random(seed)

    # -- build full NORMAL grid --
    grid = [[TileType.NORMAL] * grid_cols for _ in range(grid_rows)]
    height_map = [[0] * grid_cols for _ in range(grid_rows)]

    # -- choose a split style --
    # All styles are designed so HIGH region is in the isometric "back" (low r+c).
    # In isometric projection: depth = r + c, low depth = back, high depth = front.
    if style is None:
        style = rng.choice(_SIMPLE_ELEVATOR_STYLES)

    if style == "iso_straight":
        # Split along r + c = threshold (perpendicular to iso depth axis).
        # HIGH is in the back (low r+c), LOW in front (high r+c).
        max_depth = grid_rows + grid_cols - 2
        threshold = rng.randint(3, max_depth - 3)
        for r in range(grid_rows):
            for c in range(grid_cols):
                if r + c < threshold:
                    height_map[r][c] = 1
        # Elevator on the boundary
        boundary = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                if height_map[r][c] == 0:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_rows and 0 <= nc < grid_cols and height_map[nr][nc] == 1:
                            boundary.append((r, c))
                            break
        if not boundary:
            return None
        elev_r, elev_c = rng.choice(boundary)

    elif style == "iso_jagged":
        # Like iso_straight but with a jagged/noisy boundary for visual variety.
        max_depth = grid_rows + grid_cols - 2
        base_threshold = rng.randint(3, max_depth - 3)
        for r in range(grid_rows):
            for c in range(grid_cols):
                jitter = rng.randint(-1, 1)
                if r + c < base_threshold + jitter:
                    height_map[r][c] = 1
        boundary = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                if height_map[r][c] == 0:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_rows and 0 <= nc < grid_cols and height_map[nr][nc] == 1:
                            boundary.append((r, c))
                            break
        if not boundary:
            return None
        elev_r, elev_c = rng.choice(boundary)

    elif style == "iso_step":
        # Stepped split: threshold varies by row or column, creating a staircase.
        max_depth = grid_rows + grid_cols - 2
        base = rng.randint(3, max_depth - 3)
        step_axis = rng.choice(["row", "col"])
        for r in range(grid_rows):
            for c in range(grid_cols):
                offset = (r // 2) if step_axis == "row" else (c // 2)
                if r + c < base + offset:
                    height_map[r][c] = 1
        boundary = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                if height_map[r][c] == 0:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_rows and 0 <= nc < grid_cols and height_map[nr][nc] == 1:
                            boundary.append((r, c))
                            break
        if not boundary:
            return None
        elev_r, elev_c = rng.choice(boundary)

    elif style == "L_shape":
        # L-shaped elevated region (top-right + bottom-right, or similar)
        corner = rng.choice(["TL", "TR", "BL", "BR"])
        mid_r = rng.randint(2, grid_rows - 3)
        mid_c = rng.randint(2, grid_cols - 3)
        for r in range(grid_rows):
            for c in range(grid_cols):
                if corner == "TR" and (r <= mid_r or c >= mid_c):
                    if not (r <= mid_r and c < mid_c):
                        height_map[r][c] = 1
                elif corner == "TL" and (r <= mid_r or c <= mid_c):
                    if not (r <= mid_r and c > mid_c):
                        height_map[r][c] = 1
                elif corner == "BR" and (r >= mid_r or c >= mid_c):
                    if not (r >= mid_r and c < mid_c):
                        height_map[r][c] = 1
                elif corner == "BL" and (r >= mid_r or c <= mid_c):
                    if not (r >= mid_r and c > mid_c):
                        height_map[r][c] = 1
        # Elevator at boundary
        boundary = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                if height_map[r][c] == 0:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid_rows and 0 <= nc < grid_cols and height_map[nr][nc] == 1:
                            boundary.append((r, c))
                            break
        if not boundary:
            return None
        elev_r, elev_c = rng.choice(boundary)

    else:  # blob
        # Grow a random blob from a seed cell for the elevated region
        blob_seed_r = rng.randint(1, grid_rows - 2)
        blob_seed_c = rng.randint(1, grid_cols - 2)
        blob = {(blob_seed_r, blob_seed_c)}
        target_size = (grid_rows * grid_cols) // 3 + rng.randint(-2, 2)
        frontier = [(blob_seed_r, blob_seed_c)]
        while len(blob) < target_size and frontier:
            rng.shuffle(frontier)
            r, c = frontier.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < grid_rows and 0 <= nc < grid_cols
                        and (nr, nc) not in blob and rng.random() < 0.6):
                    blob.add((nr, nc))
                    frontier.append((nr, nc))
        for r, c in blob:
            height_map[r][c] = 1
        # Elevator at boundary of blob
        boundary = []
        for r, c in blob:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_rows and 0 <= nc < grid_cols and (nr, nc) not in blob:
                    boundary.append((nr, nc))
                    break
        if not boundary:
            return None
        elev_r, elev_c = rng.choice(boundary)

    # -- ensure HIGH region is in the isometric "back" (low r+c) to avoid occlusion --
    # In isometric projection, depth = r + c: low = back, high = front.
    # Flip if the HIGH region extends further toward the front than the LOW region.
    # Use median depth to be robust against outliers.
    high_cells_pre = [(r, c) for r in range(grid_rows) for c in range(grid_cols) if height_map[r][c] == 1]
    low_cells_pre = [(r, c) for r in range(grid_rows) for c in range(grid_cols) if height_map[r][c] == 0]
    if high_cells_pre and low_cells_pre:
        high_depths = sorted(r + c for r, c in high_cells_pre)
        low_depths = sorted(r + c for r, c in low_cells_pre)
        median_high = high_depths[len(high_depths) // 2]
        median_low = low_depths[len(low_depths) // 2]
        if median_high >= median_low:
            # Flip: swap HIGH and LOW so HIGH is in the back
            for r in range(grid_rows):
                for c in range(grid_cols):
                    height_map[r][c] = 1 - height_map[r][c]
            height_map[elev_r][elev_c] = 0  # elevator always at height 0

    # -- place elevator --
    grid[elev_r][elev_c] = TileType.ELEVATOR
    elevators = {(elev_r, elev_c)}
    elevator_positions = [[elev_r, elev_c]]
    height_map[elev_r][elev_c] = 0  # accessible from both levels

    # -- collect cells per height level --
    low_cells = [(r, c) for r in range(grid_rows) for c in range(grid_cols)
                 if height_map[r][c] == 0 and (r, c) != (elev_r, elev_c)]
    high_cells = [(r, c) for r in range(grid_rows) for c in range(grid_cols)
                  if height_map[r][c] == 1]

    if len(low_cells) < 3 or len(high_cells) < 3:
        return None

    # -- place start on low, goal on high (or vice versa) --
    if rng.choice([True, False]):
        start_r, start_c = rng.choice(low_cells)
        goal_r, goal_c = rng.choice(high_cells)
    else:
        start_r, start_c = rng.choice(high_cells)
        goal_r, goal_c = rng.choice(low_cells)

    grid[goal_r][goal_c] = TileType.GOAL
    start_block = Block(start_r, start_c, Orientation.STANDING)

    # -- carve voids --
    protected = {(start_r, start_c), (goal_r, goal_c), (elev_r, elev_c)}
    # Protect cells reachable by one tip from start (standing block tips 2 cells)
    for dr, dc in [(-2, 0), (-1, 0), (1, 0), (2, 0), (0, -2), (0, -1), (0, 1), (0, 2)]:
        nr, nc = start_r + dr, start_c + dc
        if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
            protected.add((nr, nc))

    all_cells = [(r, c) for r in range(grid_rows) for c in range(grid_cols)]
    rng.shuffle(all_cells)
    for r, c in all_cells:
        if (r, c) in protected:
            continue
        if rng.random() < void_fraction:
            grid[r][c] = TileType.VOID

    # Reject layouts whose initial combined isometric render still hides LOW tiles.
    if _simple_elevator_has_occlusion(grid, height_map, elevators, (start_r, start_c)):
        return None

    # -- BFS solve --
    board = Board(grid_rows, grid_cols, grid, {}, {}, set(), height_map, elevators)
    state = GameState(board, start_block, goal_r, goal_c, frozenset(), 0)

    if not state.is_valid():
        return None

    solution = bfs_solve(state)
    if solution is None or len(solution) != target_depth:
        return None

    # -- verify elevator is REQUIRED --
    test_grid = [row[:] for row in grid]
    for er, ec in elevators:
        test_grid[er][ec] = TileType.NORMAL
    test_board = Board(grid_rows, grid_cols, test_grid, {}, {}, set(), height_map, set())
    test_state = GameState(test_board, start_block, goal_r, goal_c, frozenset(), 0)
    if bfs_solve(test_state) is not None:
        return None

    return {
        "board": board.to_dict(),
        "block": {"r": start_r, "c": start_c, "orientation": 0},
        "goal": {"r": goal_r, "c": goal_c},
        "target_depth": len(solution),
        "solution": [int(a) for a in solution],
        "seed": seed,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "has_elevators": True,
        "num_elevators": len(elevators),
        "elevator_positions": elevator_positions,
        "height_map": height_map,
        "style": style,
    }


def generate_elevator_level_simple(
    target_depth: int,
    grid_rows: int = 5,
    grid_cols: int = 7,
    seed: Optional[int] = None,
    void_fraction: float = 0.20,
    style: Optional[str] = None,
) -> Optional[Dict]:
    """
    Generate a compact elevator level, searching over multiple candidates and
    returning the most interesting accepted one.

    The underlying single-attempt generator keeps the same geometric rules as
    before. This wrapper broadens the search across style-specific retries and
    ranks successful candidates by a lightweight branchiness/deception score so
    that exact-depth matches are less often narrow, one-route artifacts.
    """
    planner_rng = random.Random(seed)
    style_pool = [style] if style is not None else list(_SIMPLE_ELEVATOR_STYLES)
    if style is None:
        planner_rng.shuffle(style_pool)

    attempts_per_style = (
        _SIMPLE_ELEVATOR_FORCED_STYLE_ATTEMPTS
        if style is not None
        else _SIMPLE_ELEVATOR_STYLE_ATTEMPTS
    )

    best_level: Optional[Dict] = None
    best_key: Optional[Tuple[float, int, int, int, int, int, int, int, int]] = None

    for style_idx, style_name in enumerate(style_pool):
        for attempt_idx in range(attempts_per_style):
            attempt_seed = planner_rng.randrange(1 << 30)
            attempt_void_fraction = _simple_attempt_void_fraction(void_fraction, planner_rng)
            level = _generate_elevator_level_simple_once(
                target_depth=target_depth,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                seed=attempt_seed,
                void_fraction=attempt_void_fraction,
                style=style_name,
            )
            if level is None:
                continue

            level["seed"] = seed
            level["generation_subseed"] = attempt_seed
            level["requested_void_fraction"] = void_fraction
            level["void_fraction"] = round(attempt_void_fraction, 4)
            level["style_attempt_index"] = attempt_idx
            level["style_pool_index"] = style_idx
            level["quality_metrics"] = _score_simple_elevator_level(level)
            level["quality_score"] = level["quality_metrics"]["quality_score"]

            sort_key = _simple_level_sort_key(level)
            if best_key is None or sort_key > best_key:
                best_level = level
                best_key = sort_key

    return best_level


def generate_crumble_elevator_level(
    target_depth: int,
    grid_rows: int = 5,
    grid_cols: int = 7,
    seed: Optional[int] = None,
    void_fraction: float = 0.2,
    style: Optional[str] = None,
    crumble_mode: Optional[str] = None,
) -> Optional[Dict]:
    """
    Generate a quality-aware elevator level with CRUMBLE mechanics.

    Uses the compact one-elevator family. Candidates require a CRUMBLE core with
    real one-way commitment, then ranked by a combined branchiness +
    crumble-impact score.
    """
    planner_rng = random.Random(seed)
    mode_pool = [crumble_mode] if crumble_mode is not None else list(_SIMPLE_CRUMBLE_MODES)
    if crumble_mode is None:
        planner_rng.shuffle(mode_pool)

    attempts_per_mode = (
        _SIMPLE_CRUMBLE_FORCED_MODE_ATTEMPTS
        if crumble_mode is not None
        else _SIMPLE_CRUMBLE_MODE_ATTEMPTS
    )

    best_level: Optional[Dict] = None
    best_key = None

    base_attempt_budget = _SIMPLE_CRUMBLE_BASE_ATTEMPTS
    for base_attempt_idx in range(base_attempt_budget):
        base_seed = planner_rng.randrange(1 << 30)
        base_level = generate_elevator_level_simple(
            target_depth=target_depth,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            seed=base_seed,
            void_fraction=void_fraction,
            style=style,
        )
        if base_level is None:
            continue

        context = _build_simple_crumble_context(base_level)
        if context is None:
            continue

        mode_order = list(mode_pool)
        if crumble_mode is None:
            planner_rng.shuffle(mode_order)

        for mode_idx, mode_name in enumerate(mode_order):
            crumble_level = _inject_crumble_region_candidate(
                base_level=base_level,
                context=context,
                rng=planner_rng,
                target_depth=target_depth,
                mode=mode_name,
                max_attempts=attempts_per_mode,
            )
            if crumble_level is None:
                continue

            crumble_level["seed"] = seed
            crumble_level["base_seed"] = base_seed
            crumble_level["base_attempt_index"] = base_attempt_idx
            crumble_level["crumble_mode_pool_index"] = mode_idx
            crumble_level["requested_crumble_mode"] = crumble_mode
            crumble_level["requested_num_elevators"] = 1

            sort_key = _simple_crumble_level_sort_key(crumble_level)
            if best_key is None or sort_key > best_key:
                best_level = crumble_level
                best_key = sort_key

    return best_level


