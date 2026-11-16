"""Generate the benchmark's 5x7, one-elevator, crumble planning levels."""
from collections import deque
from blox.constants import Action, TileType
from blox.env import GameState, bfs_solve
from blox.generator.levels import generate_crumble_elevator_level, _simple_crumble_level_fingerprint

RECIPE = {"rows": 5, "cols": 7, "elevators": 1, "void_fraction": 0.2,
          "depth": {"easy": 10, "hard": 16}, "connected": True,
          "max_attempts": {"easy": 50000, "hard": 100000}}


def validate_level(level, depth):
    """Check connected geometry, exact BFS depth, and the exported action sequence."""
    state = GameState.from_dict(level)
    board = state.board
    solids = {(r, c) for r in range(board.rows) for c in range(board.cols)
              if board.grid[r][c] != TileType.VOID}
    start = next(iter(solids))
    seen, queue = {start}, deque([start])
    while queue:
        r, c = queue.popleft()
        for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if cell in solids and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    if seen != solids or (board.rows, board.cols) != (5, 7) or len(board.elevators) != 1:
        return False
    if not board.has_tile_type(TileType.CRUMBLE):
        return False
    solution = bfs_solve(state)
    if solution is None or len(solution) != depth or len(level["solution"]) != depth:
        return False
    for action in level["solution"]:
        valid, _ = state.apply_action(Action(action))
        if not valid:
            return False
    return state.is_won()


def generate(num_samples, seed, output_dir, config):
    """Keep the original candidate ranking and reject duplicates or invalid levels."""
    depth = RECIPE["depth"][config]
    seen = set()
    count = 0
    for attempt in range(RECIPE["max_attempts"][config]):
        sample_seed = seed + attempt
        level = generate_crumble_elevator_level(target_depth=depth, seed=sample_seed)
        if level is None:
            continue
        fingerprint = _simple_crumble_level_fingerprint(level)
        if fingerprint in seen or not validate_level(level, depth):
            continue
        seen.add(fingerprint)
        yield {"id": f"level_{count:04d}", "difficulty": config, **level}
        count += 1
        if count == num_samples:
            return
    raise RuntimeError(f"Only found {count}/{num_samples} valid BLOX levels")
