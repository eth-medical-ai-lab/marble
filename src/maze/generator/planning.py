"""Generate exact-depth planning boards with replayable forward solutions."""
from __future__ import annotations
from collections import deque
import random
from typing import Any, Dict, List, Optional, Tuple
from maze.env import Board, Edit, GameState, Side, bfs_path, is_forbidden_lane, ODD_IDXS, ROTATIONS

RECIPE = {"size": 7, "depth": {"easy": 2, "hard": 4}, "objects": 1,
          "anchor_shapes": ["T", "L", "I"], "random_start": True,
          "max_attempts": 10000, "rotation": "absolute quarter-turns"}
SIDES = tuple(s.value for s in (Side.LEFT, Side.RIGHT, Side.TOP, Side.BOTTOM))


def reachable_positions(board: Board, start: Tuple[int, int]) -> set:
    q = deque([start])
    seen = {start}
    while q:
        x, y = q.popleft()
        for nx, ny in board.neighbors(x, y):
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return seen


def shift_position_for_insert(
    position: Tuple[int, int],
    side: str,
    idx: int,
    width: int,
    height: int,
) -> Tuple[int, int]:
    x, y = position
    if side == "left" and y == idx:
        return ((x + 1) % width, y)
    if side == "right" and y == idx:
        return ((x - 1) % width, y)
    if side == "top" and x == idx:
        return (x, (y + 1) % height)
    if side == "bottom" and x == idx:
        return (x, (y - 1) % height)
    return position


def forward_shortest_turns_solver(
    board: Board,
    start_pos: Tuple[int, int],
    target_label: str,
    max_turns: int = 10,
) -> Optional[Dict[str, Any]]:
    """BFS over turns to find a minimal insertion sequence to reach target_label.

    Respects the forward rule: each turn inserts once, then moves arbitrarily;
    enforces the forbidden-lane constraint.
    Returns dict with keys: D, insertion_seq, final_movement_path, endpoints.
    """
    if target_label not in board.objects:
        return None

    # 0-turn check: can the player reach the target without insertions?
    goal0 = board.objects[target_label]
    path0 = bfs_path(board, start_pos, goal0)
    if path0:
        return {
            "D": 0,
            "insertion_seq": [],
            "final_movement_path": path0,
            "endpoints": [],
        }

    frontier: List[Tuple[Board, Tuple[int, int], Optional[Tuple[str, int]]]] = [
        (board, start_pos, None)
    ]
    visited = {(board.signature(), start_pos, None)}
    parent: Dict[Tuple, Tuple[Optional[Tuple], Edit, Tuple[int, int]]] = {}

    def make_key(b, pos, last_lane):
        return (b.signature(), pos, last_lane)

    for depth in range(1, max_turns + 1):
        next_frontier = []
        for bcurr, pos, last_lane in frontier:
            parent_key = make_key(bcurr, pos, last_lane)
            for side in SIDES:
                for idx in ODD_IDXS:
                    if last_lane is not None:
                        prev_side, prev_idx = last_lane
                        if is_forbidden_lane(Edit(prev_side, prev_idx, 0), side, idx):
                            continue
                    for rotate in ROTATIONS:
                        next_board = bcurr.clone()
                        next_board.rotate_extra(rotate)
                        next_board.insert(side, idx)
                        player_after = shift_position_for_insert(
                            pos, side, idx, next_board.w, next_board.h
                        )
                        goal1 = next_board.objects.get(target_label, pos)
                        sig_board = next_board.signature()

                        path_to_goal = bfs_path(next_board, player_after, goal1)
                        if path_to_goal:
                            leaf_key = (sig_board, goal1, (side, idx))
                            parent[leaf_key] = (parent_key, Edit(side, idx, rotate), goal1)

                            endpoints, edits = [], []
                            cur = leaf_key
                            while cur in parent:
                                p_key, p_edit, ep = parent[cur]
                                edits.append(p_edit)
                                endpoints.append(ep)
                                if p_key is None:
                                    break
                                cur = p_key
                            edits.reverse()
                            endpoints.reverse()
                            return {
                                "D": depth,
                                "insertion_seq": edits,
                                "final_movement_path": path_to_goal,
                                "endpoints": endpoints,
                            }

                        # No successor can be explored beyond the requested depth.
                        if depth == max_turns:
                            continue
                        for p in sorted(reachable_positions(next_board, player_after)):
                            key = (sig_board, p, (side, idx))
                            if key in visited:
                                continue
                            visited.add(key)
                            parent[key] = (parent_key, Edit(side, idx, rotate), p)
                            next_frontier.append((next_board, p, (side, idx)))
        frontier = next_frontier
        if not frontier:
            break
    return None


def make_record(board, start, target, result):
    """Replay solver edits; save distinct post-insert and post-move states."""
    state = GameState(board.clone(), {"P1": start}, {"P1": board.objects[target]})
    record = {"target_piece": target, "depth": result["D"], "initial_state": state.to_dict(), "steps": []}
    for turn, (edit, endpoint) in enumerate(zip(result["insertion_seq"], result["endpoints"]), 1):
        state.apply_insert(edit.side, edit.idx, edit.rotate_extra)
        after_insert = state.to_dict()
        path = bfs_path(state.board, state.players["P1"], endpoint)
        if not path:
            raise RuntimeError("Solver returned an unreachable movement endpoint")
        state.players["P1"] = endpoint
        record["steps"].append({
            "turn": turn, "insertion": edit.to_dict(),
            "movement": {"direction": "move", "path": path},
            "after_insertion": after_insert, "after_movement": state.to_dict(),
        })
    if state.players["P1"] != state.board.objects[target]:
        raise RuntimeError("Generated solution does not reach its target")
    record["final_state"] = state.to_dict()
    return record


def generate(num_samples, seed, output_dir, config):
    """Keep unique boards whose shortest solution has exactly the requested depth."""
    depth = RECIPE["depth"][config]
    seen = set()
    count = 0
    for attempt in range(RECIPE["max_attempts"]):
        sample_seed = seed + attempt
        rng = random.Random(sample_seed)
        start = (rng.randrange(7), rng.randrange(7))
        board = Board.strategic_board(seed=sample_seed, object_count=1,
                                      strategic_anchor_shapes=("T", "L", "I"))
        target = next(iter(board.objects))
        fingerprint = (board.signature(), start, target)
        if fingerprint in seen:
            continue
        print(f"Searching MAZE seed {sample_seed} for depth {depth}", flush=True)
        result = forward_shortest_turns_solver(board, start, target, max_turns=depth)
        if result is None or result["D"] != depth:
            continue
        seen.add(fingerprint)
        record = make_record(board, start, target, result)
        yield {"id": f"puzzle_{count:04d}", "seed": sample_seed, "difficulty": config, **record}
        count += 1
        if count == num_samples:
            return
    raise RuntimeError(f"Only found {count}/{num_samples} exact-depth MAZE boards")
