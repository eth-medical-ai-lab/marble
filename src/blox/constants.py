"""
Blox constants: enums, movement transition table, and rendering colors.
"""
from enum import IntEnum
from typing import Dict, Tuple


class Orientation(IntEnum):
    STANDING = 0
    LYING_H = 1   # lying horizontally (occupies (r,c) and (r,c+1))
    LYING_V = 2   # lying vertically   (occupies (r,c) and (r+1,c))


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_NAMES = {Action.UP: "UP", Action.DOWN: "DOWN", Action.LEFT: "LEFT", Action.RIGHT: "RIGHT"}
ACTION_FROM_NAME = {v: k for k, v in ACTION_NAMES.items()}


class TileType(IntEnum):
    VOID = 0
    NORMAL = 1
    GOAL = 2
    SWITCH = 3
    BRIDGE = 4
    FRAGILE = 5
    ELEVATOR = 6
    CRUMBLE = 7


# Movement transition table:
# Key: (Orientation, Action) -> (new_orientation, delta_row, delta_col)
# delta is applied to the block's anchor (r, c) to get the new anchor.
MOVEMENT_TABLE: Dict[Tuple[Orientation, Action], Tuple[Orientation, int, int]] = {
    # STANDING
    (Orientation.STANDING, Action.RIGHT): (Orientation.LYING_H, 0, 1),
    (Orientation.STANDING, Action.LEFT):  (Orientation.LYING_H, 0, -2),
    (Orientation.STANDING, Action.DOWN):  (Orientation.LYING_V, 1, 0),
    (Orientation.STANDING, Action.UP):    (Orientation.LYING_V, -2, 0),
    # LYING_H
    (Orientation.LYING_H, Action.RIGHT):  (Orientation.STANDING, 0, 2),
    (Orientation.LYING_H, Action.LEFT):   (Orientation.STANDING, 0, -1),
    (Orientation.LYING_H, Action.DOWN):   (Orientation.LYING_H, 1, 0),
    (Orientation.LYING_H, Action.UP):     (Orientation.LYING_H, -1, 0),
    # LYING_V
    (Orientation.LYING_V, Action.DOWN):   (Orientation.STANDING, 2, 0),
    (Orientation.LYING_V, Action.UP):     (Orientation.STANDING, -1, 0),
    (Orientation.LYING_V, Action.RIGHT):  (Orientation.LYING_V, 0, 1),
    (Orientation.LYING_V, Action.LEFT):   (Orientation.LYING_V, 0, -1),
}


# Rendering colors
COLOR_BG = "#1a1a2e"
COLOR_NORMAL = "#d2b48c"
COLOR_VOID = "#2a2a3e"
COLOR_GOAL = "#c0392b"
COLOR_GOAL_DIAMOND = "#e74c3c"
COLOR_BLOCK = "#3498db"
COLOR_BLOCK_SHADOW = "#2471a3"
COLOR_GRID_LINE = "#444466"
COLOR_TEXT = "#ecf0f1"
COLOR_LEGEND_BG = "#16213e"

# Switch / Bridge colors (group 0 = orange, group 1 = purple)
COLOR_SWITCH = "#e67e22"
COLOR_SWITCH_MARKER = "#f39c12"
COLOR_BRIDGE_ON = "#27ae60"
COLOR_BRIDGE_OFF = "#7f8c8d"

# Per-group colors: (switch_tile, switch_marker, bridge_on, bridge_off)
GROUP_COLORS = {
    0: {"switch": "#e67e22", "switch_marker": "#f39c12",
        "bridge_on": "#27ae60", "bridge_off": "#7f8c8d"},
    1: {"switch": "#8e44ad", "switch_marker": "#a569bd",
        "bridge_on": "#5dade2", "bridge_off": "#5d6d7e"},
}
COLOR_FRAGILE = "#e8d5a3"
COLOR_FRAGILE_MARKER = "#b8956a"
COLOR_CRUMBLE = "#c8c2b8"
COLOR_CRUMBLE_MARKER = "#6a6258"

# Elevator / elevated tile colors
COLOR_ELEVATED = "#d2b48c"         # same as normal — height shown by elevation offset
COLOR_ELEVATOR = "#9b59b6"         # purple for elevator tiles
COLOR_ELEVATOR_MARKER = "#8e44ad"  # darker purple marker (up/down arrows)
