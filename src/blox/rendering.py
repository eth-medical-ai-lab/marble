"""BLOX rendering."""
from __future__ import annotations
import base64
import re
import io
from typing import List, Optional, Tuple
from blox.constants import (
    Orientation,
    TileType,
    COLOR_BG,
    COLOR_BLOCK,
    COLOR_BRIDGE_OFF,
    COLOR_CRUMBLE,
    COLOR_CRUMBLE_MARKER,
    COLOR_ELEVATED,
    COLOR_ELEVATOR,
    COLOR_ELEVATOR_MARKER,
    COLOR_FRAGILE,
    COLOR_FRAGILE_MARKER,
    COLOR_GOAL,
    COLOR_GOAL_DIAMOND,
    COLOR_NORMAL,
    COLOR_TEXT,
    COLOR_VOID,
    GROUP_COLORS,
)
from blox.env import Board, GameState, Block


def render_b64(state: GameState, start_pos=None) -> str:
    """Render state to a base64 PNG string."""
    svg = render_board_svg(
        state.board,
        block=state.block,
        goal_r=state.goal_r,
        goal_c=state.goal_c,
        active_bridges=state.active_bridges,
        current_height=state.current_height,
        collapsed_crumble=state.collapsed_crumble,
        start_pos=start_pos,
    )
    png_bytes = svg_to_png(svg)
    return base64.b64encode(png_bytes).decode("utf-8")


def _group_switch_color(board: Board, r: int, c: int):
    """Return (tile_color, marker_color) for a switch based on its group."""
    group = board.switches.get((r, c), 0)
    gc = GROUP_COLORS.get(group, GROUP_COLORS[0])
    return gc["switch"], gc["switch_marker"]


def _group_bridge_color(board: Board, r: int, c: int, is_on: bool):
    """Return tile_color for a bridge based on its group and state."""
    group = board.bridges.get((r, c), 0)
    gc = GROUP_COLORS.get(group, GROUP_COLORS[0])
    return gc["bridge_on"] if is_on else gc["bridge_off"]


def _effective_tile(board: Board, r: int, c: int, collapsed_crumble: frozenset) -> int:
    """Return the tile visible at a cell after crumble collapse."""
    return board.effective_tile_at(r, c, collapsed_crumble)


def _darken(hex_color: str, factor: float) -> str:
    """Darken a hex colour by multiplying each RGB channel by *factor* (0–1)."""
    h = hex_color.lstrip("#")
    r = int(min(255, int(h[0:2], 16) * factor))
    g = int(min(255, int(h[2:4], 16) * factor))
    b = int(min(255, int(h[4:6], 16) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def _iso_corner(
    i: float, j: float,
    tw: float, th: float,
    ox: float, oy: float,
) -> Tuple[float, float]:
    """Screen (x, y) for grid corner/point (*i*, *j*) at z = 0.

    Near-overhead with reduced yaw: the grid is only slightly rotated
    (~20°) so the column axis goes mostly rightward and the row axis
    goes mostly downward, like the original Blox perspective.
    """
    # Column axis (j): mostly right, slight down
    # Row axis (i): mostly down, slight left
    import math
    yaw = math.radians(20)  # rotation from aligned grid (0=no rotation, 45=standard iso)
    scale = tw * 0.55       # tile spacing
    # j-axis (columns): rightward
    jx = scale * math.cos(yaw)
    jy = scale * math.sin(yaw) * 0.75  # foreshorten vertical
    # i-axis (rows): downward
    ix = -scale * math.sin(yaw)
    iy = scale * math.cos(yaw) * 0.75
    sx = ox + j * jx + i * ix
    sy = oy + j * jy + i * iy
    return sx, sy


def _pts(points: List[Tuple[float, float]]) -> str:
    """Format a list of (x, y) pairs as an SVG ``points`` attribute value."""
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _lerp_point(a: Tuple[float, float], b: Tuple[float, float], t: float) -> Tuple[float, float]:
    """Linear interpolation between two 2D points."""
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _quad_point(
    tl: Tuple[float, float],
    tr: Tuple[float, float],
    br: Tuple[float, float],
    bl: Tuple[float, float],
    u: float,
    v: float,
) -> Tuple[float, float]:
    """Bilinear interpolation on the tile top face."""
    left = _lerp_point(tl, bl, v)
    right = _lerp_point(tr, br, v)
    return _lerp_point(left, right, u)


CRUMBLE_EDGE_PROFILES = {
    "edge_v1": {
        "left_a": 0.10,
        "left_dent": (0.07, 0.22),
        "top_recover": (0.14, 0.03),
        "top_edge_a": 0.76,
        "top_chip": (0.89, 0.10),
        "right_a": 0.22,
        "right_b": 0.38,
        "right_dent": (0.84, 0.47),
        "right_c": 0.63,
        "front_a": 0.31,
        "front_dent": (0.55, 0.85),
        "front_b": 0.72,
        "flat": [
            (0.00, 0.10), (0.07, 0.22), (0.14, 0.03), (0.76, 0.00),
            (0.89, 0.10), (1.00, 0.22), (1.00, 0.38), (0.84, 0.47),
            (1.00, 0.63), (1.00, 1.00), (0.72, 1.00), (0.55, 0.85),
            (0.31, 1.00), (0.00, 1.00),
        ],
    },
    "edge_v2": {
        "left_a": 0.07,
        "left_dent": (0.08, 0.18),
        "top_recover": (0.16, 0.04),
        "top_edge_a": 0.80,
        "top_chip": (0.86, 0.12),
        "right_a": 0.24,
        "right_b": 0.40,
        "right_dent": (0.82, 0.44),
        "right_c": 0.60,
        "front_a": 0.36,
        "front_dent": (0.52, 0.80),
        "front_b": 0.67,
        "flat": [
            (0.00, 0.07), (0.08, 0.18), (0.16, 0.04), (0.80, 0.00),
            (0.86, 0.12), (1.00, 0.24), (1.00, 0.40), (0.82, 0.44),
            (1.00, 0.60), (1.00, 1.00), (0.67, 1.00), (0.52, 0.80),
            (0.36, 1.00), (0.00, 1.00),
        ],
    },
    "edge_v3": {
        "left_a": 0.12,
        "left_dent": (0.05, 0.24),
        "top_recover": (0.12, 0.05),
        "top_edge_a": 0.72,
        "top_chip": (0.86, 0.08),
        "right_a": 0.19,
        "right_b": 0.34,
        "right_dent": (0.79, 0.42),
        "right_c": 0.57,
        "front_a": 0.27,
        "front_dent": (0.58, 0.82),
        "front_b": 0.76,
        "flat": [
            (0.00, 0.12), (0.05, 0.24), (0.12, 0.05), (0.72, 0.00),
            (0.86, 0.08), (1.00, 0.19), (1.00, 0.34), (0.79, 0.42),
            (1.00, 0.57), (1.00, 1.00), (0.76, 1.00), (0.58, 0.82),
            (0.27, 1.00), (0.00, 1.00),
        ],
    },
}


def _is_crumble_edge_style(style: str) -> bool:
    return style in CRUMBLE_EDGE_PROFILES


def _crumble_edge_profile(style: str) -> dict:
    return CRUMBLE_EDGE_PROFILES.get(style, CRUMBLE_EDGE_PROFILES["edge_v3"])


def _crumble_marker_iso(
    r: int,
    c: int,
    tw: float,
    th: float,
    ox: float,
    oy: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> List[str]:
    tl = _iso_corner(r, c, tw, th, ox, oy)
    tr = _iso_corner(r, c + 1, tw, th, ox, oy)
    br = _iso_corner(r + 1, c + 1, tw, th, ox, oy)
    bl = _iso_corner(r + 1, c, tw, th, ox, oy)

    if x_offset or y_offset:
        tl = (tl[0] + x_offset, tl[1] + y_offset)
        tr = (tr[0] + x_offset, tr[1] + y_offset)
        br = (br[0] + x_offset, br[1] + y_offset)
        bl = (bl[0] + x_offset, bl[1] + y_offset)

    slab = [
        _quad_point(tl, tr, br, bl, 0.12, 0.18),
        _quad_point(tl, tr, br, bl, 0.24, 0.08),
        _quad_point(tl, tr, br, bl, 0.73, 0.11),
        _quad_point(tl, tr, br, bl, 0.86, 0.20),
        _quad_point(tl, tr, br, bl, 0.90, 0.36),
        _quad_point(tl, tr, br, bl, 0.80, 0.54),
        _quad_point(tl, tr, br, bl, 0.89, 0.77),
        _quad_point(tl, tr, br, bl, 0.71, 0.89),
        _quad_point(tl, tr, br, bl, 0.33, 0.87),
        _quad_point(tl, tr, br, bl, 0.14, 0.74),
        _quad_point(tl, tr, br, bl, 0.18, 0.51),
        _quad_point(tl, tr, br, bl, 0.10, 0.31),
    ]

    parts = [
        f'<polygon points="{_pts(slab)}" fill="#d4cec5" '
        f'stroke="{COLOR_CRUMBLE_MARKER}" stroke-width="0.8" opacity="0.88"/>'
    ]

    crack_sets = [
        [(0.26, 0.24), (0.43, 0.38), (0.38, 0.56), (0.53, 0.74)],
        [(0.56, 0.18), (0.62, 0.34), (0.75, 0.45)],
        [(0.46, 0.38), (0.61, 0.56), (0.58, 0.75)],
    ]
    stroke_width = max(0.9, tw * 0.014)
    for crack in crack_sets:
        crack_pts = [_quad_point(tl, tr, br, bl, u, v) for u, v in crack]
        parts.append(
            f'<polyline points="{_pts(crack_pts)}" fill="none" '
            f'stroke="{COLOR_CRUMBLE_MARKER}" stroke-width="{stroke_width:.1f}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="0.56"/>'
        )
    return parts


def _crumble_marker_flat(x: float, y: float, cell: float) -> List[str]:
    slab = _crumble_tile_polygon_flat(x, y, cell)
    parts = [
        f'<polygon points="{_pts(slab)}" fill="#d4cec5" '
        f'stroke="{COLOR_CRUMBLE_MARKER}" stroke-width="{max(0.9, cell * 0.020):.1f}" '
        f'opacity="0.88"/>'
    ]

    crack_sets = [
        [(0.25, 0.25), (0.42, 0.39), (0.37, 0.58), (0.52, 0.74)],
        [(0.57, 0.19), (0.63, 0.35), (0.76, 0.46)],
        [(0.46, 0.39), (0.61, 0.57), (0.58, 0.76)],
    ]
    stroke_width = max(1.1, cell * 0.026)
    for crack in crack_sets:
        crack_pts = [
            (x + cell * px, y + cell * py)
            for px, py in crack
        ]
        parts.append(
            f'<polyline points="{_pts(crack_pts)}" fill="none" '
            f'stroke="{COLOR_CRUMBLE_MARKER}" stroke-width="{stroke_width:.1f}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="0.56"/>'
        )
    return parts


def _crumble_tile_polygon_flat(
    x: float,
    y: float,
    cell: float,
    style: str = "slab",
) -> List[Tuple[float, float]]:
    """Return either the inset slab or the chipped tile silhouette for crumble tiles."""
    if _is_crumble_edge_style(style):
        return [
            (x + cell * px, y + cell * py)
            for px, py in _crumble_edge_profile(style)["flat"]
        ]
    return [
        (x + cell * 0.12, y + cell * 0.18),
        (x + cell * 0.24, y + cell * 0.08),
        (x + cell * 0.73, y + cell * 0.11),
        (x + cell * 0.86, y + cell * 0.20),
        (x + cell * 0.90, y + cell * 0.36),
        (x + cell * 0.80, y + cell * 0.54),
        (x + cell * 0.89, y + cell * 0.77),
        (x + cell * 0.71, y + cell * 0.89),
        (x + cell * 0.33, y + cell * 0.87),
        (x + cell * 0.14, y + cell * 0.74),
        (x + cell * 0.18, y + cell * 0.51),
        (x + cell * 0.10, y + cell * 0.31),
    ]


def _tile_faces(
    r: int, c: int,
    tw: float, th: float,
    ox: float, oy: float,
    thickness: float,
    top_col: str, left_col: str, right_col: str,
    stroke: Optional[str] = None,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    edge_damage: bool = False,
    edge_damage_variant: str = "edge_v3",
) -> List[str]:
    """Return SVG ``<polygon>`` strings for a thin isometric tile cuboid.

    *x_offset* / *y_offset* shift the tile in screen space (used for
    the exploded-view rendering of elevated tiles).
    """
    tl = _iso_corner(r, c, tw, th, ox, oy)
    tr = _iso_corner(r, c + 1, tw, th, ox, oy)
    br = _iso_corner(r + 1, c + 1, tw, th, ox, oy)
    bl = _iso_corner(r + 1, c, tw, th, ox, oy)

    # Apply screen-space offset (exploded view)
    if x_offset or y_offset:
        tl = (tl[0] + x_offset, tl[1] + y_offset)
        tr = (tr[0] + x_offset, tr[1] + y_offset)
        br = (br[0] + x_offset, br[1] + y_offset)
        bl = (bl[0] + x_offset, bl[1] + y_offset)

    stk = f' stroke="{stroke}" stroke-width="0.5"' if stroke else ""
    parts: List[str] = []

    if edge_damage:
        profile = _crumble_edge_profile(edge_damage_variant)
        left_a = _lerp_point(tl, bl, profile["left_a"])
        left_dent = _quad_point(tl, tr, br, bl, *profile["left_dent"])
        top_recover = _quad_point(tl, tr, br, bl, *profile["top_recover"])
        top_edge_a = _lerp_point(tl, tr, profile["top_edge_a"])
        top_chip = _quad_point(tl, tr, br, bl, *profile["top_chip"])
        right_a = _lerp_point(tr, br, profile["right_a"])
        right_b = _lerp_point(tr, br, profile["right_b"])
        right_dent = _quad_point(tl, tr, br, bl, *profile["right_dent"])
        right_c = _lerp_point(tr, br, profile["right_c"])
        front_a = _lerp_point(bl, br, profile["front_a"])
        front_dent = _quad_point(tl, tr, br, bl, *profile["front_dent"])
        front_b = _lerp_point(bl, br, profile["front_b"])

        right_edge = [right_a, right_b, right_dent, right_c, br]
        front_edge = [bl, front_a, front_dent, front_b, br]
        right_bottom = [(x, y + thickness) for x, y in reversed(right_edge)]
        front_bottom = [(x, y + thickness) for x, y in reversed(front_edge)]
        top_face = [
            left_a,
            left_dent,
            top_recover,
            top_edge_a,
            top_chip,
            right_a,
            right_b,
            right_dent,
            right_c,
            br,
            front_b,
            front_dent,
            front_a,
            bl,
        ]

        parts.append(
            f'<polygon points="{_pts(front_edge + front_bottom)}" fill="{left_col}"{stk}/>'
        )
        parts.append(
            f'<polygon points="{_pts(right_edge + right_bottom)}" fill="{right_col}"{stk}/>'
        )
        parts.append(
            f'<polygon points="{_pts(top_face)}" fill="{top_col}"{stk}/>'
        )
        return parts

    # Left visible face (high-r edge — faces the viewer on the left)
    parts.append(
        f'<polygon points="{_pts([bl, br, (br[0], br[1] + thickness), (bl[0], bl[1] + thickness)])}" '
        f'fill="{left_col}"{stk}/>'
    )
    # Right visible face (high-c edge — faces the viewer on the right)
    parts.append(
        f'<polygon points="{_pts([tr, br, (br[0], br[1] + thickness), (tr[0], tr[1] + thickness)])}" '
        f'fill="{right_col}"{stk}/>'
    )
    # Top face
    parts.append(
        f'<polygon points="{_pts([tl, tr, br, bl])}" fill="{top_col}"{stk}/>'
    )
    return parts


def _block_faces(
    r: int, c: int,
    dr: int, dc: int,
    height: float,
    tw: float, th: float,
    ox: float, oy: float,
    top_col: str, left_col: str, right_col: str,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> List[str]:
    """Return SVG ``<polygon>`` strings for a 3-D block cuboid.

    The block sits on the tile surface and extends upward by *height*
    screen-pixels.  It spans *dr* rows x *dc* columns starting at (*r*, *c*).
    *x_offset* / *y_offset* shift the block for exploded-view elevated rendering.
    """
    # Ground-level corners (z = 0)
    g_tl = _iso_corner(r, c, tw, th, ox, oy)
    g_tr = _iso_corner(r, c + dc, tw, th, ox, oy)
    g_br = _iso_corner(r + dr, c + dc, tw, th, ox, oy)
    g_bl = _iso_corner(r + dr, c, tw, th, ox, oy)

    # Apply screen-space offset for elevated platform
    if x_offset or y_offset:
        g_tl = (g_tl[0] + x_offset, g_tl[1] + y_offset)
        g_tr = (g_tr[0] + x_offset, g_tr[1] + y_offset)
        g_br = (g_br[0] + x_offset, g_br[1] + y_offset)
        g_bl = (g_bl[0] + x_offset, g_bl[1] + y_offset)

    # Raised corners (z = height → shift up in screen-y)
    t_tl = (g_tl[0], g_tl[1] - height)
    t_tr = (g_tr[0], g_tr[1] - height)
    t_br = (g_br[0], g_br[1] - height)
    t_bl = (g_bl[0], g_bl[1] - height)

    stk = ' stroke="#1a3a5c" stroke-width="1" stroke-linejoin="round"'
    opa = ' opacity="0.75"'
    parts: List[str] = []

    # Left face (high-r edge)
    parts.append(
        f'<polygon points="{_pts([t_bl, t_br, g_br, g_bl])}" '
        f'fill="{left_col}"{stk}{opa}/>'
    )
    # Right face (high-c edge)
    parts.append(
        f'<polygon points="{_pts([t_tr, t_br, g_br, g_tr])}" '
        f'fill="{right_col}"{stk}{opa}/>'
    )
    # Top face
    parts.append(
        f'<polygon points="{_pts([t_tl, t_tr, t_br, t_bl])}" '
        f'fill="{top_col}"{stk}{opa}/>'
    )
    return parts


def render_board_svg(
    board: Board,
    block: Optional[Block] = None,
    goal_r: Optional[int] = None,
    goal_c: Optional[int] = None,
    active_bridges: frozenset = frozenset(),
    collapsed_crumble: frozenset = frozenset(),
    cell: int = 60,
    margin: int = 8,
    show_coordinates: bool = True,
    show_legend: bool = True,
    current_height: int = 0,
    start_pos: Optional[Tuple[int, int]] = None,
    focus_floor: Optional[int] = None,
    elevator_states: Optional[dict] = None,
    crumble_style: str = "edge_v3",
) -> str:
    """Render a Blox board as an **isometric 3-D** SVG string.

    The function signature is intentionally identical to the previous flat
    renderer so that callers (env.py, eval scripts) need no changes.
    """
    # -- geometry ----------------------------------------------------------
    tw = float(cell)            # tile width  (horizontal diamond span)
    th = tw / 2                 # tile height (2 : 1 isometric ratio)
    tile_thick = th * 0.20      # edge depth
    elevation_lift = th * 1.8   # default; adjusted below for easy levels
    elevation_dx = 0.0         # no horizontal shift (pure vertical separation)
    unit_h = th * 0.5           # block cube height
    standing_h = 2 * unit_h     # standing: two cubes stacked
    lying_h = 1 * unit_h        # lying: one cube tall

    has_elevators = bool(board.elevators)
    has_multi_elevator_view = len(board.elevators) >= 2
    has_height_map = bool(board.height_map)

    rows, cols = board.rows, board.cols

    # -- canvas layout -----------------------------------------------------
    import math
    yaw = math.radians(20)
    scale = tw * 0.55
    jx, jy = scale * math.cos(yaw), scale * math.sin(yaw) * 0.75
    ix, iy = -scale * math.sin(yaw), scale * math.cos(yaw) * 0.75
    corners_x = [j * jx + i * ix for i in [0, rows] for j in [0, cols]]
    corners_y = [j * jy + i * iy for i in [0, rows] for j in [0, cols]]
    grid_w = max(corners_x) - min(corners_x)
    grid_h = max(corners_y) - min(corners_y)
    # Headroom: reserve space for elevation offsets (tight)
    if focus_floor == 0:
        extra_top = standing_h + tile_thick
    elif focus_floor is None and has_height_map:
        extra_top = standing_h + tile_thick + elevation_lift * 2.5
    else:
        extra_top = standing_h + tile_thick + elevation_lift
    coord_pad = 12 if show_coordinates else 0

    legend_w = 0  # legend now rendered as horizontal bar below the board
    legend_gap = 0

    # Origin = screen position of grid corner (0, 0)
    ox = coord_pad + margin - min(corners_x)
    oy = margin + extra_top - min(corners_y)

    legend_bar_h = 28 if show_legend else 0  # horizontal legend bar below board

    extra_right = elevation_dx + 10 if has_height_map else 0  # room for horizontal explode
    extra_bottom = 25 if show_coordinates else 0  # room for column tick labels
    total_w = coord_pad + margin * 2 + grid_w + extra_right
    total_h = margin * 2 + extra_top + grid_h + tile_thick + extra_bottom + legend_bar_h

    parts: List[str] = []
    parts.append(f'<rect width="{total_w}" height="{total_h}" fill="{COLOR_BG}"/>')

    # -- coordinate labels (drawn first so tiles/block render on top) ------
    if show_coordinates:
        fs = max(10, tw * 0.20)
        # Shift labels to match elevated tiles when focusing on HIGH floor
        label_dy = -elevation_lift if focus_floor == 1 else 0.0
        # Column labels along the bottom-left edge (Left→Bottom vertex)
        for c in range(cols):
            lx, ly = _iso_corner(rows + 0.35, c + 0.5, tw, th, ox, oy)
            ly += tile_thick + label_dy  # below the tile edge
            parts.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" font-family="monospace" '
                f'font-size="{fs:.1f}" text-anchor="middle" '
                f'dominant-baseline="central" fill="{COLOR_TEXT}">{c}</text>'
            )
        # Row labels along the bottom-right edge (avoids overlap with elevated tiles)
        for r in range(rows):
            lx, ly = _iso_corner(r + 0.5, cols + 0.4, tw, th, ox, oy)
            ly += tile_thick + label_dy
            parts.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" font-family="monospace" '
                f'font-size="{fs:.1f}" text-anchor="middle" '
                f'dominant-baseline="central" fill="{COLOR_TEXT}">{r}</text>'
            )

    # -- block metadata for in-loop drawing --------------------------------
    block_cells: set = set()
    block_trigger: Optional[Tuple[int, int]] = None
    if block is not None:
        block_cells = set(block.cells())
        # The trigger cell is the block cell drawn last in depth order so that
        # the full cuboid is painted on top of every tile it overlaps.
        if block.orientation == Orientation.STANDING:
            block_trigger = (block.r, block.c)
        elif block.orientation == Orientation.LYING_H:
            block_trigger = (block.r, block.c + 1)
        else:  # LYING_V
            block_trigger = (block.r + 1, block.c)

    # Precompute which LOW cells belong to the "back island" (block's start region)
    # Used in combined view to lift the back island above the HIGH band.
    _back_island = set()
    _min_high_depth = 999  # fallback
    if has_height_map and focus_floor is None:
        if has_multi_elevator_view:
            from collections import deque as _dq
            _low_cells = {(r, c) for r in range(rows) for c in range(cols)
                          if board.cell_height(r, c) == 0 and board.grid[r][c] != TileType.VOID}
            # Use explicit start_pos, or fall back to block position.
            _start = start_pos or (block.r, block.c) if block else None
            if _start and _start in _low_cells:
                q = _dq([_start])
                while q:
                    cr, cc = q.popleft()
                    if (cr, cc) in _back_island:
                        continue
                    _back_island.add((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (nr, nc) in _low_cells and (nr, nc) not in _back_island:
                            q.append((nr, nc))
            # Only use three-tier lifting when the LOW floor really splits into
            # two substantial regions. Single-elevator layouts stay strictly
            # two-tier so the render matches the game's LOW/HIGH semantics.
            other_low = _low_cells - _back_island
            if len(other_low) < 3:
                _back_island = set()  # no meaningful second island — easy level
        _min_high_depth = min(
            (r + c for r in range(rows) for c in range(cols)
             if board.cell_height(r, c) == 1),
            default=999
        )
        # Easy levels (no back island): use smaller elevation to avoid occlusion
        if not _back_island:
            elevation_lift = th * 1.2

    # -- draw tiles & block (back-to-front, painter's algorithm) -----------
    # Depth key = r + c.  Within a diagonal, iterate increasing r.
    # When showing both floors (focus_floor is None), draw elevated tiles
    # first at each depth so LOW tiles paint on top (LOW always in front).
    for d in range(rows + cols - 1):
        cells_at_depth = [(r, d - r) for r in range(max(0, d - cols + 1), min(rows, d + 1))]
        if focus_floor is None and has_height_map:
            # Sort: elevated (height=1) first, then ground (height=0)
            cells_at_depth.sort(key=lambda rc: -board.cell_height(rc[0], rc[1]))
        for r, c in cells_at_depth:
            tile = _effective_tile(board, r, c, collapsed_crumble)

            # Determine if this is a void-like tile (VOID or inactive BRIDGE)
            is_bridge_off = (
                tile == TileType.BRIDGE
                and board.bridges.get((r, c)) not in active_bridges
            )

            # Determine which floor this tile belongs to
            tile_floor = 0
            if has_height_map and board.cell_height(r, c) == 1:
                tile_floor = 1
            is_elevator_cell = (r, c) in board.elevators
            # Hide non-focused floor tiles (elevators always visible)
            hidden = (focus_floor is not None
                      and tile_floor != focus_floor
                      and not is_elevator_cell)

            if tile == TileType.VOID or is_bridge_off:
                # Recessed flat diamond to hint at grid structure
                tl = _iso_corner(r, c, tw, th, ox, oy)
                tr = _iso_corner(r, c + 1, tw, th, ox, oy)
                br = _iso_corner(r + 1, c + 1, tw, th, ox, oy)
                bl_pt = _iso_corner(r + 1, c, tw, th, ox, oy)
                off = tile_thick
                pts = [
                    (tl[0], tl[1] + off),
                    (tr[0], tr[1] + off),
                    (br[0], br[1] + off),
                    (bl_pt[0], bl_pt[1] + off),
                ]
                if not hidden:
                    void_color = _group_bridge_color(board, r, c, is_on=False) if is_bridge_off else COLOR_VOID
                    void_opacity = "0.6" if is_bridge_off else "0.4"
                    parts.append(
                        f'<polygon points="{_pts(pts)}" '
                        f'fill="{void_color}" opacity="{void_opacity}"/>'
                    )
                # Dashed diamond marker for inactive bridges
                if is_bridge_off:
                    cx, cy = _iso_corner(r + 0.5, c + 0.5, tw, th, ox, oy)
                    ds = tw * 0.08
                    marker = [
                        (cx, cy - ds + off), (cx + ds, cy + off),
                        (cx, cy + ds + off), (cx - ds, cy + off),
                    ]
                    parts.append(
                        f'<polygon points="{_pts(marker)}" fill="none" '
                        f'stroke="{COLOR_BRIDGE_OFF}" stroke-width="1" '
                        f'stroke-dasharray="2,2"/>'
                    )
            else:
                # Determine if this tile is elevated or an elevator
                is_elevated = has_height_map and board.cell_height(r, c) == 1
                is_elevator = (r, c) in board.elevators

                # Raised tile — pick color by type
                if tile == TileType.GOAL:
                    tc = COLOR_GOAL
                    lc = _darken(COLOR_GOAL, 0.65)
                    rc = _darken(COLOR_GOAL, 0.80)
                elif tile == TileType.SWITCH:
                    tc, _ = _group_switch_color(board, r, c)
                    lc = _darken(tc, 0.65)
                    rc = _darken(tc, 0.80)
                elif tile == TileType.BRIDGE:
                    # Active bridge
                    tc = _group_bridge_color(board, r, c, is_on=True)
                    lc = _darken(tc, 0.65)
                    rc = _darken(tc, 0.80)
                elif tile == TileType.FRAGILE:
                    tc = COLOR_FRAGILE
                    lc = _darken(COLOR_FRAGILE, 0.70)
                    rc = _darken(COLOR_FRAGILE, 0.85)
                elif tile == TileType.CRUMBLE:
                    tc = COLOR_CRUMBLE
                    lc = _darken(COLOR_CRUMBLE, 0.70)
                    rc = _darken(COLOR_CRUMBLE, 0.85)
                elif is_elevator:
                    tc = COLOR_ELEVATOR
                    lc = _darken(COLOR_ELEVATOR, 0.65)
                    rc = _darken(COLOR_ELEVATOR, 0.80)
                elif is_elevated:
                    tc = COLOR_ELEVATED
                    lc = _darken(COLOR_ELEVATED, 0.65)
                    rc = _darken(COLOR_ELEVATED, 0.80)
                else:
                    tc = COLOR_NORMAL
                    lc = _darken(COLOR_NORMAL, 0.70)
                    rc = _darken(COLOR_NORMAL, 0.85)

                # Exploded-view offsets: three tiers in combined view.
                # Back LOW island (behind HIGH band): highest lift
                # HIGH band: middle lift
                # Front LOW island: ground level (no lift)
                if focus_floor is None and has_height_map:
                    # Combined view: determine which tier this tile belongs to
                    if is_elevated and not is_elevator:
                        tile_x_off = elevation_dx
                        tile_y_off = -elevation_lift
                    elif is_elevator:
                        # Determine if this is Elevator 1 (back, near start)
                        # or Elevator 2 (front, near goal)
                        is_back_elevator = any(
                            (r+dr, c+dc) in _back_island
                            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                        ) if _back_island else False
                        # Each elevator has its own state (toggled or not)
                        e_toggled = (elevator_states or {}).get((r, c), 0)
                        if not _back_island:
                            # Easy levels: elevator follows current_height
                            if current_height == 1:
                                tile_x_off = elevation_dx
                                tile_y_off = -elevation_lift
                            else:
                                tile_x_off = 0.0
                                tile_y_off = 0.0
                        elif is_back_elevator:
                            if e_toggled == 0:
                                # Initial: at back LOW island
                                tile_x_off = elevation_dx
                                tile_y_off = -elevation_lift * 2.5
                            else:
                                # Toggled: at HIGH band
                                tile_x_off = elevation_dx
                                tile_y_off = -elevation_lift
                        else:
                            if e_toggled == 0:
                                # Initial: at HIGH band
                                tile_x_off = elevation_dx
                                tile_y_off = -elevation_lift
                            else:
                                # Toggled: at front LOW (ground)
                                tile_x_off = 0.0
                                tile_y_off = 0.0
                    elif (r, c) in _back_island:
                        # Back LOW island
                        tile_x_off = elevation_dx
                        tile_y_off = -elevation_lift * 2.5
                    else:
                        # Front LOW island
                        tile_x_off = 0.0
                        tile_y_off = 0.0
                else:
                    # Split view or no height map
                    if is_elevated and not is_elevator:
                        tile_x_off = elevation_dx
                        tile_y_off = -elevation_lift
                    elif is_elevator and current_height == 1:
                        tile_x_off = elevation_dx
                        tile_y_off = -elevation_lift
                    else:
                        tile_x_off = 0.0
                        tile_y_off = 0.0
                this_thick = tile_thick

                if not hidden:
                    # Elevator rendering depends on where the platform
                    # physically is: ghost where it's absent, solid where present.
                    # current_height==0 → platform on LOW; ==1 → platform on HIGH
                    elevator_ghost = (is_elevator_cell and focus_floor is not None
                                      and focus_floor != current_height)
                    if elevator_ghost:
                        tl = _iso_corner(r, c, tw, th, ox, oy)
                        tr = _iso_corner(r, c + 1, tw, th, ox, oy)
                        br = _iso_corner(r + 1, c + 1, tw, th, ox, oy)
                        bl_pt = _iso_corner(r + 1, c, tw, th, ox, oy)
                        # Draw ghost at the focus floor's level
                        elev_off = -elevation_lift if focus_floor == 1 else 0
                        ghost_pts = [
                            (tl[0], tl[1] + elev_off),
                            (tr[0], tr[1] + elev_off),
                            (br[0], br[1] + elev_off),
                            (bl_pt[0], bl_pt[1] + elev_off),
                        ]
                        parts.append(
                            f'<polygon points="{_pts(ghost_pts)}" '
                            f'fill="none" stroke="{COLOR_ELEVATOR}" '
                            f'stroke-width="1.5" stroke-dasharray="4,3" '
                            f'opacity="0.7"/>'
                        )
                        # Arrow marker inside the ghost outline
                        cx, cy = _iso_corner(r + 0.5, c + 0.5, tw, th, ox, oy)
                        cy += elev_off
                        fs = tw * 0.28
                        parts.append(
                            f'<text x="{cx:.1f}" y="{cy + fs * 0.15:.1f}" '
                            f'font-family="monospace" font-size="{fs:.1f}" '
                            f'text-anchor="middle" dominant-baseline="central" '
                            f'fill="{COLOR_ELEVATOR_MARKER}" opacity="0.5">'
                            f'\u25b2\u25bc</text>'
                        )
                    else:
                        parts.extend(
                            _tile_faces(r, c, tw, th, ox, oy, this_thick,
                                        tc, lc, rc, stroke="#444466",
                                        x_offset=tile_x_off, y_offset=tile_y_off,
                                        edge_damage=(tile == TileType.CRUMBLE and _is_crumble_edge_style(crumble_style)),
                                        edge_damage_variant=crumble_style)
                        )

                        # Goal marker (small diamond on the tile surface)
                        if tile == TileType.GOAL:
                            cx, cy = _iso_corner(r + 0.5, c + 0.5, tw, th, ox, oy)
                            cx += tile_x_off; cy += tile_y_off
                            ds = tw * 0.12
                            marker = [
                                (cx, cy - ds),
                                (cx + ds * 0.8, cy),
                                (cx, cy + ds),
                                (cx - ds * 0.8, cy),
                            ]
                            parts.append(
                                f'<polygon points="{_pts(marker)}" '
                                f'fill="{COLOR_GOAL_DIAMOND}" opacity="0.85"/>'
                            )
                        # Switch marker (circle on the tile surface)
                        elif tile == TileType.SWITCH:
                            _, marker_color = _group_switch_color(board, r, c)
                            cx, cy = _iso_corner(r + 0.5, c + 0.5, tw, th, ox, oy)
                            cx += tile_x_off; cy += tile_y_off
                            radius = tw * 0.10
                            parts.append(
                                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
                                f'fill="{marker_color}" opacity="0.85"/>'
                            )
                        # Fragile marker (X / crack lines on the tile surface)
                        elif tile == TileType.FRAGILE:
                            cx, cy = _iso_corner(r + 0.5, c + 0.5, tw, th, ox, oy)
                            cx += tile_x_off; cy += tile_y_off
                            ds = tw * 0.10
                            parts.append(
                                f'<line x1="{cx - ds:.1f}" y1="{cy - ds * 0.6:.1f}" '
                                f'x2="{cx + ds:.1f}" y2="{cy + ds * 0.6:.1f}" '
                                f'stroke="{COLOR_FRAGILE_MARKER}" stroke-width="1.5" opacity="0.8"/>'
                            )
                            parts.append(
                                f'<line x1="{cx + ds:.1f}" y1="{cy - ds * 0.6:.1f}" '
                                f'x2="{cx - ds:.1f}" y2="{cy + ds * 0.6:.1f}" '
                                f'stroke="{COLOR_FRAGILE_MARKER}" stroke-width="1.5" opacity="0.8"/>'
                            )
                        elif tile == TileType.CRUMBLE and crumble_style == "slab":
                            parts.extend(
                                _crumble_marker_iso(
                                    r,
                                    c,
                                    tw,
                                    th,
                                    ox,
                                    oy,
                                    x_offset=tile_x_off,
                                    y_offset=tile_y_off,
                                )
                            )
                        elif is_elevator:
                            cx, cy = _iso_corner(r + 0.5, c + 0.5, tw, th, ox, oy)
                            cx += tile_x_off; cy += tile_y_off
                            fs = tw * 0.28
                            parts.append(
                                f'<text x="{cx:.1f}" y="{cy + fs * 0.15:.1f}" '
                                f'font-family="monospace" font-size="{fs:.1f}" '
                                f'text-anchor="middle" dominant-baseline="central" '
                                f'fill="{COLOR_ELEVATOR_MARKER}" opacity="0.9">'
                                f'\u25b2\u25bc</text>'
                            )
                            # Ghost marker on the other floor (combined view only)
                            if focus_floor is None and has_height_map:
                                # Ghost at the opposite level from the solid tile
                                if is_back_elevator:
                                    if e_toggled == 0:
                                        ghost_target_x = elevation_dx
                                        ghost_target_y = -elevation_lift
                                    else:
                                        ghost_target_x = elevation_dx
                                        ghost_target_y = -elevation_lift * 2.5
                                else:
                                    if e_toggled == 0:
                                        ghost_target_x = 0.0
                                        ghost_target_y = 0.0
                                    else:
                                        ghost_target_x = elevation_dx
                                        ghost_target_y = -elevation_lift
                                # Ghost diamond at target level (raw corners + ghost_target)
                                gtl = _iso_corner(r, c, tw, th, ox, oy)
                                gtr = _iso_corner(r, c + 1, tw, th, ox, oy)
                                gbr = _iso_corner(r + 1, c + 1, tw, th, ox, oy)
                                gbl = _iso_corner(r + 1, c, tw, th, ox, oy)
                                ghost_pts = [
                                    (gtl[0] + ghost_target_x, gtl[1] + ghost_target_y),
                                    (gtr[0] + ghost_target_x, gtr[1] + ghost_target_y),
                                    (gbr[0] + ghost_target_x, gbr[1] + ghost_target_y),
                                    (gbl[0] + ghost_target_x, gbl[1] + ghost_target_y),
                                ]
                                # Ghost text position
                                ghost_x = _iso_corner(r + 0.5, c + 0.5, tw, th, ox, oy)[0] + ghost_target_x
                                ghost_y = _iso_corner(r + 0.5, c + 0.5, tw, th, ox, oy)[1] + ghost_target_y
                                parts.append(
                                    f'<polygon points="{_pts(ghost_pts)}" '
                                    f'fill="{COLOR_ELEVATOR}" fill-opacity="0.15" '
                                    f'stroke="{COLOR_ELEVATOR}" stroke-width="1.5" '
                                    f'stroke-dasharray="4,3" opacity="0.6"/>'
                                )
                                parts.append(
                                    f'<text x="{ghost_x:.1f}" y="{ghost_y + fs * 0.15:.1f}" '
                                    f'font-family="monospace" font-size="{fs * 0.8:.1f}" '
                                    f'text-anchor="middle" dominant-baseline="central" '
                                    f'fill="{COLOR_ELEVATOR_MARKER}" opacity="0.5">'
                                    f'\u25b2\u25bc</text>'
                                )

            # Draw block cuboid when we reach its trigger cell
            if block_trigger is not None and (r, c) == block_trigger:
                if block.orientation == Orientation.STANDING:
                    b_dr, b_dc, bh = 1, 1, standing_h
                elif block.orientation == Orientation.LYING_H:
                    b_dr, b_dc, bh = 1, 2, lying_h
                else:  # LYING_V
                    b_dr, b_dc, bh = 2, 1, lying_h

                # Block offset: match the tile offset of the block's position
                if focus_floor is None and has_height_map:
                    if current_height == 1:
                        block_x_off = elevation_dx
                        block_y_off = -elevation_lift
                    elif (block.r, block.c) in _back_island:
                        # Block on back LOW island (lifted tiles)
                        block_x_off = elevation_dx
                        block_y_off = -elevation_lift * 2.5
                    else:
                        block_x_off = 0.0
                        block_y_off = 0.0
                elif has_height_map and current_height == 1:
                    block_x_off = elevation_dx
                    block_y_off = -elevation_lift
                else:
                    block_x_off = 0.0
                    block_y_off = 0.0

                # Hide block when on non-focused floor
                block_hidden = (focus_floor is not None
                                and current_height != focus_floor)
                if not block_hidden:
                    parts.extend(
                        _block_faces(
                            block.r, block.c, b_dr, b_dc, bh,
                            tw, th, ox, oy,
                            COLOR_BLOCK,
                            _darken(COLOR_BLOCK, 0.60),
                            _darken(COLOR_BLOCK, 0.78),
                            x_offset=block_x_off,
                            y_offset=block_y_off,
                        )
                    )


    # -- legend bar (horizontal, below isometric grid) ---------------------
    if show_legend:
        items = [
            (COLOR_NORMAL, "Normal"),
            (COLOR_VOID, "Void"),
            (COLOR_GOAL, "Goal"),
            (COLOR_BLOCK, "Block"),
        ]
        has_fragile = any(
            board.grid[r][c] == TileType.FRAGILE
            for r in range(board.rows) for c in range(board.cols)
        )
        has_crumble = any(
            board.grid[r][c] == TileType.CRUMBLE and (r, c) not in collapsed_crumble
            for r in range(board.rows) for c in range(board.cols)
        )
        if has_fragile:
            items.append((COLOR_FRAGILE, "Fragile"))
        if has_crumble:
            items.append((COLOR_CRUMBLE, "Crumble"))
        if has_elevators:
            items.append((COLOR_ELEVATOR, "Elevator"))
        if board.switches:
            groups = sorted(set(board.switches.values()))
            for g in groups:
                gc = GROUP_COLORS.get(g, GROUP_COLORS[0])
                label = f"Switch (grp {g})" if len(groups) > 1 else "Switch"
                items.append((gc["switch"], label))
        if board.bridges:
            groups = sorted(set(board.bridges.values()))
            for g in groups:
                gc = GROUP_COLORS.get(g, GROUP_COLORS[0])
                label_on = f"Bridge ON (grp {g})" if len(groups) > 1 else "Bridge (on)"
                label_off = f"Bridge OFF (grp {g})" if len(groups) > 1 else "Bridge (off)"
                items.append((gc["bridge_on"], label_on))
                items.append((gc["bridge_off"], label_off))

        # Lay out items horizontally, centered at the bottom
        bar_y = total_h - legend_bar_h + 6
        # Draw a full-width subtle background strip so autocrop preserves it
        parts.append(
            f'<rect x="0" y="{total_h - legend_bar_h}" width="{total_w}" '
            f'height="{legend_bar_h}" fill="#1e1e30"/>'
        )
        swatch_w, gap_after_swatch, item_gap = 8, 3, 10
        # Measure total width needed
        item_widths = [swatch_w + gap_after_swatch + len(label) * 4.8 for _, label in items]
        total_legend_w = sum(item_widths) + item_gap * (len(items) - 1)
        ix = (total_w - total_legend_w) / 2  # center
        for i, (color, label) in enumerate(items):
            parts.append(
                f'<rect x="{ix:.0f}" y="{bar_y}" width="{swatch_w}" height="8" '
                f'fill="{color}" rx="1"/>'
            )
            parts.append(
                f'<text x="{ix + swatch_w + gap_after_swatch:.0f}" y="{bar_y + 7}" '
                f'font-family="monospace" font-size="7" fill="{COLOR_TEXT}">{label}</text>'
            )
            ix += item_widths[i] + item_gap


    # -- assemble SVG ------------------------------------------------------
    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}">'
    )
    return "\n".join([header] + parts + ["</svg>"])


def render_board_svg_split(
    board: Board,
    block: Optional[Block] = None,
    goal_r: Optional[int] = None,
    goal_c: Optional[int] = None,
    active_bridges: frozenset = frozenset(),
    collapsed_crumble: frozenset = frozenset(),
    cell: int = 60,
    margin: int = 25,
    show_coordinates: bool = True,
    current_height: int = 0,
    crumble_style: str = "edge_v3",
) -> str:
    """Render two side-by-side panels: LOW floor focused | HIGH floor focused.

    Each panel shows its floor at full opacity and the other floor ghosted
    at 20% opacity.  Elevators are always fully visible (they connect floors).
    """
    # Render each panel with tight margins (no legend)
    panel_margin = 10
    svg_low = render_board_svg(
        board, block, goal_r, goal_c, active_bridges,
        collapsed_crumble=collapsed_crumble,
        cell=cell, margin=panel_margin, show_coordinates=show_coordinates,
        show_legend=False, current_height=current_height, focus_floor=0,
        crumble_style=crumble_style,
    )
    svg_high = render_board_svg(
        board, block, goal_r, goal_c, active_bridges,
        collapsed_crumble=collapsed_crumble,
        cell=cell, margin=panel_margin, show_coordinates=show_coordinates,
        show_legend=False, current_height=current_height, focus_floor=1,
        crumble_style=crumble_style,
    )

    # Parse dimensions from each SVG
    def _parse_dims(svg: str):
        m = re.search(r'width="([0-9.]+)"\s+height="([0-9.]+)"', svg)
        return float(m.group(1)), float(m.group(2))

    def _strip_wrapper(svg: str) -> str:
        """Extract inner content (between <svg ...> and </svg>)."""
        start = svg.index(">") + 1
        end = svg.rindex("</svg>")
        return svg[start:end]

    w_low, h_low = _parse_dims(svg_low)
    w_high, h_high = _parse_dims(svg_high)

    gap = 15  # gap between panels
    label_h = 14  # height for floor labels
    outer_pad = 8  # outer margin around the whole composite
    panel_h = max(h_low, h_high)
    total_w = outer_pad * 2 + w_low + gap + w_high
    total_h = outer_pad + label_h + panel_h

    parts: List[str] = []
    parts.append(f'<rect width="{total_w}" height="{total_h}" fill="{COLOR_BG}"/>')

    # Floor labels
    fs = 11
    parts.append(
        f'<text x="{outer_pad + w_low / 2}" y="{outer_pad + fs}" font-family="monospace" '
        f'font-size="{fs}" text-anchor="middle" fill="{COLOR_TEXT}" '
        f'font-weight="bold">LOW floor</text>'
    )
    parts.append(
        f'<text x="{outer_pad + w_low + gap + w_high / 2}" y="{outer_pad + fs}" '
        f'font-family="monospace" font-size="{fs}" text-anchor="middle" '
        f'fill="{COLOR_TEXT}" font-weight="bold">HIGH floor</text>'
    )

    # Vertical separator between panels
    sep_x = outer_pad + w_low + gap / 2
    parts.append(
        f'<line x1="{sep_x:.1f}" y1="{outer_pad}" '
        f'x2="{sep_x:.1f}" y2="{total_h - 2}" '
        f'stroke="{COLOR_TEXT}" stroke-width="0.8" opacity="0.5"/>'
    )

    # Left panel (LOW focus)
    parts.append(f'<g transform="translate({outer_pad},{outer_pad + label_h})">')
    parts.append(_strip_wrapper(svg_low))
    parts.append('</g>')

    # Right panel (HIGH focus)
    parts.append(f'<g transform="translate({outer_pad + w_low + gap},{outer_pad + label_h})">')
    parts.append(_strip_wrapper(svg_high))
    parts.append('</g>')

    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}">'
    )
    return "\n".join([header] + parts + ["</svg>"])


def render_board_svg_flat(
    board: Board,
    block: Optional[Block] = None,
    goal_r: Optional[int] = None,
    goal_c: Optional[int] = None,
    active_bridges: frozenset = frozenset(),
    collapsed_crumble: frozenset = frozenset(),
    cell: int = 40,
    margin: int = 20,
    show_coordinates: bool = True,
    show_legend: bool = True,
    current_height: int = 0,
    crumble_style: str = "edge_v3",
) -> str:
    """Render a Blox board as a **flat top-down** SVG string.

    Same visual language (colors, symbols) as the isometric renderer,
    but with no perspective distortion — a simple 2D grid.
    """
    rows, cols = board.rows, board.cols
    coord_pad = 18 if show_coordinates else 0
    grid_w = cols * cell
    grid_h = rows * cell
    legend_bar_h = 24 if show_legend else 0
    total_w = margin * 2 + coord_pad + grid_w
    total_h = margin * 2 + coord_pad + grid_h + legend_bar_h

    # Grid origin (top-left corner of cell 0,0)
    gx = margin + coord_pad
    gy = margin + coord_pad

    block_cells: set = set()
    if block is not None:
        block_cells = set(block.cells())

    parts: List[str] = []
    parts.append(f'<rect width="{total_w}" height="{total_h}" fill="{COLOR_BG}"/>')

    # Coordinate labels
    if show_coordinates:
        fs = max(10, cell * 0.30)
        for c in range(cols):
            cx = gx + c * cell + cell / 2
            cy = gy - 6
            parts.append(
                f'<text x="{cx:.1f}" y="{cy:.1f}" font-family="monospace" '
                f'font-size="{fs:.1f}" text-anchor="middle" fill="{COLOR_TEXT}">{c}</text>'
            )
        for r in range(rows):
            cx = gx - 6
            cy = gy + r * cell + cell / 2
            parts.append(
                f'<text x="{cx:.1f}" y="{cy:.1f}" font-family="monospace" '
                f'font-size="{fs:.1f}" text-anchor="end" dominant-baseline="central" '
                f'fill="{COLOR_TEXT}">{r}</text>'
            )

    # Draw cells
    for r in range(rows):
        for c in range(cols):
            x = gx + c * cell
            y = gy + r * cell
            tile = _effective_tile(board, r, c, collapsed_crumble)

            is_elevator = (r, c) in board.elevators
            is_elevated = bool(board.height_map) and board.cell_height(r, c) == 1

            if (r, c) in block_cells:
                fill = COLOR_BLOCK
            elif tile == TileType.GOAL:
                fill = COLOR_GOAL
            elif tile == TileType.VOID:
                fill = COLOR_VOID
            elif tile == TileType.SWITCH:
                fill, _ = _group_switch_color(board, r, c)
            elif tile == TileType.BRIDGE:
                group = board.bridges.get((r, c))
                is_on = group is not None and group in active_bridges
                fill = _group_bridge_color(board, r, c, is_on)
            elif tile == TileType.FRAGILE:
                fill = COLOR_FRAGILE
            elif tile == TileType.CRUMBLE:
                fill = COLOR_CRUMBLE
            elif is_elevator:
                fill = COLOR_ELEVATOR
            elif is_elevated:
                fill = COLOR_ELEVATED
            else:
                fill = COLOR_NORMAL

            if tile == TileType.CRUMBLE and (r, c) not in block_cells and _is_crumble_edge_style(crumble_style):
                parts.append(
                    f'<polygon points="{_pts(_crumble_tile_polygon_flat(x, y, cell, crumble_style))}" '
                    f'fill="{fill}" stroke="#444466" stroke-width="0.5"/>'
                )
            else:
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'fill="{fill}" stroke="#444466" stroke-width="0.5"/>'
                )

            # Goal diamond marker (drawn under block if block is on goal)
            if tile == TileType.GOAL and (r, c) not in block_cells:
                cx = x + cell / 2
                cy = y + cell / 2
                ds = cell * 0.15
                marker = [
                    (cx, cy - ds), (cx + ds, cy),
                    (cx, cy + ds), (cx - ds, cy),
                ]
                parts.append(
                    f'<polygon points="{_pts(marker)}" '
                    f'fill="{COLOR_GOAL_DIAMOND}" opacity="0.85"/>'
                )
            # Switch circle marker
            elif tile == TileType.SWITCH and (r, c) not in block_cells:
                _, marker_color = _group_switch_color(board, r, c)
                cx = x + cell / 2
                cy = y + cell / 2
                radius = cell * 0.15
                parts.append(
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
                    f'fill="{marker_color}" opacity="0.85"/>'
                )
            # Fragile X marker
            elif tile == TileType.FRAGILE and (r, c) not in block_cells:
                cx = x + cell / 2
                cy = y + cell / 2
                ds = cell * 0.18
                parts.append(
                    f'<line x1="{cx - ds:.1f}" y1="{cy - ds:.1f}" '
                    f'x2="{cx + ds:.1f}" y2="{cy + ds:.1f}" '
                    f'stroke="{COLOR_FRAGILE_MARKER}" stroke-width="2" opacity="0.8"/>'
                )
                parts.append(
                    f'<line x1="{cx + ds:.1f}" y1="{cy - ds:.1f}" '
                    f'x2="{cx - ds:.1f}" y2="{cy + ds:.1f}" '
                    f'stroke="{COLOR_FRAGILE_MARKER}" stroke-width="2" opacity="0.8"/>'
                )
            elif tile == TileType.CRUMBLE and (r, c) not in block_cells and crumble_style == "slab":
                parts.extend(_crumble_marker_flat(x, y, cell))
            # Elevator arrow marker
            elif is_elevator and (r, c) not in block_cells:
                cx = x + cell / 2
                cy = y + cell / 2
                fs = cell * 0.35
                parts.append(
                    f'<text x="{cx:.1f}" y="{cy + fs * 0.15:.1f}" '
                    f'font-family="monospace" font-size="{fs:.1f}" '
                    f'text-anchor="middle" dominant-baseline="central" '
                    f'fill="{COLOR_ELEVATOR_MARKER}" opacity="0.9">'
                    f'\u25b2\u25bc</text>'
                )
            # Elevated tile border (subtle visual cue)
            elif is_elevated and (r, c) not in block_cells:
                parts.append(
                    f'<rect x="{x + 2}" y="{y + 2}" width="{cell - 4}" height="{cell - 4}" '
                    f'fill="none" stroke="{_darken(COLOR_ELEVATED, 0.7)}" '
                    f'stroke-width="1.5" stroke-dasharray="3,2" rx="2"/>'
                )

    # -- legend bar (horizontal, below grid) --------------------------------
    if show_legend:
        legend_items = [
            (COLOR_NORMAL, "Normal"),
            (COLOR_VOID, "Void"),
            (COLOR_GOAL, "Goal"),
            (COLOR_BLOCK, "Block"),
        ]
        if any(board.grid[r][c] == TileType.CRUMBLE and (r, c) not in collapsed_crumble
               for r in range(rows) for c in range(cols)):
            legend_items.append((COLOR_CRUMBLE, "Crumble"))
        if any(board.grid[r][c] == TileType.ELEVATOR
               for r in range(rows) for c in range(cols)):
            legend_items.append((COLOR_ELEVATOR, "Elevator"))

        bar_y = total_h - legend_bar_h + 6
        parts.append(
            f'<rect x="0" y="{total_h - legend_bar_h}" width="{total_w}" '
            f'height="{legend_bar_h}" fill="#1e1e30"/>'
        )
        sw, gap_s, gap_i = 8, 3, 10
        iw = [sw + gap_s + len(lb) * 4.8 for _, lb in legend_items]
        tlw = sum(iw) + gap_i * (len(legend_items) - 1)
        ix = (total_w - tlw) / 2
        for j, (color, label) in enumerate(legend_items):
            parts.append(
                f'<rect x="{ix:.0f}" y="{bar_y}" width="{sw}" height="8" '
                f'fill="{color}" rx="1"/>'
            )
            parts.append(
                f'<text x="{ix + sw + gap_s:.0f}" y="{bar_y + 7}" '
                f'font-family="monospace" font-size="7" fill="{COLOR_TEXT}">{label}</text>'
            )
            ix += iw[j] + gap_i

    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}">'
    )
    return "\n".join([header] + parts + ["</svg>"])


def svg_to_png(svg: str, min_dim: int = 512) -> bytes:
    """Convert SVG string to PNG bytes via cairosvg."""
    import cairosvg

    vb = re.search(r'viewBox="\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)"', svg)
    if vb:
        wpx, hpx = float(vb.group(1)), float(vb.group(2))
        base_min = min(wpx, hpx) if wpx > 0 and hpx > 0 else 1.0
        scale = max(1.0, min_dim / base_min)
        out_w = int(round(wpx * scale))
        out_h = int(round(hpx * scale))
        return cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=out_w, output_height=out_h)
    return cairosvg.svg2png(bytestring=svg.encode("utf-8"))


def render_to_png_array(svg: str, min_dim: int = 512, autocrop: bool = True, crop_pad: int = 6):
    """Convert SVG to a numpy RGB array (for gym rgb_array mode)."""
    from PIL import Image, ImageChops
    import numpy as np

    png_bytes = svg_to_png(svg, min_dim)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if autocrop:
        bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            x0 = max(0, bbox[0] - crop_pad)
            y0 = max(0, bbox[1] - crop_pad)
            x1 = min(img.width, bbox[2] + crop_pad)
            y1 = min(img.height, bbox[3] + crop_pad)
            img = img.crop((x0, y0, x1, y1))
    return np.array(img)
