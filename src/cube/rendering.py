"""CUBE piece and assembled-state rendering."""
from __future__ import annotations
import numpy as np
import io
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Sequence
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from cube.env import detect_cube_conflicts
from cube._rendering.utils import compute_oriented_placements, Face


def _draw_prism(ax, origin: Tuple[float, float, float], size: float, height: float, color: str = 'lavender'):
    """
    Draw a rectangular prism (extruded square) on a 3D axes.
    origin = (x, y, z) bottom-left corner of the cell
    size   = cell size in x and y
    height = extrusion in z
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    x, y, z = origin
    # 8 vertices
    v = [
        (x,       y,       z),        # 0 bottom rectangle
        (x+size,  y,       z),        # 1
        (x+size,  y+size,  z),        # 2
        (x,       y+size,  z),        # 3
        (x,       y,       z+height), # 4 top rectangle
        (x+size,  y,       z+height), # 5
        (x+size,  y+size,  z+height), # 6
        (x,       y+size,  z+height)  # 7
    ]
    # six faces
    faces = [
        [v[0], v[1], v[2], v[3]],  # bottom (not essential)
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],  # side
        [v[1], v[2], v[6], v[5]],  # side
        [v[2], v[3], v[7], v[6]],  # side
        [v[3], v[0], v[4], v[7]],  # side
    ]
    coll = Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=0.5)
    ax.add_collection3d(coll)


def render_faces_3d(face_arrays: List[np.ndarray],
                                  *,
                                  titles: Optional[List[str]] = None,
                                  cell_size: float = 1.0,
                                  thickness: float = 1.0,
                                  figsize: Tuple[float, float] = (10.0, 7.0),
                                  dpi: int = 100,
                                  random_view: bool = True,
                                  seed: Optional[int] = None,
                                  save_path: Optional[str] = None,
                                  return_pil: bool = False):
    """
    Render piece faces in a 2×3 grid with per-piece colors.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import io
    import random as _rnd

    assert 1 <= len(face_arrays) <= 6, f"Expected 1-6 face arrays, got {len(face_arrays)}."
    face_arrays = [np.array(arr) if not isinstance(arr, np.ndarray) else arr for arr in face_arrays]
    if titles is None:
        titles = [chr(ord('A') + i) for i in range(len(face_arrays))]
    n = face_arrays[0].shape[0]
    if seed is not None:
        _rnd.seed(seed)
    # Fixed 2×3 grid using subplots
    nrows, ncols = 2, 3
    nfaces = len(face_arrays)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, subplot_kw={'projection': '3d'})
    axs = np.array(axs).flatten().tolist()
    for ax in axs[nfaces:]:
        ax.set_visible(False)

    _PIECE_COLORS = {
        "A": "#e53935", "B": "#1e88e5", "C": "#43a047",
        "D": "#8e24aa", "E": "#fb8c00", "F": "#fdd835",
    }
    render_margin = 0.2 * cell_size
    edge_label_offset = 0.8
    for ax, arr, face_no in zip(axs[:nfaces], face_arrays, titles):
        piece_color = _PIECE_COLORS.get(face_no, 'lavender')
        for row in range(n):
            for col in range(n):
                if int(arr[row][col]) == 1:
                    origin = (col * cell_size, row * cell_size, 0.0)
                    _draw_prism(ax, origin, size=cell_size, height=thickness, color=piece_color)

        # Keep the face view tight so the composite image does not waste canvas.
        ax.set_xlim(-render_margin, n * cell_size + render_margin)
        ax.set_ylim(-render_margin, n * cell_size + render_margin)
        ax.set_zlim(0, 2 * thickness)
        ax.set_box_aspect([6, 6, 1])

        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.xaxis.pane.set_visible(False); ax.yaxis.pane.set_visible(False); ax.zaxis.pane.set_visible(False)
        ax.grid(False)
        ax.xaxis.line.set_color((0,0,0,0)); ax.yaxis.line.set_color((0,0,0,0)); ax.zaxis.line.set_color((0,0,0,0))

        base_elev, base_azim = -90 - 45, -90 - 30
        elev_offset = _rnd.uniform(-15, 20) if random_view else 0.0
        azim_offset = _rnd.uniform(-20, 20) if random_view else 0.0
        ax.view_init(elev=base_elev + elev_offset, azim=base_azim + azim_offset)

        ax.text((n-1)/2, n + edge_label_offset, 0.5, "Edge 4", ha='center', va='center', fontsize=9)
        ax.text(n + edge_label_offset, (n-1)/2, 0.5, "Edge 3", ha='center', va='center', fontsize=9)
        ax.text((n-1)/2, -edge_label_offset, 0.5, "Edge 2", ha='center', va='center', fontsize=9)
        ax.text(-edge_label_offset, (n-1)/2, 0.5, "Edge 1", ha='center', va='center', fontsize=9)

        ax.set_title(f"Face {face_no}", y=0.95, fontsize=25, color='#1F1A1A')

    # Pack the 2x3 grid tightly before saving.
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.97, wspace=0.02, hspace=0.10)

    pil_img = None
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    if return_pil:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        buf.seek(0)
        with Image.open(buf) as img:
            pil_img = img.convert("RGB")
        buf.close()
    plt.close(fig)
    return pil_img


_GRID_COLOR = "#2D2D2D"


_GRID_INNER_COLOR = "#999999"


_LABEL_BG = "#F0F0F0"


_LABEL_BORDER = "#CCCCCC"


_LABEL_TEXT = "#2D2D2D"


_LABEL_PAD = 0.85


def _compute_disassembly_offsets(
    cube_to_letter: Dict[str, str],
    cube_to_arr: Dict[str, np.ndarray],
    thickness: float,
    disassemble_offset: float,
    flush_faces: Optional[set],
) -> Dict[str, float]:
    offs = {"Front": 0.0, "Back": 0.0, "Left": 0.0, "Right": 0.0, "Top": 0.0, "Bottom": 0.0}
    if not cube_to_arr:
        return offs

    sep = max(disassemble_offset, thickness + 1e-3)
    locked = {str(face).capitalize() for face in (flush_faces or set())}
    oriented_faces = {
        cube_face.lower(): (cube_to_letter[cube_face], cube_to_arr[cube_face].astype(np.int32))
        for cube_face in cube_to_arr
    }
    for conflict in detect_cube_conflicts(oriented_faces):
        for face_name in conflict["faces"]:
            title_face = str(face_name).capitalize()
            if title_face in offs and title_face not in locked:
                offs[title_face] = sep
    return offs


def plot_cube_net(
    faces: List[Face],
    face_order: Sequence[str] = ("A", "B", "C", "D", "E", "F"),
    colors: Dict[str, str] = None,
    title: str = None,
    show_labels: bool = True,
    save_path: str = None,
    show: bool = False,
    solution: Dict[str, Tuple[str, int, int]] = None,
    show_face_names: bool = False,
) -> np.ndarray:
    faces_np = [np.asarray(f, dtype=np.uint8) for f in faces]
    n = faces_np[0].shape[0]
    if any(f.shape != (n, n) for f in faces_np):
        raise ValueError("All faces must be NxN with the same N.")
    if colors is None:
        colors = {
            "A": "#e53935",
            "B": "#1e88e5",
            "C": "#43a047",
            "D": "#8e24aa",
            "E": "#fb8c00",
            "F": "#fdd835",
        }
    H, W = 3 * n, 4 * n
    rgba = np.ones((H, W, 4), dtype=float)
    layout_by_cube_face = {
        "Top": (0 * n, 1 * n),
        "Front": (1 * n, 1 * n),
        "Bottom": (2 * n, 1 * n),
        "Left": (1 * n, 0 * n),
        "Right": (1 * n, 2 * n),
        "Back": (1 * n, 3 * n),
    }
    computed = compute_oriented_placements(faces_np, face_order, solution)
    placements: List[Tuple[str, np.ndarray, Tuple[int, int]]] = []
    for letter, arr, (cube_face, (t, l)) in computed:
        placements.append((letter, arr, (t, l)))
    for letter, face_arr, (top, left) in placements:
        mask = face_arr == 1
        color_rgba = mcolors.to_rgba(colors[letter])
        block = rgba[top : top + n, left : left + n]
        block[mask] = color_rgba

    margin_left = _LABEL_PAD
    margin_right = _LABEL_PAD
    margin_top = _LABEL_PAD
    margin_bottom = _LABEL_PAD
    fig_w = 1.1 * (W + margin_left + margin_right)
    fig_h = 0.9 * (H + margin_top + margin_bottom)
    fig, ax = plt.subplots(figsize=(fig_w / 2.54, fig_h / 2.54), dpi=150)
    ax.imshow(
        rgba,
        extent=(0, W, H, 0),
        interpolation="none",
        origin="upper",
        zorder=1,
    )

    def draw_grid(t: int, l: int, N: int) -> None:
        for r in range(N + 1):
            ax.hlines(t + r, l, l + N, colors="black", linewidth=0.8)
        for c in range(N + 1):
            ax.vlines(l + c, t, t + N, colors="black", linewidth=0.8)
        ax.plot([l, l + N, l + N, l, l], [t, t, t + N, t + N, t], color="black", linewidth=1.6)

    for (t, l) in layout_by_cube_face.values():
        draw_grid(t, l, n)

    if show_labels:
        for letter, _, (t, l) in placements:
            ax.text(l + n / 2, t + n / 2, letter, ha="center", va="center", fontsize=10, weight="bold")

    if show_face_names:
        badge_h = 0.55
        badge_w = n * 0.6
        label_positions = {
            "Top": (1.5 * n, -margin_top / 2),
            "Front": (1.5 * n, n - 0.35),
            "Bottom": (1.5 * n, H + margin_bottom / 2),
            "Left": (n / 2, n - 0.35),
            "Right": (2.5 * n, n - 0.35),
            "Back": (W - n / 2, n - 0.35),
        }
        for face_name, (t, l) in layout_by_cube_face.items():
            cx, cy = label_positions[face_name]
            badge = FancyBboxPatch(
                (cx - badge_w / 2, cy - badge_h / 2), badge_w, badge_h,
                boxstyle="round,pad=0.12",
                facecolor=_LABEL_BG, edgecolor=_LABEL_BORDER,
                linewidth=0.7, alpha=0.92, zorder=5,
            )
            ax.add_patch(badge)
            ax.text(cx, cy, face_name, ha="center", va="center",
                    fontsize=6.5, weight="semibold", color=_LABEL_TEXT,
                    fontfamily="sans-serif", zorder=6)

    handles = [mpatches.Patch(color=colors[k], label=f"Face {k}") for k in ("A", "B", "C", "D", "E", "F")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=False)

    if title:
        ax.set_title(title)
    ax.set_xlim(-margin_left, W + margin_right)
    ax.set_ylim(H + margin_bottom, -margin_top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    # Extract RGB array via tight-bbox buffer to avoid legend clipping
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    rgb_array = np.asarray(Image.open(buf).convert("RGB"))
    buf.close()
    return rgb_array


def plot_cube_3d(
    faces: List[Face],
    face_order: Sequence[str] = ("A", "B", "C", "D", "E", "F"),
    colors: Dict[str, str] = None,
    save_path: str = None,
    show: bool = False,
    solution: Dict[str, Tuple[str, int, int]] = None,
    draw_seams: bool = True,
    tile_depth: float = 1.0,
    disassemble_on_mismatch: bool = True,
    disassemble_offset: float = 0.30,
    flush_faces: Optional[set] = None,
) -> np.ndarray:
    """Render a 3D cube assembly.

    Args:
        flush_faces: set of cube face names (e.g. {"Top", "Front"}) that must
            stay flush (no offset) even if they have conflicts. Used to keep
            pre-placed/locked pieces assembled. Any face NOT in this set that
            has a seam conflict will be offset (disassembled).
            If None, all conflicting faces get offset.
    """
    faces_np = [np.asarray(f, dtype=np.uint8) for f in faces]
    n = faces_np[0].shape[0]
    if any(f.shape != (n, n) for f in faces_np):
        raise ValueError("All faces must be NxN with the same N.")
    if colors is None:
        colors = {
            "A": "#e53935", "B": "#1e88e5", "C": "#43a047",
            "D": "#8e24aa", "E": "#fb8c00", "F": "#fdd835",
        }
    placements = compute_oriented_placements(faces_np, face_order, solution)
    cube_to_letter: Dict[str, str] = {}
    cube_to_arr: Dict[str, np.ndarray] = {}
    for letter, arr, (cube_face, _) in placements:
        cube_to_letter[cube_face] = letter
        cube_to_arr[cube_face] = arr
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    dx = 1.0 / n
    thickness = max(1e-6, min(dx * float(tile_depth), dx))
    # Disassembly: check ALL shared edges of the cube.
    # If disassemble_only is set, only that face can be offset.
    # Edge adjacency: each pair of adjacent faces shares one edge.
    # Format: (faceA, edgeA_fn, faceB, edgeB_fn) — if both have 1s at shared edge, they conflict.
    offs = {"Front": 0.0, "Back": 0.0, "Left": 0.0, "Right": 0.0, "Top": 0.0, "Bottom": 0.0}
    if disassemble_on_mismatch:
        offs = _compute_disassembly_offsets(
            cube_to_letter=cube_to_letter,
            cube_to_arr=cube_to_arr,
            thickness=thickness,
            disassemble_offset=disassemble_offset,
            flush_faces=flush_faces,
        )
        nn = n
        sep = max(disassemble_offset, thickness + 1e-3)
        # All 12 edges of a cube: (faceA, sliceA, faceB, sliceB)
        _EDGES = [
            # Front neighbors
            ("Front", lambda a: a[0, :],      "Top",    lambda a: a[nn-1, :]),
            ("Front", lambda a: a[nn-1, :],   "Bottom", lambda a: a[0, :]),
            ("Front", lambda a: a[:, 0],      "Left",   lambda a: a[:, nn-1]),
            ("Front", lambda a: a[:, nn-1],   "Right",  lambda a: a[:, 0]),
            # Back neighbors
            ("Back",  lambda a: a[0, :],      "Top",    lambda a: a[0, :]),
            ("Back",  lambda a: a[nn-1, :],   "Bottom", lambda a: a[nn-1, :]),
            ("Back",  lambda a: a[:, nn-1],   "Left",   lambda a: a[:, 0]),
            ("Back",  lambda a: a[:, 0],      "Right",  lambda a: a[:, nn-1]),
            # Top-Left, Top-Right, Bottom-Left, Bottom-Right
            ("Top",   lambda a: a[:, 0],      "Left",   lambda a: a[0, :]),
            ("Top",   lambda a: a[:, nn-1],   "Right",  lambda a: a[0, :]),
            ("Bottom", lambda a: a[:, 0],     "Left",   lambda a: a[nn-1, :]),
            ("Bottom", lambda a: a[:, nn-1],  "Right",  lambda a: a[nn-1, :]),
        ]
        _flush = set(offs)
        for fA, sliceA, fB, sliceB in _EDGES:
            if fA not in cube_to_arr or fB not in cube_to_arr:
                continue
            edgeA = (sliceA(cube_to_arr[fA]) == 1)
            edgeB = (sliceB(cube_to_arr[fB]) == 1)
            if not np.any(edgeA & edgeB):
                continue
            # Conflict found — offset any non-flush face
            if fA not in _flush:
                offs[fA] = sep
            if fB not in _flush:
                offs[fB] = sep
    
    def add_quad(verts, face_color, edge_color="black", lw=0.0):
        poly = Poly3DCollection([verts])
        poly.set_facecolor(face_color)
        poly.set_edgecolor(edge_color)
        poly.set_linewidth(lw)
        ax.add_collection3d(poly)
    
    def add_line(seg, color="black", lw=1.2):
        lc = Line3DCollection([seg], colors=color, linewidths=lw)
        ax.add_collection3d(lc)
    
    def darker(color_hex: str, factor: float = 0.82):
        r, g, b = mcolors.to_rgb(color_hex)
        return (r * factor, g * factor, b * factor, 1.0)
    
    def x_span(c):
        x0 = -0.5 + c * dx
        return x0, x0 + dx
    
    def z_span(r):
        z0 = 0.5 - r * dx
        return z0, z0 - dx
    
    def y_span(c, reverse=False):
        if not reverse:
            y0 = -0.5 + c * dx
            return y0, y0 + dx
        else:
            y0 = 0.5 - c * dx
            return y0, y0 - dx
    for cube_face, arr in cube_to_arr.items():
        letter = cube_to_letter[cube_face]
        face_color = colors[letter]
        side_color = darker(face_color, 0.72)
        inner_color = darker(face_color, 0.86)
        def add_tile_box(cube_face_local: str, r: int, c: int):
            if cube_face_local == "Front":
                x0, x1 = x_span(c); z0, z1 = z_span(r); y_out = 0.5 + offs["Front"]; y_in = y_out - thickness
                add_quad([(x0, y_out, z0), (x1, y_out, z0), (x1, y_out, z1), (x0, y_out, z1)], face_color)
                add_quad([(x0, y_in, z0), (x1, y_in, z0), (x1, y_in, z1), (x0, y_in, z1)], inner_color)
                add_quad([(x0, y_in, z0), (x0, y_out, z0), (x0, y_out, z1), (x0, y_in, z1)], side_color)
                add_quad([(x1, y_in, z0), (x1, y_out, z0), (x1, y_out, z1), (x1, y_in, z1)], side_color)
                add_quad([(x0, y_in, z0), (x1, y_in, z0), (x1, y_out, z0), (x0, y_out, z0)], side_color)
                add_quad([(x0, y_in, z1), (x1, y_in, z1), (x1, y_out, z1), (x0, y_out, z1)], side_color)
            elif cube_face_local == "Back":
                x0, x1 = x_span(n - 1 - c); z0, z1 = z_span(r); y_out = -0.5 - offs["Back"]; y_in = y_out + thickness
                add_quad([(x1, y_out, z0), (x0, y_out, z0), (x0, y_out, z1), (x1, y_out, z1)], face_color)
                add_quad([(x1, y_in, z0), (x0, y_in, z0), (x0, y_in, z1), (x1, y_in, z1)], inner_color)
                add_quad([(x0, y_in, z0), (x0, y_out, z0), (x0, y_out, z1), (x0, y_in, z1)], side_color)
                add_quad([(x1, y_in, z0), (x1, y_out, z0), (x1, y_out, z1), (x1, y_in, z1)], side_color)
                add_quad([(x0, y_in, z0), (x1, y_in, z0), (x1, y_out, z0), (x0, y_out, z0)], side_color)
                add_quad([(x0, y_in, z1), (x1, y_in, z1), (x1, y_out, z1), (x0, y_out, z1)], side_color)
            elif cube_face_local == "Right":
                y0, y1 = y_span(c, reverse=True); z0, z1 = z_span(r); x_out = 0.5 + offs["Right"]; x_in = x_out - thickness
                add_quad([(x_out, y0, z0), (x_out, y1, z0), (x_out, y1, z1), (x_out, y0, z1)], face_color)
                add_quad([(x_in, y0, z0), (x_in, y1, z0), (x_in, y1, z1), (x_in, y0, z1)], inner_color)
                add_quad([(x_in, y0, z0), (x_out, y0, z0), (x_out, y1, z0), (x_in, y1, z0)], side_color)
                add_quad([(x_in, y0, z1), (x_out, y0, z1), (x_out, y1, z1), (x_in, y1, z1)], side_color)
                add_quad([(x_in, y0, z0), (x_out, y0, z0), (x_out, y0, z1), (x_in, y0, z1)], side_color)
                add_quad([(x_in, y1, z0), (x_out, y1, z0), (x_out, y1, z1), (x_in, y1, z1)], side_color)
            elif cube_face_local == "Left":
                y0, y1 = y_span(c, reverse=False); z0, z1 = z_span(r); x_out = -0.5 - offs["Left"]; x_in = x_out + thickness
                add_quad([(x_out, y0, z0), (x_out, y1, z0), (x_out, y1, z1), (x_out, y0, z1)], face_color)
                add_quad([(x_in, y0, z0), (x_in, y1, z0), (x_in, y1, z1), (x_in, y0, z1)], inner_color)
                add_quad([(x_in, y0, z0), (x_out, y0, z0), (x_out, y1, z0), (x_in, y1, z0)], side_color)
                add_quad([(x_in, y0, z1), (x_out, y0, z1), (x_out, y1, z1), (x_in, y1, z1)], side_color)
                add_quad([(x_in, y0, z0), (x_out, y0, z0), (x_out, y0, z1), (x_in, y0, z1)], side_color)
                add_quad([(x_in, y1, z0), (x_out, y1, z0), (x_out, y1, z1), (x_in, y1, z1)], side_color)
            elif cube_face_local == "Top":
                x0, x1 = x_span(c); y0, y1 = y_span(r, reverse=False); z_out = 0.5 + offs["Top"]; z_in = z_out - thickness
                add_quad([(x0, y0, z_out), (x1, y0, z_out), (x1, y1, z_out), (x0, y1, z_out)], face_color)
                add_quad([(x0, y0, z_in), (x1, y0, z_in), (x1, y1, z_in), (x0, y1, z_in)], inner_color)
                add_quad([(x0, y0, z_in), (x1, y0, z_in), (x1, y0, z_out), (x0, y0, z_out)], side_color)
                add_quad([(x0, y1, z_in), (x1, y1, z_in), (x1, y1, z_out), (x0, y1, z_out)], side_color)
                add_quad([(x0, y0, z_in), (x0, y1, z_in), (x0, y1, z_out), (x0, y0, z_out)], side_color)
                add_quad([(x1, y0, z_in), (x1, y1, z_in), (x1, y1, z_out), (x1, y0, z_out)], side_color)
            elif cube_face_local == "Bottom":
                x0, x1 = x_span(c); y0, y1 = y_span(r, reverse=True); z_out = -0.5 - offs["Bottom"]; z_in = z_out + thickness
                add_quad([(x0, y0, z_out), (x1, y0, z_out), (x1, y1, z_out), (x0, y1, z_out)], face_color)
                add_quad([(x0, y0, z_in), (x1, y0, z_in), (x1, y1, z_in), (x0, y1, z_in)], inner_color)
                add_quad([(x0, y0, z_in), (x1, y0, z_in), (x1, y0, z_out), (x0, y0, z_out)], side_color)
                add_quad([(x0, y1, z_in), (x1, y1, z_in), (x1, y1, z_out), (x0, y1, z_out)], side_color)
                add_quad([(x0, y0, z_in), (x0, y1, z_in), (x0, y1, z_out), (x0, y0, z_out)], side_color)
                add_quad([(x1, y0, z_in), (x1, y1, z_in), (x1, y1, z_out), (x1, y0, z_out)], side_color)
            else:
                return
        for r in range(n):
            for c in range(n):
                if arr[r, c] != 1:
                    continue
                add_tile_box(cube_face, r, c)
    if draw_seams and "Front" in cube_to_arr:
        F = cube_to_arr["Front"]
        if "Top" in cube_to_arr:
            T = cube_to_arr["Top"]
            rF, rT = 0, n - 1
            for c in range(n):
                if F[rF, c] == 1 and T[rT, c] == 1:
                    x0 = -0.5 + c * dx
                    x1 = x0 + dx
                    y, z = 0.5, 0.5
                    add_line([(x0, y, z), (x1, y, z)], color="black", lw=1.2)
        if "Bottom" in cube_to_arr:
            B = cube_to_arr["Bottom"]
            rF, rB = n - 1, 0
            for c in range(n):
                if F[rF, c] == 1 and B[rB, c] == 1:
                    x0 = -0.5 + c * dx
                    x1 = x0 + dx
                    y, z = 0.5, -0.5
                    add_line([(x0, y, z), (x1, y, z)], color="black", lw=1.2)
        if "Left" in cube_to_arr:
            L = cube_to_arr["Left"]
            cF, cL = 0, n - 1
            for r in range(n):
                if F[r, cF] == 1 and L[r, cL] == 1:
                    z0 = 0.5 - r * dx
                    z1 = z0 - dx
                    x, y = -0.5, 0.5
                    add_line([(x, y, z0), (x, y, z1)], color="black", lw=1.2)
        if "Right" in cube_to_arr:
            R = cube_to_arr["Right"]
            cF, cR = n - 1, 0
            for r in range(n):
                if F[r, cF] == 1 and R[r, cR] == 1:
                    z0 = 0.5 - r * dx
                    z1 = z0 - dx
                    x, y = 0.5, 0.5
                    add_line([(x, y, z0), (x, y, z1)], color="black", lw=1.2)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    # Extract RGB array from figure
    fig.canvas.draw()
    rgb_array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return rgb_array


def plot_cube_3d_views(
    faces: List[Face],
    face_order: Sequence[str] = ("A", "B", "C", "D", "E", "F"),
    colors: Dict[str, str] = None,
    save_path: str = None,
    solution: Dict[str, Tuple[str, int, int]] = None,
    draw_seams: bool = True,
    preset: str = "iso6",
    tile_depth: float = 1.0,
    disassemble_on_mismatch: bool = True,
    disassemble_offset: float = 0.30,
    flush_faces: Optional[set] = None,
) -> np.ndarray:
    """Render multi-view 3D cube assembly.

    Args:
        flush_faces: set of cube face names that stay flush (no offset).
            Any conflicting face NOT in this set gets offset.
    """
    faces_np = [np.asarray(f, dtype=np.uint8) for f in faces]
    n = faces_np[0].shape[0]
    if any(f.shape != (n, n) for f in faces_np):
        raise ValueError("All faces must be NxN with the same N.")
    if colors is None:
        colors = {
            "A": "#e53935", "B": "#1e88e5", "C": "#43a047",
            "D": "#8e24aa", "E": "#fb8c00", "F": "#fdd835",
        }
    placements = compute_oriented_placements(faces_np, face_order, solution)
    cube_to_letter: Dict[str, str] = {}
    cube_to_arr: Dict[str, np.ndarray] = {}
    for letter, arr, (cube_face, _) in placements:
        cube_to_letter[cube_face] = letter
        cube_to_arr[cube_face] = arr
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    if preset == "iso6":
        views = [
            ("iso-NE", 20, 35),
            ("iso-NW", 20, 145),
            ("iso-SE", 20, -35),
            ("iso-SW", 20, -145),
            ("top", 75, 35),
            ("bottom", -75, 35),
        ]
        nrows, ncols = 2, 3
    else:
        raise ValueError("Unknown preset for multi-views.")
    fig = plt.figure(figsize=(9, 6), dpi=100)
    axes = [fig.add_subplot(nrows, ncols, i + 1, projection="3d") for i in range(len(views))]
    dx = 1.0 / n
    thickness = max(1e-6, min(dx * float(tile_depth), dx))
    # Compute disassembly offsets once (same for all subplots)
    # Check ALL 12 edges of the cube, not just Front's neighbors.
    offs = {"Front": 0.0, "Back": 0.0, "Left": 0.0, "Right": 0.0, "Top": 0.0, "Bottom": 0.0}
    if disassemble_on_mismatch:
        offs = _compute_disassembly_offsets(
            cube_to_letter=cube_to_letter,
            cube_to_arr=cube_to_arr,
            thickness=thickness,
            disassemble_offset=disassemble_offset,
            flush_faces=flush_faces,
        )
        nn = n
        sep = max(disassemble_offset, thickness + 1e-3)
        _EDGES = [
            ("Front", lambda a: a[0, :],      "Top",    lambda a: a[nn-1, :]),
            ("Front", lambda a: a[nn-1, :],   "Bottom", lambda a: a[0, :]),
            ("Front", lambda a: a[:, 0],      "Left",   lambda a: a[:, nn-1]),
            ("Front", lambda a: a[:, nn-1],   "Right",  lambda a: a[:, 0]),
            ("Back",  lambda a: a[0, :],      "Top",    lambda a: a[0, :]),
            ("Back",  lambda a: a[nn-1, :],   "Bottom", lambda a: a[nn-1, :]),
            ("Back",  lambda a: a[:, nn-1],   "Left",   lambda a: a[:, 0]),
            ("Back",  lambda a: a[:, 0],      "Right",  lambda a: a[:, nn-1]),
            ("Top",   lambda a: a[:, 0],      "Left",   lambda a: a[0, :]),
            ("Top",   lambda a: a[:, nn-1],   "Right",  lambda a: a[0, :]),
            ("Bottom", lambda a: a[:, 0],     "Left",   lambda a: a[nn-1, :]),
            ("Bottom", lambda a: a[:, nn-1],  "Right",  lambda a: a[nn-1, :]),
        ]
        _flush = set(offs)
        for fA, sliceA, fB, sliceB in _EDGES:
            if fA not in cube_to_arr or fB not in cube_to_arr:
                continue
            edgeA = (sliceA(cube_to_arr[fA]) == 1)
            edgeB = (sliceB(cube_to_arr[fB]) == 1)
            if not np.any(edgeA & edgeB):
                continue
            if fA not in _flush:
                offs[fA] = sep
            if fB not in _flush:
                offs[fB] = sep
    
    def add_quad(ax, verts, face_color, edge_color="black", lw=0.0):
        poly = Poly3DCollection([verts])
        poly.set_facecolor(face_color)
        poly.set_edgecolor(edge_color)
        poly.set_linewidth(lw)
        ax.add_collection3d(poly)
    def add_line(ax, seg, color="black", lw=1.2):
        lc = Line3DCollection([seg], colors=color, linewidths=lw)
        ax.add_collection3d(lc)
    def x_span(c):
        x0 = -0.5 + c * dx
        return x0, x0 + dx
    def z_span(r):
        z0 = 0.5 - r * dx
        return z0, z0 - dx
    def y_span(c, reverse=False):
        if not reverse:
            y0 = -0.5 + c * dx
            return y0, y0 + dx
        else:
            y0 = 0.5 - c * dx
            return y0, y0 - dx
    def darker(color_hex: str, factor: float = 0.82):
        r, g, b = mcolors.to_rgb(color_hex)
        return (r * factor, g * factor, b * factor, 1.0)
    for ax, (name, elev, azim) in zip(axes, views):
        for cube_face, arr in cube_to_arr.items():
            letter = cube_to_letter[cube_face]
            face_color = colors[letter]
            side_color = darker(face_color, 0.72)
            inner_color = darker(face_color, 0.86)
            for r in range(n):
                for c in range(n):
                    if arr[r, c] != 1:
                        continue
                    if cube_face == "Front":
                        x0, x1 = x_span(c); z0, z1 = z_span(r); y_out = 0.5 + offs["Front"]; y_in = y_out - thickness
                        add_quad(ax, [(x0, y_out, z0), (x1, y_out, z0), (x1, y_out, z1), (x0, y_out, z1)], face_color)
                        add_quad(ax, [(x0, y_in, z0), (x1, y_in, z0), (x1, y_in, z1), (x0, y_in, z1)], inner_color)
                        add_quad(ax, [(x0, y_in, z0), (x0, y_out, z0), (x0, y_out, z1), (x0, y_in, z1)], side_color)
                        add_quad(ax, [(x1, y_in, z0), (x1, y_out, z0), (x1, y_out, z1), (x1, y_in, z1)], side_color)
                        add_quad(ax, [(x0, y_in, z0), (x1, y_in, z0), (x1, y_out, z0), (x0, y_out, z0)], side_color)
                        add_quad(ax, [(x0, y_in, z1), (x1, y_in, z1), (x1, y_out, z1), (x0, y_out, z1)], side_color)
                    elif cube_face == "Back":
                        x0, x1 = x_span(n - 1 - c); z0, z1 = z_span(r); y_out = -0.5 - offs["Back"]; y_in = y_out + thickness
                        add_quad(ax, [(x1, y_out, z0), (x0, y_out, z0), (x0, y_out, z1), (x1, y_out, z1)], face_color)
                        add_quad(ax, [(x1, y_in, z0), (x0, y_in, z0), (x0, y_in, z1), (x1, y_in, z1)], inner_color)
                        add_quad(ax, [(x0, y_in, z0), (x0, y_out, z0), (x0, y_out, z1), (x0, y_in, z1)], side_color)
                        add_quad(ax, [(x1, y_in, z0), (x1, y_out, z0), (x1, y_out, z1), (x1, y_in, z1)], side_color)
                        add_quad(ax, [(x0, y_in, z0), (x1, y_in, z0), (x1, y_out, z0), (x0, y_out, z0)], side_color)
                        add_quad(ax, [(x0, y_in, z1), (x1, y_in, z1), (x1, y_out, z1), (x0, y_out, z1)], side_color)
                    elif cube_face == "Right":
                        y0, y1 = y_span(c, reverse=True); z0, z1 = z_span(r); x_out = 0.5 + offs["Right"]; x_in = x_out - thickness
                        add_quad(ax, [(x_out, y0, z0), (x_out, y1, z0), (x_out, y1, z1), (x_out, y0, z1)], face_color)
                        add_quad(ax, [(x_in, y0, z0), (x_in, y1, z0), (x_in, y1, z1), (x_in, y0, z1)], inner_color)
                        add_quad(ax, [(x_in, y0, z0), (x_out, y0, z0), (x_out, y1, z0), (x_in, y1, z0)], side_color)
                        add_quad(ax, [(x_in, y0, z1), (x_out, y0, z1), (x_out, y1, z1), (x_in, y1, z1)], side_color)
                        add_quad(ax, [(x_in, y0, z0), (x_out, y0, z0), (x_out, y0, z1), (x_in, y0, z1)], side_color)
                        add_quad(ax, [(x_in, y1, z0), (x_out, y1, z0), (x_out, y1, z1), (x_in, y1, z1)], side_color)
                    elif cube_face == "Left":
                        y0, y1 = y_span(c, reverse=False); z0, z1 = z_span(r); x_out = -0.5 - offs["Left"]; x_in = x_out + thickness
                        add_quad(ax, [(x_out, y0, z0), (x_out, y1, z0), (x_out, y1, z1), (x_out, y0, z1)], face_color)
                        add_quad(ax, [(x_in, y0, z0), (x_in, y1, z0), (x_in, y1, z1), (x_in, y0, z1)], inner_color)
                        add_quad(ax, [(x_in, y0, z0), (x_out, y0, z0), (x_out, y1, z0), (x_in, y1, z0)], side_color)
                        add_quad(ax, [(x_in, y0, z1), (x_out, y0, z1), (x_out, y1, z1), (x_in, y1, z1)], side_color)
                        add_quad(ax, [(x_in, y0, z0), (x_out, y0, z0), (x_out, y0, z1), (x_in, y0, z1)], side_color)
                        add_quad(ax, [(x_in, y1, z0), (x_out, y1, z0), (x_out, y1, z1), (x_in, y1, z1)], side_color)
                    elif cube_face == "Top":
                        x0, x1 = x_span(c); y0, y1 = y_span(r, reverse=False); z_out = 0.5 + offs["Top"]; z_in = z_out - thickness
                        add_quad(ax, [(x0, y0, z_out), (x1, y0, z_out), (x1, y1, z_out), (x0, y1, z_out)], face_color)
                        add_quad(ax, [(x0, y0, z_in), (x1, y0, z_in), (x1, y1, z_in), (x0, y1, z_in)], inner_color)
                        add_quad(ax, [(x0, y0, z_in), (x1, y0, z_in), (x1, y0, z_out), (x0, y0, z_out)], side_color)
                        add_quad(ax, [(x0, y1, z_in), (x1, y1, z_in), (x1, y1, z_out), (x0, y1, z_out)], side_color)
                        add_quad(ax, [(x0, y0, z_in), (x0, y1, z_in), (x0, y1, z_out), (x0, y0, z_out)], side_color)
                        add_quad(ax, [(x1, y0, z_in), (x1, y1, z_in), (x1, y1, z_out), (x1, y0, z_out)], side_color)
                    elif cube_face == "Bottom":
                        x0, x1 = x_span(c); y0, y1 = y_span(r, reverse=True); z_out = -0.5 - offs["Bottom"]; z_in = z_out + thickness
                        add_quad(ax, [(x0, y0, z_out), (x1, y0, z_out), (x1, y1, z_out), (x0, y1, z_out)], face_color)
                        add_quad(ax, [(x0, y0, z_in), (x1, y0, z_in), (x1, y1, z_in), (x0, y1, z_in)], inner_color)
                        add_quad(ax, [(x0, y0, z_in), (x1, y0, z_in), (x1, y0, z_out), (x0, y0, z_out)], side_color)
                        add_quad(ax, [(x0, y1, z_in), (x1, y1, z_in), (x1, y1, z_out), (x0, y1, z_out)], side_color)
                        add_quad(ax, [(x0, y0, z_in), (x0, y1, z_in), (x0, y1, z_out), (x0, y0, z_out)], side_color)
                        add_quad(ax, [(x1, y0, z_in), (x1, y1, z_in), (x1, y1, z_out), (x1, y0, z_out)], side_color)
                    else:
                        continue
        if draw_seams and "Front" in cube_to_arr:
            F = cube_to_arr["Front"]
            if "Top" in cube_to_arr:
                T = cube_to_arr["Top"]
                rF, rT = 0, n - 1
                for c in range(n):
                    if F[rF, c] == 1 and T[rT, c] == 1:
                        x0 = -0.5 + c * dx
                        x1 = x0 + dx
                        y, z = 0.5, 0.5
                        add_line(ax, [(x0, y, z), (x1, y, z)], color="black", lw=1.1)
            if "Bottom" in cube_to_arr:
                B = cube_to_arr["Bottom"]
                rF, rB = n - 1, 0
                for c in range(n):
                    if F[rF, c] == 1 and B[rB, c] == 1:
                        x0 = -0.5 + c * dx
                        x1 = x0 + dx
                        y, z = 0.5, -0.5
                        add_line(ax, [(x0, y, z), (x1, y, z)], color="black", lw=1.1)
            if "Left" in cube_to_arr:
                L = cube_to_arr["Left"]
                cF, cL = 0, n - 1
                for r in range(n):
                    if F[r, cF] == 1 and L[r, cL] == 1:
                        z0 = 0.5 - r * dx
                        z1 = z0 - dx
                        x, y = -0.5, 0.5
                        add_line(ax, [(x, y, z0), (x, y, z1)], color="black", lw=1.1)
            if "Right" in cube_to_arr:
                R = cube_to_arr["Right"]
                cF, cR = n - 1, 0
                for r in range(n):
                    if F[r, cF] == 1 and R[r, cR] == 1:
                        z0 = 0.5 - r * dx
                        z1 = z0 - dx
                        x, y = 0.5, 0.5
                        add_line(ax, [(x, y, z0), (x, y, z1)], color="black", lw=1.1)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(-0.6, 0.6)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        ax.set_axis_off()
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name, fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    # Extract RGB array from figure
    fig.canvas.draw()
    rgb_array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return rgb_array


def plot_cube_assembled_voxels(
    faces: List[Face],
    face_order: Sequence[str] = ("A", "B", "C", "D", "E", "F"),
    colors: Dict[str, str] = None,
    save_path: str = None,
    show: bool = False,
    solution: Dict[str, Tuple[str, int, int]] = None,
) -> None:
    """
    Render the cube as an assembled set of mini-cubes (voxels) forming a 1-voxel thick shell.
    Each face's 1-cells become voxels on the corresponding outer plane. Edges/corners unify
    to a single voxel to ensure distances fit and there is no overlap.
    """
    faces_np = [np.asarray(f, dtype=np.uint8) for f in faces]
    n = faces_np[0].shape[0]
    if any(f.shape != (n, n) for f in faces_np):
        raise ValueError("All faces must be NxN with the same N.")
    if colors is None:
        colors = {
            "A": "#e53935", "B": "#1e88e5", "C": "#43a047",
            "D": "#8e24aa", "E": "#fb8c00", "F": "#fdd835",
        }
    placements = compute_oriented_placements(faces_np, face_order, solution)
    cube_to_letter: Dict[str, str] = {}
    cube_to_arr: Dict[str, np.ndarray] = {}
    for letter, arr, (cube_face, _) in placements:
        cube_to_letter[cube_face] = letter
        cube_to_arr[cube_face] = arr
    filled = np.zeros((n, n, n), dtype=bool)
    facecolors = np.zeros((n, n, n, 4), dtype=float)
    def set_voxel(x: int, y: int, z: int, rgba):
        if 0 <= x < n and 0 <= y < n and 0 <= z < n:
            if not filled[x, y, z]:
                filled[x, y, z] = True
                facecolors[x, y, z] = rgba
    if "Front" in cube_to_arr:
        arr = cube_to_arr["Front"]; rgba = mcolors.to_rgba(colors[cube_to_letter["Front"]]); y = n - 1
        for r in range(n):
            for c in range(n):
                if arr[r, c] == 1:
                    set_voxel(c, y, (n - 1) - r, rgba)
    if "Back" in cube_to_arr:
        arr = cube_to_arr["Back"]; rgba = mcolors.to_rgba(colors[cube_to_letter["Back"]]); y = 0
        for r in range(n):
            for c in range(n):
                if arr[r, c] == 1:
                    set_voxel((n - 1) - c, y, (n - 1) - r, rgba)
    if "Right" in cube_to_arr:
        arr = cube_to_arr["Right"]; rgba = mcolors.to_rgba(colors[cube_to_letter["Right"]]); x = n - 1
        for r in range(n):
            for c in range(n):
                if arr[r, c] == 1:
                    set_voxel(x, (n - 1) - c, (n - 1) - r, rgba)
    if "Left" in cube_to_arr:
        arr = cube_to_arr["Left"]; rgba = mcolors.to_rgba(colors[cube_to_letter["Left"]]); x = 0
        for r in range(n):
            for c in range(n):
                if arr[r, c] == 1:
                    set_voxel(x, c, (n - 1) - r, rgba)
    if "Top" in cube_to_arr:
        arr = cube_to_arr["Top"]; rgba = mcolors.to_rgba(colors[cube_to_letter["Top"]]); z = n - 1
        for r in range(n):
            for c in range(n):
                if arr[r, c] == 1:
                    set_voxel(c, r, z, rgba)
    if "Bottom" in cube_to_arr:
        arr = cube_to_arr["Bottom"]; rgba = mcolors.to_rgba(colors[cube_to_letter["Bottom"]]); z = 0
        for r in range(n):
            for c in range(n):
                if arr[r, c] == 1:
                    set_voxel(c, (n - 1) - r, z, rgba)
    fig = plt.figure(figsize=(7.2, 7.2), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(filled, facecolors=facecolors, edgecolor="black", linewidth=0.4)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()
    ax.view_init(elev=24, azim=35)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
