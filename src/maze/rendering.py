"""
Rendering utilities for the Labyrinth game (ported for bench).
"""
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
from maze.env import Board
from maze.assets import load_svg_asset, load_piece_svg
from maze.constants import PIECE_ROT_OFFSETS
import re


def _adjust_texture_opacity(svg_content: str, opacity: float) -> str:
    if opacity != 0.2:
        svg_content = re.sub(r'opacity="0\.2"', f'opacity="{opacity}"', svg_content)
    return svg_content


def render_board(board: Board, players: Optional[Dict[str, Tuple[int,int]]]=None, show_arrows: bool=True, cell:int=72, margin:int=0.2,
            highlight_insertion: Optional[Tuple[str, int]]=None, highlight_path: Optional[List[Tuple[int, int]]]=None,
            target_piece: Optional[str] = None, show_info: bool = True, show_legend: bool = False, show_harness: bool = False, show_guides: bool = True, texture_opacity: float = 0.2, show_coordinates: bool = False) -> str:
    players = players or {}

    PIECE_MARGIN_PX = margin
    PIECES_IN_A_ROW = board.w
    # Tile gap - set to 0 when show_coordinates to make tiles touch for better path visibility
    TILE_GAP = 0 if show_coordinates else 1.5

    total_width = (cell + TILE_GAP) * (PIECES_IN_A_ROW - 1) + cell
    total_height = (cell + TILE_GAP) * (PIECES_IN_A_ROW - 1) + cell + PIECE_MARGIN_PX * 0.6

    legend_panel_w = 0
    legend_panel_h = 0
    if show_legend:
        _legend_cell = cell * 0.45
        _legend_gap = PIECE_MARGIN_PX * 0.3
        _legend_vgap = _legend_gap + 12
        _cols = 4
        _rows = 3
        _grid_w = _legend_cell * _cols + _legend_gap * (_cols - 1)
        _grid_h = _legend_cell * _rows + _legend_vgap * (_rows - 1)
        _row_label_w = 18
        _panel_pad = 12
        _title_h = 20
        _header_gap = 8
        legend_panel_w = _row_label_w + _grid_w + _panel_pad * 2
        legend_panel_h = _title_h + _header_gap + _grid_h + _panel_pad * 2 + 14

    extra_area_width = max(cell * 1.2, legend_panel_w + PIECE_MARGIN_PX)

    wpx = total_width + extra_area_width + PIECE_MARGIN_PX * 2
    right_panel_height = 0
    if show_info:
        right_panel_height = cell + 40 + cell
        if show_legend:
            right_panel_height += 30 + legend_panel_h
    elif show_legend:
        right_panel_height = legend_panel_h
    hpx = max(total_height + PIECE_MARGIN_PX * 2, right_panel_height + PIECE_MARGIN_PX * 2)

    offset_x = 0
    offset_y = 0
    if show_arrows and show_guides:
        offset_x = 40
        offset_y = 40

    parts: List[str] = []

    if offset_x or offset_y:
        parts.append(f'<g transform="translate({offset_x},{offset_y})">')

    def text(x,y, txt, size=18, anchor="middle"):
        parts.append(f'<text x="{x}" y="{y}" font-family="monospace" font-size="{size}" text-anchor="{anchor}" dominant-baseline="middle" fill="black" font-weight="bold">{txt}</text>')

    def triangle(points, fill="black"):
        pts = " ".join(f"{x},{y}" for x,y in points)
        parts.append(f'<polygon points="{pts}" fill="{fill}"/>')

    max_right = 0.0
    max_bottom = 0.0

    base_width = (cell + TILE_GAP) * (PIECES_IN_A_ROW - 1) + cell
    base_height = (cell + TILE_GAP) * (PIECES_IN_A_ROW - 1) + cell
    expansion = PIECE_MARGIN_PX * 0.2

    board_area_width = base_width + expansion * 2
    board_area_height = base_height + expansion * 2
    board_area_x = PIECE_MARGIN_PX - expansion
    board_area_y = PIECE_MARGIN_PX - expansion
    board_center_y = board_area_y + board_area_height / 2
    parts.append(f'<rect x="{board_area_x}" y="{board_area_y}" width="{board_area_width}" height="{board_area_height}" fill="#e0e0e0" stroke="#e0e0e0" stroke-width="1" rx="8"/>')
    board_right = board_area_x + board_area_width
    board_bottom = board_area_y + board_area_height
    max_right = max(max_right, board_right + (12 if show_arrows else 0))
    max_bottom = max(max_bottom, board_bottom + (12 if show_arrows else 0))

    # Add coordinate labels on edges if requested
    if show_coordinates:
        coord_font_size = cell * 0.22
        coord_offset = 24  # Offset to position between board and edge labels
        # Column labels (x-axis) at top - moved further up
        for x in range(board.w):
            label_x = PIECE_MARGIN_PX + x * (cell + TILE_GAP) + cell / 2
            label_y = PIECE_MARGIN_PX - coord_offset
            parts.append(f'<text x="{label_x}" y="{label_y}" font-family="monospace" font-size="{coord_font_size}" text-anchor="middle" dominant-baseline="middle" fill="#333" font-weight="bold">{x}</text>')
        # Row labels (y-axis) on left - moved further left
        for y in range(board.h):
            label_x = PIECE_MARGIN_PX - coord_offset
            label_y = PIECE_MARGIN_PX + y * (cell + TILE_GAP) + cell / 2
            parts.append(f'<text x="{label_x}" y="{label_y}" font-family="monospace" font-size="{coord_font_size}" text-anchor="middle" dominant-baseline="middle" fill="#333" font-weight="bold">{y}</text>')

    parts.append(f'<defs>')
    for y in range(board.h):
        for x in range(board.w):
            parts.append(f'<clipPath id="clip_{x}_{y}">')
            parts.append(f'<rect x="0" y="0" width="{cell}" height="{cell}"/>')
            parts.append(f'</clipPath>')
    parts.append(f'<clipPath id="clip_extra">')
    parts.append(f'<rect x="0" y="0" width="{cell}" height="{cell}"/>')
    parts.append(f'</clipPath>')
    parts.append(f'</defs>')

    for y in range(board.h):
        for x in range(board.w):
            ox = PIECE_MARGIN_PX + x * (cell + TILE_GAP)
            oy = PIECE_MARGIN_PX + y * (cell + TILE_GAP)

            tile = board.grid[y][x]
            svg_content = load_svg_asset(tile.shape, tile.rot)
            svg_content = _adjust_texture_opacity(svg_content, texture_opacity)

            parts.append(f'<rect x="{ox}" y="{oy}" width="{cell}" height="{cell}" fill="#f5f5f5" stroke="none"/>')
            max_right = max(max_right, ox + cell)
            max_bottom = max(max_bottom, oy + cell)

            scale_factor = cell / 300
            parts.append(f'<g transform="translate({ox},{oy})" clip-path="url(#clip_{x}_{y})">')
            parts.append(f'<g transform="scale({scale_factor})">')
            parts.append(svg_content)
            parts.append('</g>')
            parts.append('</g>')

            for lab, (oxx, oyy) in board.objects.items():
                if oxx==x and oyy==y:
                    piece_svg = load_piece_svg(lab)
                    piece_size = cell * 0.9
                    offset = (piece_size / 2)
                    piece_x = ox + cell * 0.5 - offset
                    piece_y = oy + cell * 0.5 - offset
                    rot_q = (tile.rot + PIECE_ROT_OFFSETS.get(lab, 0)) % 4
                    piece_rotation = rot_q * 90
                    parts.append(f'<g transform="translate({piece_x},{piece_y}) scale({piece_size/300}) rotate({piece_rotation}, 150, 150)">')
                    parts.append(piece_svg)
                    parts.append('</g>')

            if show_harness:
                label_bg_x = ox + cell * 0.02
                label_bg_y = oy + cell * 0.06
                label_bg_w = cell * 0.96
                label_bg_h = cell * 0.22
                parts.append(f'<rect x="{label_bg_x}" y="{label_bg_y}" width="{label_bg_w}" height="{label_bg_h}" fill="#000" opacity="0.35" rx="4"/>')
                code = f"{tile.shape}{tile.rot}"
                piece_lab = None
                for _lab, (_oxx, _oyy) in board.objects.items():
                    if _oxx == x and _oyy == y:
                        piece_lab = _lab
                        break
                label_txt = f"({x},{y}) {code}" + (f" {piece_lab}" if piece_lab else "")
                text_x = ox + cell / 2
                text_y = label_bg_y + label_bg_h * 0.58
                parts.append(f'<text x="{text_x}" y="{text_y}" font-family="monospace" font-size="{cell*0.15}" text-anchor="middle" dominant-baseline="middle" fill="#fff">{label_txt}</text>')

    for lab,(px,py) in (players.items() if players else []):
        ox = PIECE_MARGIN_PX + px * (cell + TILE_GAP)
        oy = PIECE_MARGIN_PX + py * (cell + TILE_GAP)
        cx,cy = ox + cell/2, oy + cell/2
        parts.append(f'<circle cx="{cx+2}" cy="{cy+2}" r="{cell*0.25}" fill="#333" opacity="0.3"/>')
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{cell*0.25}" fill="#4CAF50" stroke="#2E7D32" stroke-width="3"/>')
        text(cx, cy, lab, size=20)

    if show_arrows:
        if show_guides:
            text(board_area_x - 28, board_center_y, "LEFT", size=12, anchor="end")
            text(board_area_x + board_area_width + 28, board_center_y, "RIGHT", size=12, anchor="start")
            text(board_area_x + board_area_width/2, board_area_y - 32, "TOP", size=12)
            text(board_area_x + board_area_width/2, board_area_y + board_area_height + 34, "BOTTOM", size=12)
        idxs = [i for i in range(board.w) if i % 2 == 1]
        for r in idxs:
            ymid = PIECE_MARGIN_PX + r * (cell + TILE_GAP) + cell/2
            left_edge = board_area_x - 12
            triangle([(left_edge, ymid), (left_edge+10, ymid-8), (left_edge+10, ymid+8)], fill="#8B4513")
            if show_guides and not show_coordinates:
                text(left_edge - 4, ymid, f"{r}", size=12, anchor="end")
            right_edge = board_area_x + board_area_width + 12
            triangle([(right_edge, ymid), (right_edge-10, ymid-8), (right_edge-10, ymid+8)], fill="#8B4513")
            if show_guides and not show_coordinates:
                text(right_edge + 4, ymid, f"{r}", size=12, anchor="start")
        for c in idxs:
            xmid = PIECE_MARGIN_PX + c * (cell + TILE_GAP) + cell/2
            top_edge = board_area_y - 12
            triangle([(xmid, top_edge), (xmid-8, top_edge+10), (xmid+8, top_edge+10)], fill="#8B4513")
            if show_guides and not show_coordinates:
                text(xmid, top_edge - 8, f"{c}", size=12)
            bottom_edge = board_area_y + board_area_height + 12
            triangle([(xmid, bottom_edge), (xmid-8, bottom_edge-10), (xmid+8, bottom_edge-10)], fill="#8B4513")
            if show_guides and not show_coordinates:
                text(xmid, bottom_edge + 8, f"{c}", size=12)

    if highlight_insertion:
        side, idx = highlight_insertion
        if side in ('left', 'right'):
            ymid = PIECE_MARGIN_PX + idx * (cell + TILE_GAP) + cell/2
            if side == 'left':
                left_edge = board_area_x - 12
                parts.append(f'<polygon points="{left_edge},{ymid} {left_edge+10},{ymid-8} {left_edge+10},{ymid+8}" fill="red" opacity="0.8"/>')
                parts.append(f'<rect x="{left_edge-2}" y="{ymid-10}" width="14" height="20" fill="none" stroke="red" stroke-width="2" rx="3"/>')
            else:
                right_edge = board_area_x + board_area_width + 12
                parts.append(f'<polygon points="{right_edge},{ymid} {right_edge-10},{ymid-8} {right_edge-10},{ymid+8}" fill="red" opacity="0.8"/>')
                parts.append(f'<rect x="{right_edge-12}" y="{ymid-10}" width="14" height="20" fill="none" stroke="red" stroke-width="2" rx="3"/>')
        else:
            xmid = PIECE_MARGIN_PX + idx * (cell + TILE_GAP) + cell/2
            if side == 'top':
                top_edge = board_area_y - 12
                parts.append(f'<polygon points="{xmid},{top_edge} {xmid-8},{top_edge+10} {xmid+8},{top_edge+10}" fill="red" opacity="0.8"/>')
                parts.append(f'<rect x="{xmid-10}" y="{top_edge-2}" width="20" height="14" fill="none" stroke="red" stroke-width="2" rx="3"/>')
            else:
                bottom_edge = board_area_y + board_area_height + 12
                parts.append(f'<polygon points="{xmid},{bottom_edge} {xmid-8},{bottom_edge-10} {xmid+8},{bottom_edge-10}" fill="red" opacity="0.8"/>')
                parts.append(f'<rect x="{xmid-10}" y="{bottom_edge-12}" width="20" height="14" fill="none" stroke="red" stroke-width="2" rx="3"/>')

    if highlight_path and len(highlight_path) >= 2:
        # Draw a path line with arrowhead from start to destination
        # Note: we're inside the offset_x/offset_y <g> group, so no need to add offset
        def _cell_center(x, y):
            cx = PIECE_MARGIN_PX + x * (cell + TILE_GAP) + cell / 2
            cy = PIECE_MARGIN_PX + y * (cell + TILE_GAP) + cell / 2
            return cx, cy
        pts = [_cell_center(x, y) for x, y in highlight_path]
        pts_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        parts.append(f'<polyline points="{pts_str}" fill="none" stroke="red" stroke-width="3" '
                     f'stroke-linecap="round" stroke-linejoin="round" opacity="0.7" '
                     f'marker-end="url(#arrowhead)"/>')
        # Arrowhead marker definition
        parts.insert(0, '<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" '
                     'refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" '
                     'fill="red" opacity="0.7"/></marker></defs>')
    elif highlight_path and len(highlight_path) == 1:
        # Single point — draw a circle marker at destination
        x, y = highlight_path[0]
        cx = PIECE_MARGIN_PX + x * (cell + TILE_GAP) + cell / 2
        cy = PIECE_MARGIN_PX + y * (cell + TILE_GAP) + cell / 2
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{cell * 0.3}" '
                     f'fill="none" stroke="red" stroke-width="3" opacity="0.7"/>')

    if show_info:
        ex_ox = PIECE_MARGIN_PX + PIECES_IN_A_ROW * (cell + TILE_GAP) + PIECE_MARGIN_PX * 2 + (28 if show_guides else 0)
        ex_oy = board_center_y - cell - 20
        parts.append(f'<rect x="{ex_ox}" y="{ex_oy}" width="{cell}" height="{cell}" fill="#f5f5f5" stroke="#8B4513" stroke-width="2"/>')
        max_right = max(max_right, ex_ox + cell)
        max_bottom = max(max_bottom, ex_oy + cell)
        svg_content = load_svg_asset(board.extra_tile.shape, board.extra_tile.rot)
        svg_content = _adjust_texture_opacity(svg_content, texture_opacity)
        parts.append(f'<g transform="translate({ex_ox},{ex_oy})" clip-path="url(#clip_extra)">')
        parts.append(f'<g transform="scale({cell/300})">')
        parts.append(svg_content)
        parts.append('</g>')
        parts.append('</g>')

        for lab, (oxx, oyy) in board.objects.items():
            if oxx == -1 and oyy == -1:
                piece_svg = load_piece_svg(lab)
                piece_size = cell * 0.8
                offset = (piece_size / 2)
                piece_x = ex_ox + cell * 0.5 - offset
                piece_y = ex_oy + cell * 0.5 - offset
                rot_q = (board.extra_tile.rot + PIECE_ROT_OFFSETS.get(lab, 0)) % 4
                piece_rotation = rot_q * 90
                parts.append(f'<g transform="translate({piece_x},{piece_y}) scale({piece_size/300}) rotate({piece_rotation}, 150, 150)">')
                parts.append(piece_svg)
                parts.append('</g>')

        text(ex_ox + cell/2, ex_oy - 20, "Extra Tile", size=14)
        parts.append(f'<rect x="{ex_ox-5}" y="{ex_oy-5}" width="{cell+10}" height="{cell+10}" fill="none" stroke="#8B4513" stroke-width="2" stroke-dasharray="5,5" opacity="0.7"/>')

        target_ox = ex_ox
        target_oy = board_center_y + 40
        parts.append(f'<rect x="{target_ox}" y="{target_oy}" width="{cell}" height="{cell}" fill="#fff3cd" stroke="#ffc107" stroke-width="2"/>')
        max_right = max(max_right, target_ox + cell)
        max_bottom = max(max_bottom, target_oy + cell)
        target_piece_svg = load_piece_svg(target_piece or 'I')
        target_size = cell * 0.8
        target_x = target_ox + cell * 0.5 - (target_size / 2)
        target_y = target_oy + cell * 0.5 - (target_size / 2)
        t_rot_q = PIECE_ROT_OFFSETS.get(target_piece or '', 0)
        t_rotation = (t_rot_q % 4) * 90
        parts.append(f'<g transform="translate({target_x},{target_y}) scale({target_size/300}) rotate({t_rotation}, 150, 150)">')
        parts.append(target_piece_svg)
        parts.append('</g>')
        text(target_ox + cell/2, target_oy - 10, "Target", size=14)

    if offset_x or offset_y:
        parts.append('</g>')

    wpx_final = max(wpx, max_right + PIECE_MARGIN_PX) + offset_x
    hpx_final = max(hpx, max_bottom + PIECE_MARGIN_PX) + offset_y
    header = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{wpx_final}" height="{hpx_final}" viewBox="0 0 {wpx_final} {hpx_final}">',
        f'<rect width="{wpx_final}" height="{hpx_final}" fill="#f5f5dc" stroke="none"/>'
    ]
    footer = ['</svg>']
    return "\n".join(header + parts + footer)



