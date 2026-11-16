from __future__ import annotations

import os
from typing import Optional
from maze.constants import PIECE_MAPPING

# Base dir is this module directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
PIECES_DIR = os.path.join(ASSETS_DIR, 'pieces')


def load_svg_asset(shape: str, rotation: int) -> str:
    """Load and return the SVG content for a given tile shape and rotation.
    Expects tile assets under assets/.
    """
    shape_to_file = {
        'I': 'straight.svg',
        'L': 'corner.svg',
        'T': 't-shape.svg',
    }
    filename = shape_to_file.get(shape, 'straight.svg')
    filepath = os.path.abspath(os.path.join(ASSETS_DIR, filename))

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            svg_content = f.read()

        # Heuristic rotation of the main group (assets are 300x300)
        start_marker = '<g mask="url(#mask'
        start_idx = svg_content.find(start_marker)
        if start_idx != -1:
            g_start = svg_content.find('<g', start_idx)
            if g_start != -1:
                depth = 0
                g_end: Optional[int] = -1
                for i in range(g_start, len(svg_content)):
                    if svg_content[i:i+2] == '<g':
                        depth += 1
                    elif svg_content[i:i+4] == '</g>':
                        depth -= 1
                        if depth == 0:
                            g_end = i + 4
                            break
                if g_end and g_end != -1:
                    inner = svg_content[g_start:g_end]
                    total_rotation = (rotation * 90) % 360 if shape != 'T' else ((270 + rotation * 90) % 360)
                    if total_rotation != 0:
                        transform = f' transform="rotate({total_rotation} 150 150)"'
                        inner = inner.replace('<g', f'<g{transform}', 1)
                    return inner
        return svg_content
    except FileNotFoundError:
        return '<rect width="300" height="300" fill="#ccc" stroke="black" stroke-width="2"/>'


def load_piece_svg(piece_name: str) -> str:
    """Load and return the piece SVG content from assets/pieces; fallback to a circle."""
    actual_piece_name = PIECE_MAPPING.get(piece_name, piece_name)
    filepath = os.path.join(PIECES_DIR, f"{actual_piece_name}.svg")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return '<circle cx="150" cy="150" r="50" fill="#ff6b6b" stroke="#c92a2a" stroke-width="3"/>'



