"""
Labyrinth constants (ported for multimodal-reasoning-bench/maze).
"""

SIDES = ("left", "right", "top", "bottom")
ODD_IDXS = (1, 3, 5)

PIECE_NAMES = {
        'A': 'Knight Helmet', 'B': 'Candles', 'C': 'Dagger', 'D': 'Diamond',
        'E': 'Treasure', 'F': 'Ring', 'G': 'Holy Grail', 'H': 'Keys',
        'I': 'Crown', 'J': 'Potion', 'K': 'Coins', 'L': 'Book',
        'M': 'Bomb', 'N': 'Bat', 'O': 'Bug', 'P': 'Cat',
        'Q': 'Dinosaur', 'R': 'Ghost', 'S': 'Lizard', 'T': 'Mermaid',
        'U': 'Mouse', 'V': 'Owl', 'W': 'Pony', 'X': 'Cannon'
}

PIECE_MAPPING = {
        # Strategic treasure pieces (A-L)
        'A': 'KnightHelmet',
        'B': 'Candles',
        'C': 'Dagger',
        'D': 'Diamond',
        'E': 'Treasure',
        'F': 'Ring',
        'G': 'HolyGrail',
        'H': 'Keys',
        'I': 'Crown',
        'J': 'Potion',
        'K': 'Coins',
        'L': 'Book',

        # Additional random objects (M-X)
        'M': 'Bomb',          # Explosive piece
        'N': 'Bat',           # Flying piece
        'O': 'Bug',           # Insect piece
        'P': 'Cat',           # Pet piece
        'Q': 'Dinosaur',      # Ancient piece
        'R': 'Ghost',         # Spooky piece
        'S': 'Lizard',        # Reptile piece
        'T': 'Mermaid',       # Mythical piece
        'U': 'Mouse',         # Small piece
        'V': 'Owl',           # Wise piece
        'W': 'Pony',          # Animal piece
        'X': 'Cannon',        # Artillery piece
    }

# Map piece names to single letters for the object system
PIECE_LETTER_MAP = {v: k for k, v in PIECE_MAPPING.items()}

# Per-piece rotation offsets (quarter-turns) to correct base art orientation
# Keys are piece letters used in board.objects
PIECE_ROT_OFFSETS = {
    'A': 2,
    'B': 2,
}

# Strategic treasure letters occupy fixed anchor cells on the standard 7x7
# Labyrinth board.
STRATEGIC_PIECE_LETTERS = tuple("ABCDEFGHIJKL")

# Candidate anchor cells for strategic treasure pieces.
# Note: keys use (row, col) = (y, x).
STRATEGIC_ANCHOR_POSITIONS = (
            (0, 2),
            (0, 4),
            (2, 0),
            (2, 2),
            (2, 4),
            (2, 6),
            (4, 0),
            (4, 2),
            (4, 4),
            (4, 6),
            (6, 2),
            (6, 4),
        )

# Allowed quarter-turn orientations for strategic anchors.
STRATEGIC_ANCHOR_ORIENTATIONS = ("N", "E", "S", "W")

# Allowed tile shapes for strategic anchors when we want to broaden the board
# family beyond the classic all-T layout.
STRATEGIC_ANCHOR_SHAPE_CHOICES = ("T", "L", "I")

# Corner positions that must be corner tiles with correct orientation
CORNER_ORIENTATIONS = {
            (0, 0): 'E',
            (0, 6): 'S',
            (6, 0): 'N',
            (6, 6): 'W',
}



