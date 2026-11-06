#!/usr/bin/env python3
"""Generate assets/banner.png in the exact ANSI Shadow block-art style (shades of red)."""

from pathlib import Path
import numpy as np
from PIL import Image


def _build_glyph_masks() -> dict[str, np.ndarray]:
    """Construct exact 26x11 subpixel-antialiased masks for ANSI Shadow glyphs."""
    empty = np.zeros((26, 11), dtype=np.float32)
    full = np.ones((26, 11), dtype=np.float32)

    # Vertical double-line profile across 11 columns: ..292.292..
    v_prof = np.array([0.0, 0.0, 0.25, 1.0, 0.25, 0.0, 0.25, 1.0, 0.25, 0.0, 0.0], dtype=np.float32)
    v_line = np.tile(v_prof[None, :], (26, 1))

    # Horizontal double-line profile across 26 rows (active at rows 10..15)
    h_line = np.zeros((26, 11), dtype=np.float32)
    for y, val in [(10, 0.25), (11, 1.0), (12, 0.25), (13, 0.25), (14, 1.0), (15, 0.25)]:
        h_line[y, :] = val

    # ╗ (top-right corner)
    tr = np.zeros((26, 11), dtype=np.float32)
    tr[10, :8] = 0.25
    tr[11, :8] = 1.0
    tr[11, 8] = 0.25
    tr[12, :6] = 0.25
    tr[12, 6:9] = [0.44, 1.0, 0.25]
    tr[13, :4] = 0.25
    tr[13, 6:9] = [0.25, 1.0, 0.25]
    tr[14, :4] = 1.0
    tr[14, 4] = 0.25
    tr[14, 6:9] = [0.25, 1.0, 0.25]
    tr[15, :2] = 0.25
    tr[15, 2:5] = [0.44, 1.0, 0.25]
    tr[15, 6:9] = [0.25, 1.0, 0.25]
    tr[16:, :] = v_prof[None, :]

    # ╔ (top-left corner = horizontal mirror of ╗)
    tl = tr[:, ::-1].copy()
    # ╝ (bottom-right corner = vertical mirror of ╗)
    br = tr[::-1, :].copy()
    # ╚ (bottom-left corner = both-axis mirror of ╗)
    bl = tr[::-1, ::-1].copy()

    return {
        " ": empty,
        "█": full,
        "║": v_line,
        "═": h_line,
        "╗": tr,
        "╔": tl,
        "╝": br,
        "╚": bl,
    }


LINE1_ROWS = [
    "██████╗ ██╗    ██╗    ██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗    ███████╗██████╗  █████╗  ██████╗███████╗",
    "██╔══██╗██║    ██║    ██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝    ██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝",
    "██████╔╝██║    ██║ █╗ ██║█████╗  ██║██║  ███╗███████║   ██║       ███████╗██████╔╝███████║██║     █████╗  ",
    "██╔═══╝ ██║    ██║███╗██║██╔══╝  ██║██║   ██║██╔══██║   ██║       ╚════██║██╔═══╝ ██╔══██║██║     ██╔══╝  ",
    "██║     ██║    ╚███╔███╔╝███████╗██║╚██████╔╝██║  ██║   ██║       ███████║██║     ██║  ██║╚██████╗███████╗",
    "╚═╝     ╚═╝     ╚══╝╚══╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝       ╚══════╝╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝",
]

LINE2_ROWS = [
    "      ██████╗ ██████╗  ██████╗  ██████╗ ██████╗  █████╗ ███╗   ███╗███╗   ███╗██╗███╗   ██╗ ██████╗       ",
    "      ██╔══██╗██╔══██╗██╔═══██╗██╔════╝ ██╔══██╗██╔══██╗████╗ ████║████╗ ████║██║████╗  ██║██╔════╝       ",
    "      ██████╔╝██████╔╝██║   ██║██║  ███╗██████╔╝███████║██╔████╔██║██╔████╔██║██║██╔██╗ ██║██║  ███╗      ",
    "      ██╔═══╝ ██╔══██╗██║   ██║██║   ██║██╔══██╗██╔══██║██║╚██╔╝██║██║╚██╔╝██║██║██║╚██╗██║██║   ██║      ",
    "      ██║     ██║  ██║╚██████╔╝╚██████╔╝██║  ██║██║  ██║██║ ╚═╝ ██║██║ ╚═╝ ██║██║██║ ╚████║╚██████╔╝      ",
    "      ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝       ",
]

# Monochromatic Red-Only Palette (Bright Scarlet -> Vivid Crimson -> Ruby -> Deep Blood Red)
LINE1_COLORS = [
    (255, 82, 82),    # #FF5252 - Bright Scarlet Red
    (255, 64, 64),    # #FF4040 - Vivid Light Red
    (244, 48, 48),    # #F43030 - Bright Crimson
    (229, 36, 36),    # #E52424 - Crimson Red
    (210, 28, 28),    # #D21C1C - Cardinal Red
    (192, 22, 22),    # #C01616 - Ruby Red
]

LINE2_COLORS = [
    (216, 32, 32),    # #D82020 - Bright Cardinal Red
    (198, 26, 26),    # #C61A1A - Ruby Red
    (180, 20, 20),    # #B41414 - Deep Ruby
    (162, 16, 16),    # #A21010 - Carmine Red
    (142, 12, 12),    # #8E0C0C - Garnet Red
    (122, 10, 10),    # #7A0A0A - Deep Oxblood Red
]


def generate_banner(out_path: Path) -> None:
    masks = _build_glyph_masks()
    n_cols = len(LINE1_ROWS[0])
    pad_x, pad_y = 24, 22
    gap_y = 14
    cell_w, cell_h = 11, 26
    width = n_cols * cell_w + 2 * pad_x
    height = 12 * cell_h + gap_y + 2 * pad_y - 10

    bg = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    canvas = np.tile(bg, (height, width, 1))

    for r, line in enumerate(LINE1_ROWS):
        fg = np.array(LINE1_COLORS[r], dtype=np.float32)
        y0 = pad_y + r * cell_h
        for c, ch in enumerate(line):
            mask = masks[ch][:, :, None]
            x0 = pad_x + c * cell_w
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = bg * (1.0 - mask) + fg * mask

    for r, line in enumerate(LINE2_ROWS):
        fg = np.array(LINE2_COLORS[r], dtype=np.float32)
        y0 = pad_y + 6 * cell_h + gap_y + r * cell_h
        for c, ch in enumerate(line):
            mask = masks[ch][:, :, None]
            x0 = pad_x + c * cell_w
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = bg * (1.0 - mask) + fg * mask

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(np.round(canvas), 0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "assets" / "banner.png"
    generate_banner(target)
    print(f"Generated {target}")
