"""Convert existing PNG figures to PDF for paper inclusion.

Wraps each PNG in a single-page PDF using matplotlib's PDF backend so the
page size matches the image dimensions exactly (no whitespace added).

Usage
-----
    # Convert all PNGs in figures/
    python -m src.convert_figures

    # Convert specific files
    python -m src.convert_figures figures/pca_dinov2_dinov3_radio.png

    # Convert to a specific output directory
    python -m src.convert_figures --out-dir figures/pdf/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"


def png_to_pdf(src: Path, dst: Path, dpi: int = 150) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    img = np.array(Image.open(src).convert("RGB"))
    h, w = img.shape[:2]
    # Figure size in inches = pixel size / dpi, so the PDF page matches the image.
    fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi))
    ax.imshow(img)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  {src.name}  →  {dst}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="*", type=Path,
                   help="PNG files to convert (default: all PNGs in figures/)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: same as input file)")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    srcs: list[Path] = list(args.inputs) if args.inputs else sorted(FIGURES_DIR.glob("*.png"))
    if not srcs:
        print("No PNG files found.", file=sys.stderr)
        return 1

    for src in srcs:
        src = Path(src)
        out_dir = args.out_dir or src.parent
        dst = out_dir / src.with_suffix(".pdf").name
        try:
            png_to_pdf(src, dst, dpi=args.dpi)
        except Exception as e:
            print(f"  [error] {src}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
