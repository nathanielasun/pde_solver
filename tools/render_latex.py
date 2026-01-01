#!/usr/bin/env python3
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Render LaTeX to PNG")
    parser.add_argument("--latex", required=True, help="LaTeX input")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for rendering")
    parser.add_argument("--color", default="white", help="Text color")
    parser.add_argument("--fontsize", type=int, default=18, help="Font size (points)")
    args = parser.parse_args()

    text = args.latex.strip()
    if not text:
        print("LaTeX input is empty", file=sys.stderr)
        return 1

    if "$" not in text:
        text = f"${text}$"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable: {exc}", file=sys.stderr)
        return 1

    try:
        fig = plt.figure(figsize=(6, 1.6), dpi=args.dpi)
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.01, 0.5, text, fontsize=args.fontsize, va="center", color=args.color)
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1, transparent=True)
        plt.close(fig)
    except Exception as exc:
        print(f"render failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
