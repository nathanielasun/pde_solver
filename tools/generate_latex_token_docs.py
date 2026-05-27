#!/usr/bin/env python3
"""Generate docs/reference/LATEX_TOKEN_REGISTRY.md from the C++ token registry."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
EXPORT_BIN = BUILD_DIR / "latex_token_export"
OUT_PATH = ROOT / "docs" / "reference" / "LATEX_TOKEN_REGISTRY.md"


def build_exporter() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cmake", "-S", str(ROOT), "-B", str(BUILD_DIR)],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(BUILD_DIR), "--target", "latex_token_export"],
        check=True,
    )


def load_catalog() -> dict:
    if not EXPORT_BIN.exists():
        build_exporter()
    result = subprocess.run(
        [str(EXPORT_BIN)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def render_markdown(catalog: dict) -> str:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for entry in catalog.get("entries", []):
        by_category[entry.get("category", "unknown")].append(entry)

    lines = [
        "# LaTeX Token Registry",
        "",
        "This document is **generated** from the authoritative C++ registry. "
        "Do not edit by hand; regenerate with:",
        "",
        "```bash",
        "python3 tools/generate_latex_token_docs.py",
        "```",
        "",
        "Sources: [`include/latex_patterns.h`](../include/latex_patterns.h), "
        "[`include/latex_token_registry.h`](../include/latex_token_registry.h), "
        "[`src/latex_token_registry.cpp`](../src/latex_token_registry.cpp).",
        "",
    ]

    lines.append("## Normalization rules")
    lines.append("")
    lines.append("| From | To |")
    lines.append("| --- | --- |")
    for rule in catalog.get("normalization", []):
        lines.append(f"| `{rule['from']}` | `{rule['to']}` |")
    lines.append("")

    lines.append("## Conservation rewrites (parse-time)")
    lines.append("")
    lines.append("| From | To | Note |")
    lines.append("| --- | --- | --- |")
    for rule in catalog.get("conservation_rewrites", []):
        lines.append(
            f"| `{rule['from']}` | `{rule['to']}` | {rule.get('note', '')} |"
        )
    lines.append("")

    lines.append("## PDE pattern tokens by category")
    lines.append("")
    for category in sorted(by_category.keys()):
        lines.append(f"### {category}")
        lines.append("")
        lines.append("| Pattern | Maps to | Display |")
        lines.append("| --- | --- | --- |")
        for entry in by_category[category]:
            lines.append(
                f"| `{entry.get('pattern', '')}` | `{entry.get('maps_to', '')}` | "
                f"{entry.get('display_note', '')} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    OUT_PATH.write_text(render_markdown(catalog), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
