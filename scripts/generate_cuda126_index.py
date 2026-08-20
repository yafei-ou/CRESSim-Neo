#!/usr/bin/env python3
"""Create a static PEP 503 simple index for CUDA 12.6 wheel artifacts."""

from __future__ import annotations

import argparse
import html
import shutil
from pathlib import Path


def page(title: str, links: list[str]) -> str:
    body = "\n".join(f'<a href="{html.escape(link)}">{html.escape(link)}</a><br/>' for link in links)
    return f"<!doctype html><html><head><title>{html.escape(title)}</title></head><body>{body}</body></html>\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    wheels = sorted(args.wheels.glob("cressim_neo-*.whl"))
    if not wheels:
        raise SystemExit(f"No cressim_neo wheels found in {args.wheels}")

    project = args.output / "cressim-neo"
    project.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for wheel in wheels:
        destination = project / wheel.name
        shutil.copy2(wheel, destination)
        names.append(wheel.name)
    (project / "index.html").write_text(page("cressim-neo CUDA 12.6", names), encoding="utf-8")
    (args.output / "index.html").write_text(page("CRESSim-Neo CUDA wheels", ["cressim-neo/"]), encoding="utf-8")


if __name__ == "__main__":
    main()
