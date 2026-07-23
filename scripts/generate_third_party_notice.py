#!/usr/bin/env python3
"""Generate a draft third-party notice from reviewed canonical notice files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VALID_STATUSES = {"pending", "include", "exclude"}
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def repository_file(relative_path: str) -> Path:
    candidate = (REPOSITORY_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(f"Notice file is outside the repository: {relative_path}") from error
    if not candidate.is_file():
        raise ValueError(f"Notice file does not exist: {relative_path}")
    return candidate


def rebase_markdown_headings(text: str, levels: int = 2) -> str:
    """Place an imported Markdown document beneath a component heading."""
    rebased_lines: list[str] = []
    in_fenced_code_block = False
    for line in text.splitlines():
        if line.startswith("```") or line.startswith("~~~"):
            in_fenced_code_block = not in_fenced_code_block
        if not in_fenced_code_block:
            heading = re.match(r"^(#{1,6})(\s+.*)$", line)
            if heading:
                level = min(len(heading.group(1)) + levels, 6)
                line = "#" * level + heading.group(2)
        rebased_lines.append(line)
    return "\n".join(rebased_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review",
        type=Path,
        default=REPOSITORY_ROOT / "compliance/third_party_review.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "build/compliance/THIRD_PARTY_NOTICES.draft.md",
    )
    arguments = parser.parse_args()

    with arguments.review.open(encoding="utf-8") as review_file:
        review = json.load(review_file)
    if review.get("schema_version") != 1 or not isinstance(review.get("components"), dict):
        raise ValueError(f"Unsupported review file: {arguments.review}")

    included_components: list[tuple[str, dict[str, object]]] = []
    for name, entry in sorted(review["components"].items()):
        status = entry.get("status")
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r} for {name}; expected one of {sorted(VALID_STATUSES)}")
        if status == "include":
            notice_files = entry.get("notice_files")
            if not isinstance(notice_files, list) or not notice_files:
                raise ValueError(f"Included component {name!r} must declare one or more notice_files.")
            included_components.append((name, entry))

    lines = [
        "# Third-Party Notices (Draft)",
        "",
        "This file is generated from `compliance/third_party_review.json`. Do not edit it directly; update the review registry and regenerate it.",
    ]
    if not included_components:
        lines.extend(["", "No components have been marked `include`."])

    written_files: set[Path] = set()
    for name, entry in included_components:
        lines.extend(["", f"## {name}"])
        notes = entry.get("notes")
        if isinstance(notes, str) and notes:
            lines.extend(["", notes])
        for relative_path in entry["notice_files"]:
            if not isinstance(relative_path, str):
                raise ValueError(f"Notice file for {name!r} must be a string.")
            source_path = repository_file(relative_path)
            if source_path in written_files:
                lines.extend(["", f"The canonical notice `{relative_path}` is included above."])
                continue
            written_files.add(source_path)
            text = source_path.read_text(encoding="utf-8")
            lines.extend(
                [
                    "",
                    f"_Source: `{relative_path}`_",
                    "",
                    rebase_markdown_headings(text).rstrip(),
                ]
            )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote third-party notice draft: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
