#!/usr/bin/env python3
"""Create a compact, persistent review dashboard from a ScanCode JSON report."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path


VALID_STATUSES = {"pending", "include", "exclude"}
NOTICE_FILE_PREFIXES = ("license", "copying", "notice", "third_party_notice", "third-party-notice")


def component_for(path: str) -> str:
    parts = path.split("/")
    if parts and parts[0] == "CRESSim-Neo":
        parts = parts[1:]

    if len(parts) >= 3 and parts[0] == "extern" and parts[1] == "DiligentEngine":
        third_party_index = next(
            (index for index, part in enumerate(parts) if part == "ThirdParty"), None
        )
        if third_party_index is not None and len(parts) > third_party_index + 1:
            return "/".join(parts[: third_party_index + 2])
        return "extern/DiligentEngine"
    if len(parts) >= 2 and parts[0] == "extern":
        return "/".join(parts[:2])
    if parts[:3] == ["examples", "models", "psm"]:
        return "examples/models/psm"
    if parts[:2] == ["examples", "cubemaps"]:
        return "examples/cubemaps"
    if parts[:3] == ["examples", "physics", "fixtures"]:
        return "examples/physics/fixtures"
    return "first-party"


def notice_file_for(path: str) -> str | None:
    path_without_root = path.removeprefix("CRESSim-Neo/")
    filename = Path(path_without_root).name.lower()
    if filename.startswith(NOTICE_FILE_PREFIXES):
        return path_without_root
    return None


def values(items: object, keys: tuple[str, ...] = ("value", "url")) -> set[str]:
    found: set[str] = set()
    if not isinstance(items, list):
        return found
    for item in items:
        if isinstance(item, str):
            found.add(item)
        elif isinstance(item, dict):
            for key in keys:
                value = item.get(key)
                if isinstance(value, str) and value:
                    found.add(value)
                    break
    return found


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary_file:
        json.dump(data, temporary_file, indent=2)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


def add_line_references(references: set[tuple[int, int]], items: object) -> None:
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        start_line = item.get("start_line")
        end_line = item.get("end_line", start_line)
        if isinstance(start_line, int) and isinstance(end_line, int):
            references.add((start_line, end_line))


def format_line_references(references: set[tuple[int, int]]) -> str:
    if not references:
        return ""
    formatted = []
    for start_line, end_line in sorted(references):
        formatted.append(f"L{start_line}" if start_line == end_line else f"L{start_line}-{end_line}")
    return ":" + ", ".join(formatted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    with arguments.report.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    if arguments.review.exists():
        with arguments.review.open(encoding="utf-8") as review_file:
            review = json.load(review_file)
    else:
        review = {"schema_version": 1, "components": {}}

    if review.get("schema_version") != 1 or not isinstance(review.get("components"), dict):
        raise ValueError(f"Unsupported review file: {arguments.review}")

    components: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "paths": [],
            "line_references": defaultdict(set),
            "notice_files": set(),
            "licenses": set(),
            "holders": set(),
            "copyrights": set(),
            "urls": set(),
        }
    )
    for resource in report.get("files", []):
        path = resource["path"]
        component = components[component_for(path)]
        component["paths"].append(path)
        notice_file = notice_file_for(path)
        if notice_file:
            component["notice_files"].add(notice_file)
        references = component["line_references"][path]
        expression = resource.get("detected_license_expression_spdx") or resource.get(
            "detected_license_expression"
        )
        if expression:
            component["licenses"].add(expression)
        component["holders"].update(values(resource.get("holders")))
        component["copyrights"].update(values(resource.get("copyrights")))
        component["urls"].update(values(resource.get("urls")))
        for detection in resource.get("license_detections", []):
            if isinstance(detection, dict):
                add_line_references(references, detection.get("matches"))
        for field in ("copyrights", "holders", "authors", "emails", "urls"):
            add_line_references(references, resource.get(field))

    tracked_components = review["components"]
    for component, evidence in components.items():
        entry = tracked_components.setdefault(
            component,
            {
                "status": "pending",
                "notes": "",
                "notice_files": sorted(evidence["notice_files"]),
            },
        )
        status = entry.get("status")
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r} for {component}; expected one of {sorted(VALID_STATUSES)}")
        entry.setdefault("notes", "")
        entry.setdefault("notice_files", [])
        if not entry["notice_files"] and evidence["notice_files"]:
            entry["notice_files"] = sorted(evidence["notice_files"])
    write_json(arguments.review, review)

    counts = {status: 0 for status in VALID_STATUSES}
    for component in components:
        counts[tracked_components[component]["status"]] += 1

    lines = [
        "# Third-Party Scan Review",
        "",
        "Generated from ScanCode findings. Edit `compliance/third_party_review.json` to set each component's `status` to `include` or `exclude`; leave it `pending` until reviewed.",
        "",
        f"Components: {len(components)} — pending: {counts['pending']}, include: {counts['include']}, exclude: {counts['exclude']}.",
        "",
        "| Component | Decision | Evidence files | Licenses | Holders |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for name in sorted(components):
        component = components[name]
        licenses = ", ".join(sorted(component["licenses"])) or "—"
        holders = ", ".join(sorted(component["holders"])) or "—"
        lines.append(
            f"| `{escape_cell(name)}` | {tracked_components[name]['status']} | "
            f"{len(component['paths'])} | {escape_cell(licenses)} | {escape_cell(holders)} |"
        )

    for name in sorted(components):
        component = components[name]
        lines.extend(["", f"## `{name}`", ""])
        lines.append(f"Decision: **{tracked_components[name]['status']}**")
        if tracked_components[name]["notes"]:
            lines.append(f"Notes: {tracked_components[name]['notes']}")
        for label, key in (("Licenses", "licenses"), ("Holders", "holders"), ("Copyrights", "copyrights"), ("URLs", "urls")):
            entries = sorted(component[key])
            if entries:
                lines.append(f"{label}: " + "; ".join(entries))
        lines.append("Evidence paths:")
        for path in sorted(component["paths"]):
            lines.append(f"- `{path}`{format_line_references(component['line_references'][path])}")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote review dashboard: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
