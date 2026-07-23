#!/usr/bin/env python3
"""Turn a ScanCode JSON report into durable review decisions and readable evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


VALID_STATUSES = {"pending", "include", "exclude"}
REVIEW_SCHEMA_VERSION = 2
MAX_SUMMARY_VALUES = 8
MAX_EXAMPLES_PER_VALUE = 3
MAX_SAMPLE_SOURCES = 12


def component_for(path: str) -> str:
    """Map a ScanCode path to a reviewable dependency or project-owned bucket."""
    parts = path.split("/")
    if parts and parts[0] == "CRESSim-Neo":
        parts = parts[1:]

    if len(parts) >= 3 and parts[:2] == ["extern", "DiligentEngine"]:
        third_party_index = next(
            (index for index, part in enumerate(parts) if part == "ThirdParty"), None
        )
        if third_party_index is not None:
            # The CMake file beside Diligent's nested dependencies is metadata,
            # not a dependency named "CMakeLists.txt".
            if len(parts) == third_party_index + 2 and parts[-1].lower().startswith("cmakelists"):
                return "/".join(parts[: third_party_index + 1])
            if len(parts) > third_party_index + 1:
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
    return "project files with third-party clues"


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


def format_line_references(references: list[list[int]]) -> str:
    if not references:
        return ""
    formatted = []
    for start_line, end_line in references:
        formatted.append(f"L{start_line}" if start_line == end_line else f"L{start_line}-{end_line}")
    return ":" + ", ".join(formatted)


def source_evidence(resource: dict[str, object]) -> dict[str, object] | None:
    """Keep all useful ScanCode findings, without ScanCode's presentation noise."""
    references: set[tuple[int, int]] = set()
    licenses: set[str] = set()
    detections = resource.get("license_detections")
    if isinstance(detections, list):
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            expression = detection.get("license_expression_spdx") or detection.get("license_expression")
            if isinstance(expression, str) and expression:
                licenses.add(expression)
            add_line_references(references, detection.get("matches"))
    expression = resource.get("detected_license_expression_spdx") or resource.get(
        "detected_license_expression"
    )
    if isinstance(expression, str) and expression:
        licenses.add(expression)

    copyrights = values(resource.get("copyrights"))
    holders = values(resource.get("holders"))
    authors = values(resource.get("authors"))
    emails = values(resource.get("emails"))
    urls = values(resource.get("urls"))
    for field in ("copyrights", "holders", "authors", "emails", "urls"):
        add_line_references(references, resource.get(field))

    kinds = []
    if licenses:
        kinds.append("license")
    if copyrights or holders or authors:
        kinds.append("copyright")
    if emails:
        kinds.append("email")
    if urls:
        kinds.append("url")
    if not kinds:
        return None
    return {
        "path": resource["path"],
        "kinds": kinds,
        "line_ranges": [list(line_range) for line_range in sorted(references)],
        "licenses": sorted(licenses),
        "copyrights": sorted(copyrights),
        "holders": sorted(holders),
        "authors": sorted(authors),
        "emails": sorted(emails),
        "urls": sorted(urls),
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary_file:
        json.dump(data, temporary_file, indent=2)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(path)


def source_sort_key(source: dict[str, object]) -> tuple[int, str]:
    # Put the legal clues a reviewer needs first; URLs alone are lowest signal.
    priority = {"license": 0, "copyright": 1, "email": 2, "url": 3}
    kinds = source["kinds"]
    assert isinstance(kinds, list)
    return (min(priority[kind] for kind in kinds), str(source["path"]))


def compact_sources(sources: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "path": source["path"],
            "kinds": source["kinds"],
            "line_ranges": source["line_ranges"],
        }
        for source in sorted(sources, key=source_sort_key)[:MAX_SAMPLE_SOURCES]
    ]


def summarized_values(sources: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source in sources:
        values_for_source = source[key]
        assert isinstance(values_for_source, list)
        for value in values_for_source:
            if isinstance(value, str):
                occurrences[value].append(source)

    result = []
    for value, matched_sources in sorted(occurrences.items(), key=lambda item: (-len(item[1]), item[0]))[:MAX_SUMMARY_VALUES]:
        result.append(
            {
                "value": value,
                "source_count": len(matched_sources),
                "examples": [
                    {"path": source["path"], "line_ranges": source["line_ranges"]}
                    for source in sorted(matched_sources, key=source_sort_key)[:MAX_EXAMPLES_PER_VALUE]
                ],
            }
        )
    return result


def evidence_summary(sources: list[dict[str, object]]) -> dict[str, object]:
    kinds = Counter(kind for source in sources for kind in source["kinds"])
    url_only_sources = [source for source in sources if source["kinds"] == ["url"]]
    return {
        "source_count": len(sources),
        "license_source_count": kinds["license"],
        "copyright_source_count": kinds["copyright"],
        "email_source_count": kinds["email"],
        "url_only_source_count": sum(source["kinds"] == ["url"] for source in sources),
        "license_expressions": summarized_values(sources, "licenses"),
        "copyright_examples": summarized_values(sources, "copyrights"),
        "sample_sources": compact_sources(sources),
        "url_only_examples": [
            {
                "path": source["path"],
                "line_ranges": source["line_ranges"],
                "urls": source["urls"],
            }
            for source in sorted(url_only_sources, key=source_sort_key)[:MAX_EXAMPLES_PER_VALUE]
        ],
    }


def default_review_entry(component: str) -> dict[str, object]:
    # A project-owned file can still embed third-party code or assets. Keep it
    # pending until a reviewer has distinguished a mere reference from content
    # that is distributed with the project.
    if component == "project files with third-party clues":
        return {
            "status": "pending",
            "notes": "Review whether these are only references or contain distributed third-party material.",
            "notice_files": [],
        }
    return {"status": "pending", "notes": "", "notice_files": []}


def load_review(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"schema_version": REVIEW_SCHEMA_VERSION, "components": {}}
    with path.open(encoding="utf-8") as review_file:
        review = json.load(review_file)
    if review.get("schema_version") not in {1, REVIEW_SCHEMA_VERSION} or not isinstance(review.get("components"), dict):
        raise ValueError(f"Unsupported review file: {path}")
    # Version 1 stored all generated evidence in the durable review registry.
    # Preserve only the user-owned decisions when migrating it.
    review["schema_version"] = REVIEW_SCHEMA_VERSION
    for entry in review["components"].values():
        if isinstance(entry, dict):
            entry.pop("evidence", None)
    components = review["components"]
    assert isinstance(components, dict)
    # Repair names written by the first version of this helper. These are
    # mechanical names, not user choices, so preserving the entry under the
    # corrected key preserves all decisions without leaving a fake component.
    for name in tuple(components):
        corrected = None
        if name in {"first-party", "first-party references"}:
            corrected = "project files with third-party clues"
        elif name.endswith("/ThirdParty/CMakeLists.txt"):
            corrected = name.removesuffix("/CMakeLists.txt")
        if corrected and corrected not in components:
            entry = components.pop(name)
            if (
                name == "first-party references"
                and isinstance(entry, dict)
                and entry.get("status") == "exclude"
                and entry.get("notes") == "Project-owned files and dependency references; not a third-party component."
            ):
                components[corrected] = default_review_entry(corrected)
            else:
                components[corrected] = entry
        elif corrected:
            components.pop(name)
    return review


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


def display_values(items: list[dict[str, object]]) -> str:
    values = [str(item["value"]) for item in items]
    return "; ".join(values) if values else "—"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    with arguments.report.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    review = load_review(arguments.review)
    tracked_components = review["components"]
    assert isinstance(tracked_components, dict)

    grouped_sources: dict[str, list[dict[str, object]]] = defaultdict(list)
    for resource in report.get("files", []):
        if not isinstance(resource, dict) or not isinstance(resource.get("path"), str):
            continue
        source = source_evidence(resource)
        if source:
            grouped_sources[component_for(str(resource["path"]))].append(source)

    component_summaries: dict[str, dict[str, object]] = {}
    for component, sources in grouped_sources.items():
        sources.sort(key=lambda source: str(source["path"]))
        entry = tracked_components.setdefault(component, default_review_entry(component))
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid review entry for {component!r}")
        status = entry.get("status")
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r} for {component}; expected one of {sorted(VALID_STATUSES)}")
        entry.setdefault("notes", "")
        entry.setdefault("notice_files", [])
        component_summaries[component] = evidence_summary(sources)

    write_json(arguments.review, review)
    write_json(
        arguments.evidence,
        {
            "schema_version": 1,
            "source_report": str(arguments.report),
            "components": {
                component: {"summary": component_summaries[component], "sources": grouped_sources[component]}
                for component in sorted(grouped_sources)
            },
        },
    )

    counts = Counter(
        str(tracked_components[component]["status"])
        for component in grouped_sources
    )
    lines = [
        "# Third-Party Scan Review",
        "",
        "This dashboard is generated from ScanCode findings. The full grouped evidence, including exact source paths and line ranges, is in `build/compliance/third_party_evidence.json`; the untouched ScanCode report remains the authoritative raw record.",
        "",
        "Edit `compliance/third_party_review.json` only to set a component's `status`, explain `notes`, and select canonical `notice_files` for included components. Regenerating never overwrites those fields.",
        "",
        f"Components: {len(grouped_sources)} — pending: {counts['pending']}, include: {counts['include']}, exclude: {counts['exclude']}.",
        "",
        "| Component | Decision | Sources | License / copyright / email / URL-only | License expressions |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for name in sorted(grouped_sources):
        summary = component_summaries[name]
        lines.append(
            f"| `{escape_cell(name)}` | {tracked_components[name]['status']} | {summary['source_count']} | "
            f"{summary['license_source_count']} / {summary['copyright_source_count']} / {summary['email_source_count']} / {summary['url_only_source_count']} | "
            f"{escape_cell(display_values(summary['license_expressions']))} |"
        )

    for name in sorted(grouped_sources):
        summary = component_summaries[name]
        lines.extend(["", f"## `{name}`", ""])
        lines.append(f"Decision: **{tracked_components[name]['status']}**")
        if tracked_components[name]["notes"]:
            lines.append(f"Notes: {tracked_components[name]['notes']}")
        lines.append(
            "Findings: "
            f"{summary['source_count']} sources; {summary['license_source_count']} license, "
            f"{summary['copyright_source_count']} copyright, {summary['email_source_count']} email, "
            f"{summary['url_only_source_count']} URL-only."
        )
        if summary["license_expressions"]:
            lines.append("License expressions: " + display_values(summary["license_expressions"]))
        if summary["copyright_examples"]:
            lines.append("Copyright examples: " + display_values(summary["copyright_examples"]))
        if summary["url_only_examples"]:
            lines.append("URL-only examples:")
            for source in summary["url_only_examples"]:
                lines.append(
                    f"- `{source['path']}`{format_line_references(source['line_ranges'])}: "
                    + "; ".join(source["urls"])
                )
        lines.append("Review samples:")
        for source in summary["sample_sources"]:
            lines.append(
                f"- **{', '.join(source['kinds'])}** `{source['path']}`{format_line_references(source['line_ranges'])}"
            )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote review registry: {arguments.review}")
    print(f"Wrote grouped evidence: {arguments.evidence}")
    print(f"Wrote review dashboard: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
