# Third-Party Compliance

`THIRD_PARTY_NOTICES.md` is the tracked notice distributed with the C++ SDK and
Python package. Do not generate it from license-detection results alone: the
review must include only third-party code, assets, headers, and binaries that
are actually shipped by a supported distribution profile.

## Review workflow

Initialize submodules, then create a candidate ScanCode inventory. Include
Diligent for a full review:

```bash
git submodule update --init --recursive
scripts/scan_third_party.sh --include-diligent
scripts/summarize_third_party_scan.sh
```

The raw report is `build/compliance/scancode.json`. The summary produces a
review dashboard at `build/compliance/third_party_review.md` and grouped
evidence at `build/compliance/third_party_evidence.json`.

Review and record each component in `compliance/third_party_review.json` as
`pending`, `include`, or `exclude`. Included source components must name
canonical `notice_files` or precise `notice_sources`; downloaded artifacts are
resolved through `compliance/third_party_artifacts.json`. Scan results are
candidates for review, not a shipping notice.

## Generate the tracked notice

After updating the review registry, generate the notice. For a release that
bundles DXC, give the generator the matching configured build directory so it
can find the downloaded artifact notices:

```bash
python3 scripts/generate_third_party_notice.py \
  --build-dir build/linux-release \
  --output THIRD_PARTY_NOTICES.md
```

Review and commit the generated notice with its registry changes. The generator
excludes internal review notes by default; use `--include-notes` only for a
review draft. CUDA is user-supplied and is not included in the notice unless a
future distribution bundles CUDA components.
