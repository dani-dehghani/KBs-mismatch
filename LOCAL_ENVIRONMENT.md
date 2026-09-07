# Optional local environment snapshot

The release asset
[`KBs-mismatch-local-macos-arm64-v1.tar.gz`](https://github.com/dani-dehghani/KBs-mismatch/releases/tag/local-environment-macos-arm64-v1)
contains the local-only files that are intentionally excluded from the normal
Git tree.

## Contents

- `.venv/` - the original Python environment for macOS ARM64
- `tmp/` - intermediate rendered PDF pages
- `.pytest_cache/` and `.ruff_cache/`
- Python `__pycache__/` directories
- macOS `.DS_Store` metadata files

The archive contains 19,662 files and is approximately 156 MiB compressed.
It does **not** contain `.git/`; GitHub already stores the repository history,
and nesting that metadata inside the project would be unsafe and redundant.

## Integrity

```text
SHA-256: 5ea4d8f2f59d4f8f83e44935b7b9f23aad82ac4777d3bccfe9836debe9d04830
```

## Restore

Download the release asset into the repository root, verify it, and extract it:

```bash
shasum -a 256 KBs-mismatch-local-macos-arm64-v1.tar.gz
tar -xzf KBs-mismatch-local-macos-arm64-v1.tar.gz
```

This snapshot was created on Apple Silicon and should only be treated as an
archival copy. For a clean and portable environment, use:

```bash
uv sync --extra dev --extra report
```
