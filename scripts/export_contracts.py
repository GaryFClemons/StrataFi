#!/usr/bin/env python3
"""Write the JSON Schemas of Stratafi's contracts to contracts/schemas/, or check they're current.

    uv run python scripts/export_contracts.py            # write them
    uv run python scripts/export_contracts.py --check    # exit 1 if any is stale (CI runs this)

Each contract (``stratafi_core.contracts.CONTRACTS``) gets ``contracts/schemas/<name>.schema.json``,
exported from its Pydantic model. ``contracts/contract-inventory.json`` lists every contract with
its schema, its model and its fixtures, so a tool outside Python can find and check each one. The
fixtures themselves are written by hand; a unit test checks that each one validates against both
its model and its exported schema.
"""

import argparse
import json
import sys
from pathlib import Path

from stratafi_core.contracts import CONTRACTS, json_schema, schema_filename

ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = ROOT / "contracts"
SCHEMAS_DIR = CONTRACTS_DIR / "schemas"
FIXTURES_DIR = CONTRACTS_DIR / "fixtures"
INVENTORY = CONTRACTS_DIR / "contract-inventory.json"


def _json(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def fixture_dir(name: str) -> Path:
    return FIXTURES_DIR / name.removeprefix("stratafi.")


def render(root: Path = CONTRACTS_DIR) -> dict[Path, str]:
    """Every generated file under ``root``, with its text."""
    files: dict[Path, str] = {}
    inventory = []
    for name in sorted(CONTRACTS):
        model, description = CONTRACTS[name]
        schema_path = root / "schemas" / schema_filename(name)
        files[schema_path] = _json(json_schema(name))
        fixtures = sorted(
            path.relative_to(root).as_posix()
            for path in (root / "fixtures" / name.removeprefix("stratafi.")).glob("*")
            if path.is_file()
        )
        inventory.append(
            {
                "name": name,
                "description": description,
                "schema": schema_path.relative_to(root).as_posix(),
                "model": f"{model.__module__}.{model.__qualname__}",
                "fixtures": fixtures,
            }
        )
    files[root / "contract-inventory.json"] = _json({"contracts": inventory})
    return files


def main(argv: list[str] | None = None, *, root: Path = CONTRACTS_DIR) -> int:
    parser = argparse.ArgumentParser(description="Export the contracts' JSON Schemas.")
    parser.add_argument("--check", action="store_true", help="fail if any file is stale")
    args = parser.parse_args(argv)
    files = render(root)
    if args.check:
        stale = [
            path
            for path, text in files.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        known = set(files)
        extra = [path for path in (root / "schemas").glob("*.json") if path not in known]
        if stale or extra:
            for path in [*stale, *extra]:
                print(f"{path.relative_to(root.parent).as_posix()} is stale", file=sys.stderr)
            print("run `uv run python scripts/export_contracts.py`", file=sys.stderr)
            return 1
        print(f"{len(files)} contract files are current")
        return 0
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(files)} contract files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
