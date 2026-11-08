#!/usr/bin/env python3
"""Write the API's OpenAPI document to contracts/openapi.json, or check that it's current.

    uv run python scripts/export_openapi.py            # write it
    uv run python scripts/export_openapi.py --check    # exit 1 if it's stale (CI runs this)

The app is built offline (no database, storage or keys), so the document depends only on the
code. After changing the API, run this, then regenerate the web client: `npm run api:generate`
in web/.
"""

import argparse
import json
import sys
from pathlib import Path

from stratafi_api.app import create_app
from stratafi_core.settings import OfflineSettings

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "contracts" / "openapi.json"


def render() -> str:
    app = create_app(OfflineSettings(), connect=False)
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None, *, output: Path = OUTPUT) -> int:
    parser = argparse.ArgumentParser(description="Export the API's OpenAPI document.")
    parser.add_argument("--check", action="store_true", help="fail if the file is stale")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = output.read_text(encoding="utf-8") if output.is_file() else ""
        if current != text:
            print(
                f"{output.relative_to(ROOT).as_posix()} is stale: run "
                "`uv run python scripts/export_openapi.py`, then `npm run api:generate` in web/",
                file=sys.stderr,
            )
            return 1
        print(f"{output.relative_to(ROOT).as_posix()} is current")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {output.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
