"""Security reports: a dependency audit, a license report and canary coverage.

    uv run python scripts/security_report.py audit       # known vulnerabilities (OSV), locked
    uv run python scripts/security_report.py licenses    # every installed and web dependency
    uv run python scripts/security_report.py canaries    # no-text canary tests per package
    uv run python scripts/security_report.py all --out   # all three, to data/security/<date>.md

``audit`` sends the names and versions in ``uv.lock`` and ``web/package-lock.json`` (nothing
else) to OSV's batch API, https://api.osv.dev, and lists each known advisory. ``licenses``
reads the installed distributions' metadata and the web lock file, and sorts each license
into permissive, weak copyleft, or one to review (strong copyleft, unknown); ``--check`` fails
on one to review that ``REVIEWED`` doesn't name with its reason. ``canaries`` counts the tests
that use the no-text canary (``canary`` fixture, ``CanaryCheck``) in each package and fails when
a package has none. Each prints Markdown to stdout; ``--out`` writes it to a file instead, by
default under ``data/security/``, which Git ignores like the rest of ``data/``. With
``--check``, exit status 1 means something needs a person.
"""

import argparse
import importlib.metadata
import json
import re
import sys
import tomllib
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Where ``--out`` writes when given no file: ignored by Git, as all of data/ is.
REPORTS = ROOT / "data" / "security"
OSV_BATCH = "https://api.osv.dev/v1/querybatch"
PACKAGE_DIRS = (
    "packages/core",
    "packages/store",
    "packages/components",
    "packages/testing",
    "packages/sdk",
    "services/api",
    "services/worker",
)
# Listed, not required: the scripts handle fixtures and exports, not lab documents.
INFORMATIONAL_DIRS = ("scripts",)

PERMISSIVE = re.compile(
    r"^(MIT|MIT-0|MIT-CMU|ISC|0BSD|BSD|BSD-2-Clause|BSD-3-Clause|3-Clause BSD|Apache|Apache-2\.0|"
    r"PSF|PSF-2\.0|Python Software Foundation|Zlib|CC0-1\.0|Unlicense|BlueOak-1\.0\.0|"
    r"Python-2\.0|CNRI-Python|HPND|DFSG approved|OSI Approved)",
    re.IGNORECASE,
)
WEAK_COPYLEFT = re.compile(r"(LGPL|MPL|Mozilla Public License|EPL|CDDL)", re.IGNORECASE)
STRONG_COPYLEFT = re.compile(r"(AGPL|Affero|(?<!L)GPL|SSPL|EUPL|OSL)", re.IGNORECASE)
# The workspace's own packages, MIT like the rest of the repository. An editable install's
# metadata may not carry the license, so they're named here.
OWN = (
    "stratafi",
    "stratafi-api",
    "stratafi-components",
    "stratafi-core",
    "stratafi-gateway",
    "stratafi-store",
    "stratafi-testing",
    "stratafi-worker",
    "stratafi-web",
)
# Licenses a person reviewed, by package: the reason they're fine here.
REVIEWED = {
    **dict.fromkeys(OWN, "Stratafi's own, MIT"),
    "caniuse-lite": "browser-support data the web build reads; CC-BY-4.0 asks for credit only",
}


@dataclass(frozen=True)
class Dependency:
    ecosystem: str  # PyPI or npm
    name: str
    version: str
    license: str = ""
    dev: bool = False


# --- what's locked -----------------------------------------------------------------------------


def locked_python(lock: Path) -> list[Dependency]:
    """Registry packages in ``uv.lock`` (the workspace's own are left out)."""
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    found = []
    for package in data.get("package", []):
        source = package.get("source", {})
        if "registry" not in source:
            continue
        found.append(Dependency("PyPI", package["name"], package["version"]))
    return found


def locked_npm(lock: Path) -> list[Dependency]:
    """Packages in an npm lock file, version 2 or 3."""
    data = json.loads(lock.read_text(encoding="utf-8"))
    found = []
    for path, package in data.get("packages", {}).items():
        if not path or "version" not in package or package.get("link"):
            continue
        name = package.get("name") or path.rsplit("node_modules/", 1)[-1]
        found.append(
            Dependency(
                "npm",
                name,
                package["version"],
                str(package.get("license", "")),
                bool(package.get("dev", False)),
            )
        )
    return found


# --- audit -------------------------------------------------------------------------------------


def osv_queries(dependencies: Iterable[Dependency]) -> list[dict[str, object]]:
    return [
        {"package": {"name": d.name, "ecosystem": d.ecosystem}, "version": d.version}
        for d in dependencies
    ]


def query_osv(dependencies: list[Dependency], *, batch: int = 500) -> dict[Dependency, list[str]]:
    """Advisory IDs for each dependency that has any."""
    found: dict[Dependency, list[str]] = {}
    for start in range(0, len(dependencies), batch):
        chunk = dependencies[start : start + batch]
        body = json.dumps({"queries": osv_queries(chunk)}).encode()
        request = urllib.request.Request(
            OSV_BATCH, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            results = json.load(response)["results"]
        for dependency, result in zip(chunk, results, strict=True):
            ids = [vuln["id"] for vuln in result.get("vulns", [])]
            if ids:
                found[dependency] = ids
    return found


def audit_report(python: list[Dependency], npm: list[Dependency]) -> tuple[str, bool]:
    found = query_osv(python + npm)
    lines = [
        "## Dependency audit",
        "",
        f"OSV ({OSV_BATCH}) for {len(python)} Python packages in `uv.lock` and {len(npm)} npm "
        f"packages in `web/package-lock.json` ({sum(d.dev for d in npm)} of them development "
        "only).",
        "",
    ]
    if not found:
        lines.append("No known advisories.")
    else:
        lines += [
            "| Ecosystem | Package | Version | Dev only | Advisories |",
            "| --- | --- | --- | --- | --- |",
        ]
        for dependency, ids in sorted(found.items(), key=lambda item: item[0].name):
            lines.append(
                f"| {dependency.ecosystem} | {dependency.name} | {dependency.version} | "
                f"{'yes' if dependency.dev else 'no'} | {', '.join(sorted(ids))} |"
            )
    return "\n".join(lines) + "\n", not found


# --- licenses ----------------------------------------------------------------------------------


def installed_python() -> list[Dependency]:
    found = []
    for distribution in importlib.metadata.distributions():
        meta = distribution.metadata
        classifiers = [
            value.split("::")[-1].strip()
            for value in meta.get_all("Classifier") or []
            if value.startswith("License ::")
        ]
        text = (meta.get("License") or "").strip()
        license_ = (
            meta.get("License-Expression")
            or (" OR ".join(classifiers) if classifiers else "")
            or (text.splitlines()[0][:60] if text else "")
        )
        found.append(Dependency("PyPI", meta["Name"], meta["Version"], license_))
    return sorted(set(found), key=lambda d: d.name.lower())


RANK = {"permissive": 0, "weak copyleft": 1, "review": 2}


def category(license_: str) -> str:
    """permissive, weak copyleft, or review (strong copyleft or unknown). Of ``A OR B`` the
    most permissive choice counts; of ``A AND B`` the strictest."""
    text = license_.strip().strip("()").strip()
    if not text:
        return "review"
    if re.search(r"\sOR\s", text):
        return min((category(part) for part in re.split(r"\sOR\s", text)), key=RANK.__getitem__)
    if re.search(r"\sAND\s", text):
        return max((category(part) for part in re.split(r"\sAND\s", text)), key=RANK.__getitem__)
    if STRONG_COPYLEFT.search(text):
        return "review"
    if WEAK_COPYLEFT.search(text):
        return "weak copyleft"
    if PERMISSIVE.search(text):
        return "permissive"
    return "review"


def license_report(python: list[Dependency], npm: list[Dependency]) -> tuple[str, bool]:
    rows = [*python, *npm]
    counts: dict[str, int] = {}
    attention = []
    for dependency in rows:
        kind = category(dependency.license)
        counts[kind] = counts.get(kind, 0) + 1
        if kind != "permissive":
            attention.append((dependency, kind))
    unreviewed = [(d, kind) for d, kind in attention if kind == "review" and d.name not in REVIEWED]
    lines = [
        "## Licenses",
        "",
        f"{len(python)} installed Python distributions and {len(npm)} npm packages: "
        + ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))
        + ".",
        "",
        "| Ecosystem | Package | Version | License | Category | Reviewed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for dependency, kind in sorted(attention, key=lambda item: (item[1], item[0].name)):
        lines.append(
            f"| {dependency.ecosystem} | {dependency.name} | {dependency.version} | "
            f"{dependency.license or 'none given'} | {kind} | "
            f"{REVIEWED.get(dependency.name, '')} |"
        )
    lines += [
        "",
        "Every other package is permissive (MIT, BSD, Apache-2.0, ISC, PSF and the like).",
    ]
    return "\n".join(lines) + "\n", not unreviewed


# --- canaries ----------------------------------------------------------------------------------

CANARY_USE = re.compile(r"\bcanary\b\s*[:,)]|CanaryCheck|assert_not_logged|assert_absent")
TEST_DEF = re.compile(r"^def (test_\w+)\(([^)]*)", re.MULTILINE)


def canary_tests_in(package: Path) -> Iterator[str]:
    """``<file>::<test>`` for each test in ``package`` that uses the canary."""
    for path in sorted(package.rglob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        for match in TEST_DEF.finditer(text):
            end = text.find("\ndef ", match.end())
            body = text[match.start() : end if end != -1 else len(text)]
            if CANARY_USE.search(body):
                yield f"{path.name}::{match.group(1)}"


def canary_tests(package: Path) -> Iterator[str]:
    return canary_tests_in(package)


def canary_report(root: Path = ROOT) -> tuple[str, bool]:
    lines = [
        "## Canary coverage",
        "",
        "Tests that plant a canary in document, chunk or query text and check it reaches no "
        "log, metric, audit event or error.",
        "",
        "| Package | Canary tests |",
        "| --- | --- |",
    ]
    missing = []
    total = 0
    for directory in (*PACKAGE_DIRS, *INFORMATIONAL_DIRS):
        tests = list(canary_tests(root / directory))
        total += len(tests)
        if not tests and directory in PACKAGE_DIRS:
            missing.append(directory)
        note = " (informational)" if directory in INFORMATIONAL_DIRS else ""
        lines.append(f"| `{directory}`{note} | {len(tests)} |")
    lines += ["", f"{total} in all." + (f" None in: {', '.join(missing)}." if missing else "")]
    return "\n".join(lines) + "\n", not missing


# --- the command -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report", choices=("audit", "licenses", "canaries", "all"))
    parser.add_argument(
        "--out",
        type=Path,
        nargs="?",
        const=REPORTS / f"{stamp}.md",
        help="write the Markdown to this file, or to data/security/<date>.md, not stdout",
    )
    parser.add_argument("--check", action="store_true", help="exit 1 when something needs a person")
    args = parser.parse_args(argv)
    python_locked = locked_python(ROOT / "uv.lock")
    npm_locked = locked_npm(ROOT / "web" / "package-lock.json")
    sections, clean = [], True
    if args.report in ("audit", "all"):
        text, ok = audit_report(python_locked, npm_locked)
        sections.append(text)
        clean &= ok
    if args.report in ("licenses", "all"):
        text, ok = license_report(installed_python(), npm_locked)
        sections.append(text)
        clean &= ok
    if args.report in ("canaries", "all"):
        text, ok = canary_report()
        sections.append(text)
        clean &= ok
    document = f"# Security reports, {stamp}\n\n" + "\n".join(sections)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(document, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(document)
    return 0 if clean or not args.check else 1


if __name__ == "__main__":
    sys.exit(main())
