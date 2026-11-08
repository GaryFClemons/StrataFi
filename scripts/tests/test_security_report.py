"""The security reports: lock files read, licenses sorted, canaries counted.
The audit itself calls OSV, which the unit tier never does."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def report() -> ModuleType:
    path = ROOT / "scripts" / "security_report.py"
    spec = importlib.util.spec_from_file_location("script_security_report", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("license_", "kind"),
    [
        ("MIT", "permissive"),
        ("Apache-2.0 OR BSD-3-Clause", "permissive"),
        ("BSD License", "permissive"),
        ("LGPL-3.0-only", "weak copyleft"),
        ("MPL-2.0 AND MIT", "weak copyleft"),
        ("AGPL-3.0-only", "review"),
        ("GPL-3.0-or-later", "review"),
        ("GPL-2.0 OR MIT", "permissive"),
        ("", "review"),
        ("CC-BY-4.0", "review"),
    ],
)
def test_licenses_sort_into_categories(report: ModuleType, license_: str, kind: str) -> None:
    assert report.category(license_) == kind


def test_the_locks_are_read_without_the_workspaces_own_packages(report: ModuleType) -> None:
    python = report.locked_python(ROOT / "uv.lock")
    names = {dependency.name for dependency in python}
    assert "sqlalchemy" in names
    assert not any(name.startswith("stratafi") for name in names)
    npm = report.locked_npm(ROOT / "web" / "package-lock.json")
    assert any(dependency.name == "react" and not dependency.dev for dependency in npm)
    assert report.osv_queries(python[:1]) == [
        {
            "package": {"name": python[0].name, "ecosystem": "PyPI"},
            "version": python[0].version,
        }
    ]


def test_an_unreviewed_strong_copyleft_license_needs_a_person(report: ModuleType) -> None:
    gpl = report.Dependency("PyPI", "some-gpl-thing", "1.0", "GPL-3.0-only")
    text, clean = report.license_report([gpl], [])
    assert not clean
    assert "some-gpl-thing" in text
    ours = report.Dependency("PyPI", "stratafi-core", "0.1.0", "")
    assert report.license_report([ours], [])[1]


def test_a_report_prints_or_goes_to_the_ignored_data_folder(
    report: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert report.REPORTS == ROOT / "data" / "security"
    monkeypatch.setattr(report, "REPORTS", tmp_path / "security")
    assert report.main(["canaries"]) == 0
    assert capsys.readouterr().out.startswith("# Security reports, ")
    assert report.main(["canaries", "--out"]) == 0
    [written] = (tmp_path / "security").iterdir()
    assert written.suffix == ".md"
    assert written.read_text(encoding="utf-8").startswith("# Security reports, ")


def test_every_package_has_canary_tests(report: ModuleType) -> None:
    text, clean = report.canary_report()
    assert clean, text


def test_canary_tests_are_found_in_a_package(report: ModuleType, tmp_path: Path) -> None:
    tests = tmp_path / "pkg" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_x.py").write_text(
        "def test_a(canary) -> None:\n    canary.assert_not_logged()\n\n"
        "def test_b() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    found = [name.rsplit("::", 1)[1] for name in report.canary_tests_in(tmp_path / "pkg")]
    assert found == ["test_a"]
    assert json.dumps(found)
