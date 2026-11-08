import pytest

from stratafi import __version__
from stratafi.cli import main


def test_cli_reports_its_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"stratafi {__version__}"
