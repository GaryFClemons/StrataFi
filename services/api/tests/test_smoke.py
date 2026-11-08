import stratafi_api


def test_package_imports_with_a_version() -> None:
    assert stratafi_api.__version__ == "0.1.0"
