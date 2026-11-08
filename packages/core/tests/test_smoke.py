import stratafi_core


def test_package_imports_with_a_version() -> None:
    assert stratafi_core.__version__ == "0.1.0"
