import stratafi_store


def test_package_imports_with_a_version() -> None:
    assert stratafi_store.__version__ == "0.1.0"
