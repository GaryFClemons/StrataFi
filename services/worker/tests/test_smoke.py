import stratafi_worker


def test_package_imports_with_a_version() -> None:
    assert stratafi_worker.__version__ == "0.1.0"
