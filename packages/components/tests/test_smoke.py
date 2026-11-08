import stratafi_components


def test_package_imports_with_a_version() -> None:
    assert stratafi_components.__version__ == "0.1.0"


def test_in_process_models_and_fakes_are_lab_only() -> None:
    """A knowledge profile that uses one is level C."""
    from stratafi_components.embed import FakeHashEmbed
    from stratafi_components.embedders import SentenceTransformersEmbed
    from stratafi_components.generate import FakeGenerate
    from stratafi_components.rerank import FakeOverlapRerank
    from stratafi_components.rerankers import CrossEncoderRerank

    for component in (
        SentenceTransformersEmbed,
        CrossEncoderRerank,
        FakeHashEmbed,
        FakeOverlapRerank,
        FakeGenerate,
    ):
        assert component.info.lab_only, component.info.id
