"""Stratafi's built-in pipeline components.

``default_registry()`` holds every built-in component this installation can run. Heavy
libraries sit behind extras; a component whose extra isn't installed is still registered, and
refuses to run with ``extra_missing``.
"""

from stratafi_core.components import Component, ComponentRegistry

__version__ = "0.1.0"


def builtin_components() -> list[type[Component]]:
    """The built-in components, in catalog order: each stage's, the query-time ones, then the
    trainers."""
    from stratafi_components import (
        chunk,
        chunkers,
        clean,
        embed,
        embedders,
        enrich,
        extract,
        generate,
        index,
        parsers,
        pgindex,
        rerank,
        rerankers,
        scan,
        trainers,
    )

    return [
        *extract.COMPONENTS,
        *scan.COMPONENTS,
        *parsers.components(),
        *clean.COMPONENTS,
        *chunk.COMPONENTS,
        *chunkers.COMPONENTS,
        *enrich.COMPONENTS,
        *embed.COMPONENTS,
        *embedders.COMPONENTS,
        *index.COMPONENTS,
        *pgindex.COMPONENTS,
        *rerank.COMPONENTS,
        *rerankers.COMPONENTS,
        *generate.COMPONENTS,
        *trainers.COMPONENTS,
    ]


def default_registry() -> ComponentRegistry:
    return ComponentRegistry(builtin_components())
