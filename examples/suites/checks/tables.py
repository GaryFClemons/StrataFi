"""Custom checks for examples/suites/tables.yaml.

The lab runs them in its sandbox, over the run's chunks and documents, only where the owner
turned custom checks on after the sandbox's security review. Try one on your own records first:

    from stratafi.checks import CheckContext
    ctx = CheckContext.from_records("bundle", chunks=[...], documents=[...])
    print(tables_not_split(ctx).to_dict())

A result's detail holds IDs, codes and numbers: the lab drops anything that reads as text.
"""

from stratafi.checks import CheckContext, CheckResult


def tables_not_split(ctx: CheckContext) -> CheckResult:
    """Every table stays in one chunk: in the Halcyon handbook some answers sit only in a table
    row (pay bands, wholesale price tiers, holding times), and a split table loses its header."""
    split = [c.chunk_id for c in ctx.chunks() if c.metadata.get("table_part")]
    return CheckResult.assert_true(not split, value=len(split), detail={"chunks": split[:20]})


def every_document_has_chunks(ctx: CheckContext) -> CheckResult:
    """Each document the run parsed gave at least one chunk."""
    chunked = {chunk.document_id for chunk in ctx.chunks()}
    parsed = [d for d in ctx.documents() if d.parse_status in ("succeeded", "cached")]
    empty = [d.document_id for d in parsed if d.document_id not in chunked]
    if not parsed:
        return CheckResult.skip("nothing_parsed")
    return CheckResult.assert_true(
        not empty, value=len(empty), detail={"documents": empty[:20]}, code="documents_unchunked"
    )
