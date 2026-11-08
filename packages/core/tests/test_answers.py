"""Answer-level evaluation: pinned prompts, the graders, answer metrics, judge
calibration and drafted questions."""

import json

import pytest

from stratafi_core.answers import (
    NOT_IN_SOURCES,
    PROMPTS,
    ContextPassage,
    Verdicts,
    abstains,
    agreement,
    answer_request,
    automatic_label,
    citations_in,
    detect_refusal,
    draft_request,
    grade_citation,
    grade_exact,
    grade_refusal,
    grade_regex,
    grade_rubric,
    parse_judge_score,
    parse_questions,
    render_sources,
    score_answer,
    source_id,
    summarize,
    token_f1,
)

PASSAGES = [
    ContextPassage(
        "refunds-v3-p1",
        "refunds",
        "3",
        "2.1 Refund window",
        "A drink can be refunded within 14 calendar days of purchase, with the receipt.",
        True,
    ),
    ContextPassage(
        "refunds-v2-p2",
        "refunds",
        "2",
        "2.1 Refund window",
        "Refunds are allowed for 30 days.",
        False,
    ),
]


def test_prompts_are_pinned_by_version_and_hash() -> None:
    """Changing a prompt's text changes its hash; a change needs a new version, and this
    test updated with it."""
    assert {prompt_id: prompt.sha256[:12] for prompt_id, prompt in PROMPTS.items()} == {
        "grounded-qa-v2": "ddceefa381ab",
        "judge-faithfulness-v2": "42bf28fcbd5f",
        "judge-correctness-v2": "8bd57b8efd98",
        "draft-questions-v2": "bd959b27395b",
    }
    request = answer_request("How long is the refund window?", PASSAGES, day="2026-11-02")
    system, user = request.messages()
    assert "an organization's documents" in system["content"]
    assert f'"{NOT_IN_SOURCES}"' in system["content"]
    assert user["content"].startswith("Today's date is 2026-11-02.")
    assert "[refunds-v3-p1] refunds, version 3, 2.1 Refund window" in user["content"]
    drafted = draft_request("refunds", "Some text.", count=2)
    assert '{"questions": [{"question": "<the question>",' in drafted.messages()[1]["content"]


def test_source_ids_name_the_document_version_and_rank() -> None:
    assert source_id("refunds", "3", 1) == "refunds-v3-p1"
    assert source_id(None, None, 4) == "passage-p4"
    assert source_id("a b/c", "1.2", 2) == "a-b-c-v1.2-p2"
    assert render_sources([]) == "(none)"


@pytest.mark.parametrize(
    ("text", "refused"),
    [
        ("I can't approve that refund.", True),
        ("I\u2019m not able to change another barista's schedule.", True),  # a curly quote
        ("I won't share anyone's pay rate.", True),
        ("I'll have to decline that request.", True),
        ("That decision has to be made by your District Manager.", True),
        ("Changing pay is outside my role.", True),
        ("Refunds over $50 must be approved by the café manager.", False),
        ("Outside food is not allowed in the café.", False),
        ("I can't find a closing time in these passages.", False),
    ],
)
def test_refusals_are_first_person_declines_and_hand_offs(text: str, refused: bool) -> None:
    assert detect_refusal(text) is refused
    assert grade_refusal(text, refuse=True).passed is refused
    assert grade_refusal(text, refuse=False).passed is not refused


@pytest.mark.parametrize(
    "text",
    [
        NOT_IN_SOURCES,
        "The sources don't say what the fee is.",
        "That isn't in the sources.",
        "I couldn't find anything about overtime in the documents.",
        "I'm unable to find that in the provided sources.",
        "There's no information about holiday hours in the passages.",
        "I cannot decide that for you.",
    ],
)
def test_abstaining_is_declining_or_saying_the_sources_dont_answer(text: str) -> None:
    assert abstains(text)


@pytest.mark.parametrize(
    "text",
    [
        "The refund window is 14 days [refunds-v3-p1].",
        "The documents say baristas don't answer the phone during a rush [ops-v1-p2].",
    ],
)
def test_an_answer_with_content_doesnt_abstain(text: str) -> None:
    assert not abstains(text)


def test_the_other_graders() -> None:
    assert citations_in("Yes [a-v1-p1], and [b-v2-p3; c-v1-p1] [not a valid id!]") == [
        "a-v1-p1",
        "b-v2-p3",
        "c-v1-p1",
    ]
    assert grade_regex("It is $5,000.", "5,000").passed
    assert not grade_regex("It is $5,000.", r"\b120\b", must_match=True).passed
    assert grade_regex("It is $5,000.", r"\b120\b", must_match=False).passed
    assert grade_exact(" Yes ", "yes", ignore_case=True).passed
    assert not grade_exact("Yes", "yes").passed
    cited = grade_citation(
        "Up to $50 [refunds-v2] [refunds-v1]",
        ["refunds-v1", "refunds-v2"],
        required=["refunds-v2"],
        forbidden=["refunds-v1"],
    )
    assert (cited.score, cited.passed, cited.detail) == (0.75, False, "cited forbidden refunds-v1")
    unknown = grade_citation("See [refunds-v9]", ["refunds-v2"])
    assert unknown.detail == "1 citation(s) to unknown sources"
    assert grade_citation(NOT_IN_SOURCES, [], expect_none=True).passed
    assert grade_rubric(4).passed
    assert not grade_rubric(3).passed
    assert grade_rubric(None).score is None
    assert parse_judge_score('{"score": 4, "reason": "fine"}') == 4
    assert parse_judge_score('```json\n{"score": 2}\n```') == 2
    assert parse_judge_score("no score here") is None
    assert parse_judge_score('{"score": 9}') is None


def test_an_answer_is_scored_for_abstention_and_citations() -> None:
    expected = [("refunds", "3")]
    good = score_answer(
        item_id="q1",
        expected=expected,
        reference="A drink can be refunded for 14 days with the receipt.",
        answer="A drink can be refunded within 14 calendar days of purchase [refunds-v3-p1].",
        passages=PASSAGES,
        verdicts=Verdicts(faithfulness=5, correctness=4),
    )
    assert (good.abstained, good.abstention_correct) == (False, True)
    assert (good.citation_precision, good.citation_recall) == (1.0, 1.0)
    assert good.reference_f1 is not None
    assert good.reference_f1 > 0.5
    assert good.metrics()["faithfulness_pass"] == 1.0
    stale = score_answer(
        item_id="q1",
        expected=expected,
        reference=None,
        answer="30 days [refunds-v2-p2] [made-up-p9].",
        passages=PASSAGES,
    )
    assert stale.citation_precision == 0.0  # the old version, and a source it wasn't given
    assert (stale.unknown_citations, stale.citation_recall) == (1, 0.0)
    assert stale.metrics()["faithfulness_pass"] is None
    wrongly_silent = score_answer(
        item_id="q1", expected=expected, reference=None, answer=NOT_IN_SOURCES, passages=PASSAGES
    )
    assert (wrongly_silent.abstention_correct, wrongly_silent.citation_precision) == (False, None)
    rightly_silent = score_answer(
        item_id="q2", expected=[], reference=None, answer=NOT_IN_SOURCES, passages=PASSAGES
    )
    assert rightly_silent.abstention_correct
    assert rightly_silent.metrics()["answered"] is None  # only queries with evidence count
    summary = summarize([good, stale, wrongly_silent, rightly_silent], resamples=200)
    assert summary["counts"] == {"queries": 4, "with_evidence": 3, "abstained": 2, "judged": 1}
    assert summary["abstention_accuracy"]["mean"] == 0.75
    assert summary["answered"]["n"] == 3
    assert summary["correctness_pass"]["n"] == 1
    assert automatic_label(good.as_dict(), "faithful") is True
    assert automatic_label(stale.as_dict(), "faithful") is None
    assert automatic_label(rightly_silent.as_dict(), "abstained") is True
    assert token_f1("", "anything") == 0.0


def test_agreement_reports_the_share_and_cohens_kappa() -> None:
    perfect = agreement([(True, True), (False, False)] * 5, resamples=200)
    assert perfect.agreement is not None
    assert (perfect.n, perfect.agreement.mean, perfect.kappa) == (10, 1.0, 1.0)
    # Agreeing only as often as chance would: kappa 0.
    chance = agreement(
        [(True, True), (True, False), (False, True), (False, False)] * 5, resamples=200
    )
    assert chance.kappa == 0.0
    assert chance.agreement is not None
    assert chance.agreement.low < 0.5 < chance.agreement.high
    # Everyone always says yes: agreement, but nothing for kappa to measure.
    assert agreement([(True, True)] * 4).kappa is None
    assert agreement([]).as_dict()["agreement"] is None


def test_drafted_questions_keep_only_evidence_from_the_passage() -> None:
    passage = (
        "Receipts help.  A drink is refundable for 14 days.\nOrders over $5,525 need approval."
    )
    drafts = [
        ("How long is a drink refundable?", "A drink is refundable for 14 days."),
        ("How long is a drink refundable?", "Orders over $5,525 need approval."),  # a repeat
        ("What is the fee?", "The fee is $30."),  # not in the passage
        ("Short", "A drink is refundable for 14 days."),  # too short to be a question
        ("Which orders need approval?", "Orders over  $5,525 need approval."),  # spacing differs
    ]
    reply = json.dumps(
        {
            "questions": [
                {"question": question, "evidence": evidence} for question, evidence in drafts
            ]
        }
    )
    found = parse_questions(f"Here you go:\n{reply}", passage, limit=5)
    assert [(draft.question, draft.evidence) for draft in found] == [
        ("How long is a drink refundable?", "A drink is refundable for 14 days."),
        ("Which orders need approval?", "Orders over $5,525 need approval."),
    ]
    assert len(parse_questions(reply, passage, limit=1)) == 1
    assert parse_questions("not json", passage, limit=3) == []
    assert parse_questions('{"questions": "none"}', passage, limit=3) == []
