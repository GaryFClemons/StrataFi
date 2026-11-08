"""Generators: the lab's fake answers, judges and drafts questions without a model;
``generate/gateway`` speaks the gateway's chat completions."""

import json

import pytest
from fastapi.testclient import TestClient

from stratafi_components import default_registry
from stratafi_components.generate import FakeGenerate, GatewayGenerate
from stratafi_components.models import GatewayProvider
from stratafi_core.answers import (
    NOT_IN_SOURCES,
    ContextPassage,
    answer_request,
    correctness_request,
    draft_request,
    faithfulness_request,
    parse_judge_score,
    parse_questions,
)
from stratafi_core.components import ComponentError, StageContext
from stratafi_testing.fake_gateway import FakeGateway, create_app

TEXT = (
    "Funds from a check deposited to a new account are available on the ninth business day. "
    "An account is a new account during the first 30 calendar days after it is opened."
)
PASSAGES = [
    ContextPassage("funds-availability-v3-p1", "funds-availability", "3", "5.1", TEXT, True),
    ContextPassage("wires-v1-p2", "wires", "1", "2", "Wires need a callback.", False),
]


def fake(request, **params):  # type: ignore[no-untyped-def]
    component = FakeGenerate()
    return component.generate(request, component.resolve_params(params), StageContext())


def test_the_generators_are_query_time_components() -> None:
    registry = default_registry()
    assert registry.get("generate/fake").info.lab_only
    gateway = registry.get("generate/gateway").info
    assert (gateway.stage, gateway.egress, gateway.deterministic) == (
        "generate",
        "model_gateway",
        False,
    )


def test_the_fake_answers_from_a_cited_sentence_or_abstains() -> None:
    answered = fake(
        answer_request(
            "How many calendar days is an account new after it is opened?", PASSAGES, day="x"
        )
    )
    assert answered.text == (
        "An account is a new account during the first 30 calendar days after it is opened. "
        "[funds-availability-v3-p1]"
    )
    assert answered.revision == "fake-generate-v1"
    assert answered.prompt_tokens > 0
    silent = fake(answer_request("What fee applies to a returned check?", PASSAGES, day="x"))
    assert silent.text == NOT_IN_SOURCES
    assert fake(answer_request("Anything?", [], day="x")).text == NOT_IN_SOURCES


def test_the_fake_judges_by_word_overlap() -> None:
    quoted = f"{TEXT.split('. ')[1]} [funds-availability-v3-p1]"
    judged = fake(faithfulness_request("q", PASSAGES, quoted))
    assert parse_judge_score(judged.text) == 5
    invented = "Overdraft fees are waived for customers over 65 [wires-v1-p2]."
    assert parse_judge_score(fake(faithfulness_request("q", PASSAGES, invented)).text) == 1
    assert parse_judge_score(fake(faithfulness_request("q", PASSAGES, NOT_IN_SOURCES)).text) == 5
    reference = "A new account is one in its first 30 calendar days."
    close = fake(correctness_request("q", reference, quoted))
    far = fake(correctness_request("q", reference, "Wires need a callback."))
    assert parse_judge_score(close.text) > parse_judge_score(far.text)  # type: ignore[operator]


def test_the_fake_drafts_questions_with_evidence_from_the_passage() -> None:
    reply = fake(draft_request("funds-availability", TEXT, count=2))
    drafted = parse_questions(reply.text, TEXT, limit=2)
    assert len(drafted) == 2
    assert drafted[0].evidence.startswith("An account is a new account")  # numbers first
    assert all(draft.question.startswith("What does funds-availability say") for draft in drafted)


def provider(gateway: FakeGateway) -> StageContext:
    client = TestClient(create_app(gateway), base_url="http://gateway")
    models = GatewayProvider("http://gateway", client=client, sleep=lambda seconds: None)
    return StageContext(services={"models": models})


def test_the_gateway_generator_sends_the_prompt_by_alias() -> None:
    gateway = FakeGateway(chat_reply=json.dumps({"score": 4}))
    context = provider(gateway)
    component = GatewayGenerate()
    params = component.resolve_params({"alias": "chat-default"})
    revision = component.revision_for(params, context)
    assert revision == gateway.aliases["chat-default"].revision
    result = component.generate(faithfulness_request("q", PASSAGES, "a"), params, context)
    assert parse_judge_score(result.text) == 4
    assert (result.revision, result.completion_tokens) == (revision, 2)  # the fake counts words
    assert result.prompt_tokens > 0
    assert [(call.endpoint, call.alias, call.inputs) for call in gateway.records][-1] == (
        "chat",
        "chat-default",
        2,
    )
    assert context.metrics["gateway.calls"] == 1


def test_the_gateway_refuses_chat_to_service_tokens() -> None:
    context = provider(FakeGateway(refuse_chat=True))
    component = GatewayGenerate()
    with pytest.raises(ComponentError) as refused:
        component.generate(
            answer_request("q", PASSAGES, day="x"), component.resolve_params({}), context
        )
    assert refused.value.code == "gateway_refused_403"
    assert not refused.value.retryable
