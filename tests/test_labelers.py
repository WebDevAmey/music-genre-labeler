import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from labeler.labelers import LLMLabeler, KeywordLabeler, LabelingError, parse_response
from labeler.task import LABELS


def response(payload, stop_reason="end_turn"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    """Records the request and returns a canned response (or raises)."""

    def __init__(self, result):
        self.result, self.calls = result, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_parses_valid_response():
    s = parse_response(response({"label": "rock", "confidence": 0.8, "rationale": " distorted guitar "}))
    assert s == ("rock", 0.8, "distorted guitar")


@pytest.mark.parametrize("bad", [
    response({"label": "polka", "confidence": 0.8, "rationale": "x"}),
    response({"label": "rock", "confidence": 1.7, "rationale": "x"}),
    response({"label": "rock"}),
    response("not json"),
    response({"label": "rock", "confidence": 0.5, "rationale": "x"}, stop_reason="refusal"),
    response({"label": "rock", "confidence": 0.5, "rationale": "x"}, stop_reason="max_tokens"),
    SimpleNamespace(stop_reason="end_turn", content=[]),
])
def test_rejects_bad_responses(bad):
    with pytest.raises(LabelingError):
        parse_response(bad)


def test_request_shape():
    client = FakeClient(response({"label": "jazz_blues", "confidence": 0.9, "rationale": "sax"}))
    labeler = LLMLabeler(client=client, model="claude-opus-5-5")
    assert labeler.suggest("smooth saxophone").label == "jazz_blues"

    req = client.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"]["format"]["schema"]["properties"]["label"]["enum"] == list(LABELS)
    assert "<caption>\nsmooth saxophone\n</caption>" == req["messages"][0]["content"]
    assert req["fallbacks"] == "default"
    assert "temperature" not in req


def test_no_fallback_for_models_without_it():
    client = FakeClient(response({"label": "pop", "confidence": 0.5, "rationale": "x"}))
    LLMLabeler(client=client, model="claude-haiku-5-5").suggest("x")
    assert "fallbacks" not in client.calls[0] and "betas" not in client.calls[0]


def _api_error(cls, status):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=req), body=None)


def test_transient_api_errors_become_labeling_errors():
    labeler = LLMLabeler(client=FakeClient(_api_error(anthropic.RateLimitError, 429)))
    with pytest.raises(LabelingError):
        labeler.suggest("x")


def test_auth_errors_propagate():
    labeler = LLMLabeler(client=FakeClient(_api_error(anthropic.AuthenticationError, 401)))
    with pytest.raises(anthropic.AuthenticationError):
        labeler.suggest("x")


def test_keyword_baseline():
    k = KeywordLabeler()
    s = k.suggest("A heavy metal track with distorted guitar and pounding drums.")
    assert s.label == "rock" and 0 < s.confidence <= 1
    assert k.suggest("A dog barks twice.").label == "other"
    # Whole words only: "rapid" is not "rap".
    assert k.suggest("a rapid drum fill").label == "other"
