"""Tests for the vLLM HTTP wrappers.

These functions are unusual: they swallow transport and parsing errors and
return sentinel values instead of raising. That behavior is what every
images/*/main.py depends on, so it is pinned here. No network is touched --
requests.post is replaced wholesale.
"""

import json

import pandas as pd
import pytest
import requests

from sherpai_schemas import (
    LlmResponse,
    batch_vectorization,
    format_gemma_prompt,
    inference_completion,
    inference_conversation,
)

POST_TARGET = "sherpai_schemas.llm_interface.requests.post"


class StubResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload=None, text="", raises=None):
        self._payload = payload
        self.text = text
        self._raises = raises

    def raise_for_status(self):
        if self._raises is not None:
            raise self._raises

    def json(self):
        return self._payload


class RecordingPost:
    """Replacement for requests.post that records calls and replays a response."""

    def __init__(self, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if self.raises is not None:
            raise self.raises
        return self.response

    @property
    def payload(self) -> dict:
        """The JSON body of the first call, however it was passed."""
        call = self.calls[0]
        return call["json"] if "json" in call else json.loads(call["data"])


def chat_payload(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def completion_payload(*texts: str) -> dict:
    return {
        "choices": [
            {"index": i, "text": text} for i, text in enumerate(texts)
        ]
    }


def fixes_json(column: str = "ort", value: str = "Stuttgart") -> str:
    return json.dumps(
        {"fixes": [{"column": column, "corrected_value": value, "reason": "typo"}]}
    )


# --------------------------------------------------------------------------
# format_gemma_prompt -- pure
# --------------------------------------------------------------------------


def test_format_gemma_prompt_wraps_both_turns_and_opens_the_model_turn():
    prompt = format_gemma_prompt("be terse", "fix Stuttgar")

    assert prompt == (
        "<start_of_turn>system\nbe terse<end_of_turn>\n"
        "<start_of_turn>user\nfix Stuttgar<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


# --------------------------------------------------------------------------
# inference_conversation
# --------------------------------------------------------------------------


def test_inference_conversation_returns_the_message_content(monkeypatch):
    post = RecordingPost(StubResponse(payload=chat_payload("Stuttgart")))
    monkeypatch.setattr(POST_TARGET, post)

    result = inference_conversation("sys", "user", model="gemma")

    assert result == "Stuttgart"


def test_inference_conversation_posts_to_the_knowledgebase_by_default(monkeypatch):
    post = RecordingPost(StubResponse(payload=chat_payload("ok")))
    monkeypatch.setattr(POST_TARGET, post)

    inference_conversation("sys", "user", model="gemma")

    assert post.calls[0]["url"] == "http://knowledgebase:8000/v1/chat/completions"


def test_inference_conversation_builds_the_openai_message_list(monkeypatch):
    post = RecordingPost(StubResponse(payload=chat_payload("ok")))
    monkeypatch.setattr(POST_TARGET, post)

    inference_conversation("be terse", "fix this", model="gemma", temperature=0.7)

    payload = post.payload
    assert payload["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "fix this"},
    ]
    assert payload["model"] == "gemma"
    assert payload["temperature"] == 0.7
    assert payload["stream"] is False


def test_inference_conversation_omits_authorization_without_an_api_key(monkeypatch):
    post = RecordingPost(StubResponse(payload=chat_payload("ok")))
    monkeypatch.setattr(POST_TARGET, post)

    inference_conversation("sys", "user", model="gemma")

    assert "Authorization" not in post.calls[0]["headers"]


def test_inference_conversation_sends_a_bearer_token_when_given_one(monkeypatch):
    post = RecordingPost(StubResponse(payload=chat_payload("ok")))
    monkeypatch.setattr(POST_TARGET, post)

    inference_conversation("sys", "user", model="gemma", api_key="secret")

    assert post.calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_inference_conversation_returns_an_error_string_on_transport_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        POST_TARGET,
        RecordingPost(raises=requests.exceptions.ConnectionError("refused")),
    )

    result = inference_conversation("sys", "user", model="gemma")

    # Callers get a string, not an exception -- smart_cast then falls back.
    assert result.startswith("HTTP Request Error:")
    assert "refused" in result


def test_inference_conversation_returns_an_error_string_on_a_bad_status(monkeypatch):
    monkeypatch.setattr(
        POST_TARGET,
        RecordingPost(StubResponse(raises=requests.exceptions.HTTPError("503"))),
    )

    assert inference_conversation("sys", "user", model="gemma").startswith(
        "HTTP Request Error:"
    )


def test_inference_conversation_reports_an_unexpected_body_shape(monkeypatch):
    monkeypatch.setattr(
        POST_TARGET,
        RecordingPost(StubResponse(payload={}, text="<html>gateway timeout</html>")),
    )

    result = inference_conversation("sys", "user", model="gemma")

    assert result.startswith("Unexpected API Response Format:")
    assert "gateway timeout" in result


# --------------------------------------------------------------------------
# inference_completion
# --------------------------------------------------------------------------


def test_inference_completion_parses_each_choice(monkeypatch):
    monkeypatch.setattr(
        POST_TARGET,
        RecordingPost(StubResponse(payload=completion_payload(fixes_json()))),
    )

    results = inference_completion(prompt=["p"], model="gemma")

    assert len(results) == 1
    assert results[0].fixes[0].column == "ort"
    assert results[0].fixes[0].corrected_value == "Stuttgart"


def test_inference_completion_restores_the_prompt_order_from_choice_index(monkeypatch):
    # vLLM may return choices out of order; the index is authoritative.
    payload = {
        "choices": [
            {"index": 1, "text": fixes_json("plz", "70173")},
            {"index": 0, "text": fixes_json("ort", "Stuttgart")},
        ]
    }
    monkeypatch.setattr(POST_TARGET, RecordingPost(StubResponse(payload=payload)))

    results = inference_completion(prompt=["a", "b"], model="gemma")

    assert [r.fixes[0].column for r in results] == ["ort", "plz"]


def test_inference_completion_declares_the_llm_response_json_schema(monkeypatch):
    post = RecordingPost(StubResponse(payload=completion_payload(fixes_json())))
    monkeypatch.setattr(POST_TARGET, post)

    inference_completion(prompt=["p"], model="gemma", max_tokens=99)

    payload = post.payload
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["name"] == "llm_response"
    assert payload["response_format"]["json_schema"]["schema"] == (
        LlmResponse.model_json_schema()
    )
    assert payload["max_tokens"] == 99


def test_inference_completion_sends_a_bearer_token_when_given_one(monkeypatch):
    post = RecordingPost(StubResponse(payload=completion_payload(fixes_json())))
    monkeypatch.setattr(POST_TARGET, post)

    inference_completion(prompt=["p"], model="gemma", api_key="secret")

    assert post.calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_inference_completion_substitutes_a_sentinel_for_unparsable_output(monkeypatch):
    monkeypatch.setattr(
        POST_TARGET,
        RecordingPost(StubResponse(payload=completion_payload("this is not json"))),
    )

    results = inference_completion(prompt=["p"], model="gemma")

    assert results[0].fixes[0].column == ""
    assert results[0].fixes[0].reason.startswith("Failed to parse model output:")


def test_inference_completion_keeps_good_choices_alongside_bad_ones(monkeypatch):
    monkeypatch.setattr(
        POST_TARGET,
        RecordingPost(
            StubResponse(payload=completion_payload(fixes_json(), "garbage"))
        ),
    )

    results = inference_completion(prompt=["a", "b"], model="gemma")

    assert results[0].fixes[0].column == "ort"
    assert results[1].fixes[0].reason.startswith("Failed to parse model output:")


def test_inference_completion_returns_one_sentinel_per_prompt_on_transport_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        POST_TARGET,
        RecordingPost(raises=requests.exceptions.ConnectionError("refused")),
    )

    results = inference_completion(prompt=["a", "b", "c"], model="gemma")

    # The count must match, or PipelineRunner._flush_batches raises.
    assert len(results) == 3
    assert all(r.fixes[0].reason.startswith("HTTP Request Error:") for r in results)


def test_inference_completion_returns_a_single_sentinel_for_a_string_prompt(
    monkeypatch,
):
    monkeypatch.setattr(
        POST_TARGET,
        RecordingPost(raises=requests.exceptions.ConnectionError("refused")),
    )

    assert len(inference_completion(prompt="just one", model="gemma")) == 1


def test_inference_completion_reports_an_unexpected_body_shape(monkeypatch):
    monkeypatch.setattr(
        POST_TARGET,
        RecordingPost(StubResponse(payload={}, text="<html>502</html>")),
    )

    results = inference_completion(prompt=["a", "b"], model="gemma")

    assert len(results) == 2
    assert all(
        r.fixes[0].reason.startswith("Unexpected API Response Format:") for r in results
    )


# --- Characterization test: current behavior, not desired behavior ---------


def test_error_sentinels_are_the_same_aliased_object(monkeypatch):
    # FIXME: the error paths build `[sentinel] * n_prompts`, which repeats one
    # object rather than creating n. PipelineRunner.apply_batch_result mutates
    # the LlmResponse it is handed, so a mutation to one result would be seen
    # by all. Building a fresh LlmResponse per prompt would fix it.
    monkeypatch.setattr(
        POST_TARGET,
        RecordingPost(raises=requests.exceptions.ConnectionError("refused")),
    )

    results = inference_completion(prompt=["a", "b"], model="gemma")

    assert results[0] is results[1]


# --------------------------------------------------------------------------
# batch_vectorization
# --------------------------------------------------------------------------


def embedding_payload(*vectors: list[float]) -> dict:
    return {"data": [{"embedding": vector} for vector in vectors]}


def test_batch_vectorization_returns_one_embedding_per_input(monkeypatch):
    monkeypatch.setattr(
        POST_TARGET,
        RecordingPost(StubResponse(payload=embedding_payload([0.1, 0.2], [0.3, 0.4]))),
    )

    embeddings = batch_vectorization(pd.Series(["a", "b"]))

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_batch_vectorization_posts_to_the_embedbase_by_default(monkeypatch):
    post = RecordingPost(StubResponse(payload=embedding_payload([0.1])))
    monkeypatch.setattr(POST_TARGET, post)

    batch_vectorization(pd.Series(["a"]))

    assert post.calls[0]["url"] == "http://embedbase:8000/v1/embeddings"
    assert post.calls[0]["json"]["input"] == ["a"]
    assert post.calls[0]["timeout"] == 60


def test_batch_vectorization_sends_a_bearer_token_when_given_one(monkeypatch):
    post = RecordingPost(StubResponse(payload=embedding_payload([0.1])))
    monkeypatch.setattr(POST_TARGET, post)

    batch_vectorization(pd.Series(["a"]), api_key="secret")

    assert post.calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_batch_vectorization_splits_into_batches(monkeypatch):
    post = RecordingPost(StubResponse(payload=embedding_payload([0.1], [0.2])))
    monkeypatch.setattr(POST_TARGET, post)

    batch_vectorization(pd.Series(["a", "b", "c", "d", "e"]), batch_size=2)

    # 5 rows at batch_size 2 -> 3 requests, the last one partial.
    assert len(post.calls) == 3
    assert [len(call["json"]["input"]) for call in post.calls] == [2, 2, 1]


def test_batch_vectorization_makes_no_request_for_empty_input(monkeypatch):
    post = RecordingPost(StubResponse(payload=embedding_payload()))
    monkeypatch.setattr(POST_TARGET, post)

    assert batch_vectorization(pd.Series([], dtype=str)) == []
    assert post.calls == []


def test_batch_vectorization_propagates_transport_errors(monkeypatch):
    # Unlike the inference helpers, this one has no try/except -- it raises.
    monkeypatch.setattr(
        POST_TARGET,
        RecordingPost(raises=requests.exceptions.ConnectionError("refused")),
    )

    with pytest.raises(requests.exceptions.ConnectionError):
        batch_vectorization(pd.Series(["a"]))
