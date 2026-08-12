"""Shared fixtures and test doubles for the sherpai_schemas suite."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from sherpai_schemas import (
    ChangeRole,
    Decision,
    FieldChange,
    Finding,
    LifecycleStatus,
    LlmResponse,
    PipelineStage,
    PipelineTool,
    ProblemType,
    Prompts,
    Proposal,
    SherpAIInstance,
    ToolIdentity,
)

# The ERP columns the pipeline actually carries, mirroring the allow-list in
# functions.get_pure_data.
ERP_COLUMNS = {
    "hybrid": "PERS_1_42",
    "typ": "1",
    "nr": "4711",
    "klassifik": "10",
    "name1": "Gebauer GmbH",
    "zeile1": "Hauptstrasse 12",
    "plz": "70173",
    "ort": "Stuttgart",
    "land": "DE",
    "ustid": "DE123456789",
    "steuernr": "12/345/67890",
}


@pytest.fixture
def identity() -> ToolIdentity:
    return ToolIdentity(stage=PipelineStage.DETECTION, tool="misplaced", tier=1)


@pytest.fixture
def correction_identity() -> ToolIdentity:
    return ToolIdentity(stage=PipelineStage.CORRECTION, tool="misplaced", tier=1)


@pytest.fixture
def drafted_proposal(identity: ToolIdentity) -> Proposal:
    return Proposal(
        identity=identity,
        changes=[FieldChange(column="ort", value="Stuttgart", role=ChangeRole.TARGET)],
    )


def make_accepted_correction(
    identity: ToolIdentity,
    column: str = "ort",
    value: str = "Stuttgart",
    decided_at: datetime | None = None,
) -> Proposal:
    """A correction Proposal already walked through to ACCEPTED.

    `decided_at` is set explicitly rather than relying on the Decision default,
    so "latest accepted wins" assertions can't tie on a same-microsecond clock.
    """
    proposal = Proposal(
        identity=identity,
        changes=[FieldChange(column=column, value=value, role=ChangeRole.TARGET)],
    )
    proposal.mark_review_ready()
    proposal.accept(reviewer="tester")
    if decided_at is not None:
        proposal.decision = Decision(reviewer="tester", decided_at=decided_at)
    return proposal


@pytest.fixture
def accepted_correction(correction_identity: ToolIdentity) -> Proposal:
    return make_accepted_correction(correction_identity)


def make_finding(
    identity: ToolIdentity,
    problem_type: ProblemType = ProblemType.MISPLACED,
    detection_columns: tuple[str, ...] = ("ort",),
    correction: Proposal | None = None,
) -> Finding:
    detection = Proposal(
        identity=identity,
        changes=[FieldChange(column=col) for col in detection_columns],
    )
    detection.mark_review_ready()
    return Finding(problem_type=problem_type, detection=detection, correction=correction)


@pytest.fixture
def data_row() -> pd.Series:
    return pd.Series(ERP_COLUMNS)


@pytest.fixture
def utc_now() -> datetime:
    return datetime.now(UTC)


class FakeTool(PipelineTool):
    """Records every row it is handed so tests can assert on the runner's contract."""

    identity = ToolIdentity(stage=PipelineStage.DETECTION, tool="misplaced", tier=1)

    def __init__(self) -> None:
        self.seen_rows: list[pd.Series] = []

    def process_row(self, row: pd.Series, instance: SherpAIInstance) -> SherpAIInstance:
        self.seen_rows.append(row.copy())
        return instance


class BatchingFakeTool(FakeTool):
    """FakeTool that opts into the batching path by declaring a system prompt."""

    batch_system_prompt = Prompts.FIX_MISPLACED_SYSTEM
    batch_max_tokens = 16

    def __init__(self, prompt_columns: tuple[str, ...] = ("ort",)) -> None:
        super().__init__()
        self.prompt_columns = prompt_columns

    def process_row(self, row: pd.Series, instance: SherpAIInstance) -> SherpAIInstance:
        super().process_row(row, instance)
        proposal = Proposal(
            identity=self.identity,
            changes=[FieldChange(column=col) for col in self.prompt_columns],
        )
        proposal.mark_batching_ready(prompt=f"fix {row['ort']}")
        instance.add_finding(
            Finding(problem_type=ProblemType.MISPLACED, detection=proposal)
        )
        return instance


@pytest.fixture
def fake_tool() -> FakeTool:
    return FakeTool()


class RecordingInference:
    """Stand-in for llm_interface.inference_completion.

    Captures the call arguments and replays a caller-supplied result list, so
    tests never touch the network or a GPU.
    """

    def __init__(self, results: list[LlmResponse] | None = None) -> None:
        self.results = results if results is not None else []
        self.calls: list[dict] = []

    def __call__(self, *, model: str, prompt: list[str], max_tokens: int) -> list[LlmResponse]:
        self.calls.append({"model": model, "prompt": prompt, "max_tokens": max_tokens})
        return self.results


@pytest.fixture
def fake_inference_completion() -> RecordingInference:
    return RecordingInference()


def write_jsonl(path, rows: list[dict]) -> None:
    """Write pipeline-shaped input: one JSON object per line, SherpAISpace stringified."""
    frame = pd.DataFrame(rows)
    if "SherpAISpace" not in frame.columns:
        frame["SherpAISpace"] = str(SherpAIInstance())
    frame.to_json(path, lines=True, orient="records")


__all__ = [
    "ERP_COLUMNS",
    "BatchingFakeTool",
    "FakeTool",
    "LifecycleStatus",
    "RecordingInference",
    "make_accepted_correction",
    "make_finding",
    "write_jsonl",
]
