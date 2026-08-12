"""Tests for the Template Method skeleton: PipelineTool and PipelineRunner."""

import pandas as pd
import pytest
from conftest import (
    ERP_COLUMNS,
    BatchingFakeTool,
    FakeTool,
    RecordingInference,
    make_accepted_correction,
    make_finding,
    write_jsonl,
)

from sherpai_schemas import (
    FieldChange,
    Finding,
    Fix,
    LifecycleStatus,
    LlmResponse,
    PipelineRunner,
    PipelineStage,
    ProblemType,
    Proposal,
    SherpAIInstance,
    ToolIdentity,
)

# The seam: pipeline.py does `from .llm_interface import inference_completion`,
# so the bound name lives on the pipeline module, not on llm_interface.
INFERENCE_TARGET = "sherpai_schemas.pipeline.inference_completion"


@pytest.fixture
def batching_proposal(identity) -> Proposal:
    proposal = Proposal(
        identity=identity, changes=[FieldChange(column="ort", value="Stuttgar")]
    )
    proposal.mark_batching_ready("fix Stuttgar")
    return proposal


# --------------------------------------------------------------------------
# PipelineTool.apply_batch_result -- pure, no runner needed
# --------------------------------------------------------------------------


def test_apply_batch_result_updates_an_existing_change_in_place(
    fake_tool, batching_proposal
):
    response = LlmResponse(
        fixes=[Fix(column="ort", corrected_value="Stuttgart", reason="typo")]
    )

    fake_tool.apply_batch_result(batching_proposal, response)

    assert len(batching_proposal.changes) == 1
    assert batching_proposal.changes[0].value == "Stuttgart"


def test_apply_batch_result_appends_a_change_for_an_unknown_column(
    fake_tool, batching_proposal
):
    response = LlmResponse(
        fixes=[Fix(column="plz", corrected_value="70173", reason="derived")]
    )

    fake_tool.apply_batch_result(batching_proposal, response)

    assert [c.column for c in batching_proposal.changes] == ["ort", "plz"]
    assert batching_proposal.changes[-1].value == "70173"


def test_apply_batch_result_takes_the_reason_from_the_last_fix(
    fake_tool, batching_proposal
):
    response = LlmResponse(
        fixes=[
            Fix(column="ort", corrected_value="Stuttgart", reason="first"),
            Fix(column="plz", corrected_value="70173", reason="second"),
        ]
    )

    fake_tool.apply_batch_result(batching_proposal, response)

    assert batching_proposal.reason == "second"


def test_apply_batch_result_moves_the_proposal_to_review_ready(
    fake_tool, batching_proposal
):
    fake_tool.apply_batch_result(batching_proposal, LlmResponse(fixes=[]))

    assert batching_proposal.status == LifecycleStatus.REVIEW_READY
    assert batching_proposal.pending_prompt is None


def test_apply_batch_result_on_an_empty_response_leaves_changes_untouched(
    fake_tool, batching_proposal
):
    fake_tool.apply_batch_result(batching_proposal, LlmResponse(fixes=[]))

    assert [c.value for c in batching_proposal.changes] == ["Stuttgar"]


# --------------------------------------------------------------------------
# PipelineRunner -- read / process / write over tmp_path
# --------------------------------------------------------------------------


def test_run_round_trips_a_jsonl_file(tmp_path, fake_tool):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    write_jsonl(input_path, [dict(ERP_COLUMNS)])

    PipelineRunner(fake_tool, input_path, output_path).run()

    written = pd.read_json(output_path, lines=True)
    assert len(written) == 1
    assert written.loc[0, "ort"] == "Stuttgart"
    # SherpAISpace comes back out as a string, ready for the next container.
    assert isinstance(written.loc[0, "SherpAISpace"], str)
    assert SherpAIInstance.parse_from_str(written.loc[0, "SherpAISpace"]).findings == []


def test_run_preserves_findings_added_by_the_tool(tmp_path, identity):
    class DetectingTool(FakeTool):
        def process_row(self, row, instance):
            super().process_row(row, instance)
            instance.add_finding(
                make_finding(identity, problem_type=ProblemType.FORMATTING)
            )
            return instance

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    write_jsonl(input_path, [dict(ERP_COLUMNS)])

    PipelineRunner(DetectingTool(), input_path, output_path).run()

    written = pd.read_json(output_path, lines=True)
    restored = SherpAIInstance.parse_from_str(written.loc[0, "SherpAISpace"])
    assert [f.problem_type for f in restored.findings] == [ProblemType.FORMATTING]


def test_process_rows_applies_accepted_corrections_before_the_tool_sees_them(
    tmp_path, identity, utc_now
):
    """The contract every images/*/main.py relies on: the row handed to
    process_row already carries corrections accepted by earlier tools."""
    correction = make_accepted_correction(
        identity, column="ort", value="Karlsruhe", decided_at=utc_now
    )
    instance = SherpAIInstance(findings=[make_finding(identity, correction=correction)])

    row = dict(ERP_COLUMNS)  # ort == "Stuttgart" on disk
    row["SherpAISpace"] = str(instance)
    input_path = tmp_path / "input.jsonl"
    write_jsonl(input_path, [row])

    tool = FakeTool()
    PipelineRunner(tool, input_path, tmp_path / "output.jsonl").run()

    assert tool.seen_rows[0]["ort"] == "Karlsruhe"


def test_run_processes_every_row(tmp_path, fake_tool):
    rows = [dict(ERP_COLUMNS, nr=str(i)) for i in range(3)]
    input_path = tmp_path / "input.jsonl"
    write_jsonl(input_path, rows)

    PipelineRunner(fake_tool, input_path, tmp_path / "output.jsonl").run()

    assert len(fake_tool.seen_rows) == 3


def test_runner_defaults_to_the_container_job_paths(fake_tool):
    runner = PipelineRunner(fake_tool)

    assert str(runner.input_path) == "/job/input.jsonl"
    assert str(runner.output_path) == "/job/output.jsonl"


# --------------------------------------------------------------------------
# PipelineRunner._flush_batches
# --------------------------------------------------------------------------


def _frame_with(instances: list[SherpAIInstance]) -> pd.DataFrame:
    return pd.DataFrame({"SherpAISpace": instances})


def test_flush_batches_is_skipped_when_the_tool_declares_no_batch_prompt(
    monkeypatch, fake_tool, batching_proposal, identity
):
    fake = RecordingInference()
    monkeypatch.setattr(INFERENCE_TARGET, fake)
    instance = SherpAIInstance(
        findings=[
            Finding(problem_type=ProblemType.MISPLACED, detection=batching_proposal)
        ]
    )

    PipelineRunner(fake_tool)._flush_batches(_frame_with([instance]))

    assert fake.calls == []
    assert batching_proposal.status == LifecycleStatus.BATCHING_READY


def test_flush_batches_makes_no_call_when_nothing_is_pending(monkeypatch, identity):
    fake = RecordingInference()
    monkeypatch.setattr(INFERENCE_TARGET, fake)
    instance = SherpAIInstance(findings=[make_finding(identity)])  # REVIEW_READY

    PipelineRunner(BatchingFakeTool())._flush_batches(_frame_with([instance]))

    assert fake.calls == []


def test_flush_batches_collects_proposals_from_detection_and_correction(
    monkeypatch, identity
):
    detection = Proposal(identity=identity, changes=[FieldChange(column="ort")])
    detection.mark_batching_ready("detect this")
    correction = Proposal(identity=identity, changes=[FieldChange(column="plz")])
    correction.mark_batching_ready("correct this")
    instance = SherpAIInstance(
        findings=[
            Finding(
                problem_type=ProblemType.MISPLACED,
                detection=detection,
                correction=correction,
            )
        ]
    )
    fake = RecordingInference(
        results=[
            LlmResponse(fixes=[Fix(column="ort", corrected_value="A", reason="r1")]),
            LlmResponse(fixes=[Fix(column="plz", corrected_value="B", reason="r2")]),
        ]
    )
    monkeypatch.setattr(INFERENCE_TARGET, fake)

    PipelineRunner(BatchingFakeTool())._flush_batches(_frame_with([instance]))

    assert len(fake.calls[0]["prompt"]) == 2
    assert detection.changes[0].value == "A"
    assert correction.changes[0].value == "B"
    assert detection.status == LifecycleStatus.REVIEW_READY
    assert correction.status == LifecycleStatus.REVIEW_READY


def test_flush_batches_forwards_the_tools_model_and_token_budget(
    monkeypatch, batching_proposal
):
    fake = RecordingInference(results=[LlmResponse(fixes=[])])
    monkeypatch.setattr(INFERENCE_TARGET, fake)
    instance = SherpAIInstance(
        findings=[
            Finding(problem_type=ProblemType.MISPLACED, detection=batching_proposal)
        ]
    )
    tool = BatchingFakeTool()

    PipelineRunner(tool)._flush_batches(_frame_with([instance]))

    assert fake.calls[0]["model"] == tool.batch_model
    assert fake.calls[0]["max_tokens"] == tool.batch_max_tokens
    # The pending prompt is wrapped in the gemma turn format.
    assert "fix Stuttgar" in fake.calls[0]["prompt"][0]
    assert fake.calls[0]["prompt"][0].startswith("<start_of_turn>system")


def test_flush_batches_raises_when_result_count_does_not_match(
    monkeypatch, batching_proposal
):
    monkeypatch.setattr(INFERENCE_TARGET, RecordingInference(results=[]))
    instance = SherpAIInstance(
        findings=[
            Finding(problem_type=ProblemType.MISPLACED, detection=batching_proposal)
        ]
    )

    with pytest.raises(ValueError, match="Mismatch between number of prompts"):
        PipelineRunner(BatchingFakeTool())._flush_batches(_frame_with([instance]))


def test_flush_batches_ignores_non_instance_cells(monkeypatch, batching_proposal):
    fake = RecordingInference(results=[LlmResponse(fixes=[])])
    monkeypatch.setattr(INFERENCE_TARGET, fake)
    instance = SherpAIInstance(
        findings=[
            Finding(problem_type=ProblemType.MISPLACED, detection=batching_proposal)
        ]
    )
    frame = _frame_with([instance, None])

    PipelineRunner(BatchingFakeTool())._flush_batches(frame)

    assert len(fake.calls[0]["prompt"]) == 1


def test_run_drives_the_full_batching_path(tmp_path, monkeypatch):
    fake = RecordingInference(
        results=[
            LlmResponse(
                fixes=[Fix(column="ort", corrected_value="Stuttgart", reason="typo")]
            )
        ]
    )
    monkeypatch.setattr(INFERENCE_TARGET, fake)
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    write_jsonl(input_path, [dict(ERP_COLUMNS)])

    PipelineRunner(BatchingFakeTool(), input_path, output_path).run()

    written = pd.read_json(output_path, lines=True)
    restored = SherpAIInstance.parse_from_str(written.loc[0, "SherpAISpace"])
    proposal = restored.findings[0].detection
    assert proposal.status == LifecycleStatus.REVIEW_READY
    assert proposal.changes[0].value == "Stuttgart"
    assert proposal.reason == "typo"


# --------------------------------------------------------------------------
# PipelineTool contract
# --------------------------------------------------------------------------


def test_pipeline_tool_cannot_be_instantiated_without_process_row():
    from sherpai_schemas import PipelineTool

    class Incomplete(PipelineTool):
        identity = ToolIdentity(stage=PipelineStage.DETECTION, tool="missing")

    with pytest.raises(TypeError):
        Incomplete()
