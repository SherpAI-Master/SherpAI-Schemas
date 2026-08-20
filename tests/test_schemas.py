"""Unit tests for the schema/pipeline internals -- the logic llm-orchestration's
integration-style suite only reaches indirectly through tool `process_row` calls.
"""

import pandas as pd
import pytest

from sherpai_schemas import (
    ChangeRole,
    FieldChange,
    Finding,
    FormattingRules,
    LifecycleStatus,
    PipelineStage,
    Prompts,
    ProblemType,
    Proposal,
    SherpAIInstance,
    ToolIdentity,
)
from sherpai_schemas.functions import get_pure_data, parse_dimensions_from_str, parse_dimensions_to_str
from sherpai_schemas.pipeline import PipelineTool


# --------------------------------------------------------------------------
# ToolIdentity
# --------------------------------------------------------------------------


def test_compose_name_uses_stage_tool_and_tier():
    identity = ToolIdentity(stage=PipelineStage.DETECTION, tool="formatting", tier=1)

    assert identity.compose_name() == "detection_formatting_tier1.yml"


def test_compose_name_accepts_a_raw_stage_string():
    identity = ToolIdentity(stage="detection", tool="formatting", tier=1)

    assert identity.compose_name() == "detection_formatting_tier1.yml"


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("formatting", ProblemType.FORMATTING),
        ("incomplete", ProblemType.INCOMPLETE),
        ("misplaced", ProblemType.MISPLACED),
        ("missing", ProblemType.MISSING_VALUE),
        ("misspelled", ProblemType.MISSPELLED),
        ("validation", ProblemType.VALIDATION),
    ],
)
def test_as_problem_type_maps_every_real_detection_tool(tool, expected):
    identity = ToolIdentity(stage=PipelineStage.DETECTION, tool=tool, tier=1)

    assert identity.as_problem_type() == expected


def test_as_problem_type_is_none_for_a_tool_spanning_two_problem_types():
    identity = ToolIdentity(stage=PipelineStage.CORRECTION, tool="validation_missing", tier=1)

    assert identity.as_problem_type() is None


# --------------------------------------------------------------------------
# FieldChange / Proposal.single()
# --------------------------------------------------------------------------


def test_field_change_defaults_to_target_role():
    change = FieldChange(column="plz", value="70173")

    assert change.role == ChangeRole.TARGET


def _identity() -> ToolIdentity:
    return ToolIdentity(stage=PipelineStage.DETECTION, tool="formatting", tier=1)


def test_single_returns_the_only_change_with_no_role_given():
    proposal = Proposal(identity=_identity(), changes=[FieldChange(column="plz", value="7017")])

    assert proposal.single().column == "plz"


def test_single_filters_by_explicit_role():
    proposal = Proposal(
        identity=_identity(),
        changes=[
            FieldChange(column="ort", role=ChangeRole.TARGET),
            FieldChange(column="zeile1", role=ChangeRole.SOURCE),
        ],
    )

    assert proposal.single(ChangeRole.SOURCE).column == "zeile1"
    assert proposal.single(ChangeRole.TARGET).column == "ort"


def test_single_raises_with_no_matches():
    proposal = Proposal(identity=_identity(), changes=[FieldChange(column="ort", role=ChangeRole.SOURCE)])

    with pytest.raises(ValueError, match="Expected exactly one target change"):
        proposal.single()


def test_single_raises_with_more_than_one_match():
    proposal = Proposal(
        identity=_identity(),
        changes=[
            FieldChange(column="ort", role=ChangeRole.TARGET),
            FieldChange(column="plz", role=ChangeRole.TARGET),
        ],
    )

    with pytest.raises(ValueError, match="Expected exactly one target change"):
        proposal.single(ChangeRole.TARGET)


def test_single_error_names_the_requested_role():
    proposal = Proposal(identity=_identity(), changes=[])

    with pytest.raises(ValueError, match="Expected exactly one source change"):
        proposal.single(ChangeRole.SOURCE)


# --------------------------------------------------------------------------
# Proposal lifecycle
# --------------------------------------------------------------------------


def test_proposal_defaults_to_pending():
    proposal = Proposal(identity=_identity())

    assert proposal.status == LifecycleStatus.PENDING


def test_mark_review_ready_sets_status():
    proposal = Proposal(identity=_identity())

    proposal.mark_review_ready()

    assert proposal.status == LifecycleStatus.REVIEW_READY
    assert proposal.status.value == "review_ready"


def test_mark_batching_ready_sets_status_and_prompt():
    proposal = Proposal(identity=_identity())

    proposal.mark_batching_ready("fix this value")

    assert proposal.status == LifecycleStatus.BATCHING_READY
    assert proposal.pending_prompt == "fix this value"


def test_accept_only_mutates_the_instance_it_is_called_on():
    original = Proposal(identity=_identity(), changes=[FieldChange(column="plz", value="7017")])
    original.mark_review_ready()
    copy = original.model_copy(deep=True)

    assert copy == original

    original.accept(reviewer="tester")

    assert original.status == LifecycleStatus.ACCEPTED
    assert original.reviewed_by == "tester"
    assert copy.status == LifecycleStatus.REVIEW_READY
    assert original != copy


# --------------------------------------------------------------------------
# Finding
# --------------------------------------------------------------------------


def test_finding_correction_is_mutable_after_construction():
    detection = Proposal(identity=_identity(), changes=[FieldChange(column="plz", value="7017")])
    finding = Finding(problem_type=ProblemType.FORMATTING, detection=detection)

    assert finding.correction is None

    correction = Proposal(identity=_identity())
    finding.correction = correction

    assert finding.correction is correction


# --------------------------------------------------------------------------
# SherpAIInstance
# --------------------------------------------------------------------------


def _finding(problem_type: ProblemType, column: str) -> Finding:
    detection = Proposal(identity=_identity(), changes=[FieldChange(column=column, value="x")])
    detection.mark_review_ready()
    return Finding(problem_type=problem_type, detection=detection)


def test_add_finding_accumulates():
    instance = SherpAIInstance()

    instance.add_finding(_finding(ProblemType.FORMATTING, "plz"))
    instance.add_finding(_finding(ProblemType.MISSING_VALUE, "ustid"))

    assert len(instance.findings) == 2


def test_by_type_filters_and_returns_a_plain_list():
    instance = SherpAIInstance()
    instance.add_finding(_finding(ProblemType.FORMATTING, "plz"))
    instance.add_finding(_finding(ProblemType.MISSING_VALUE, "ustid"))

    formatting = instance.by_type(ProblemType.FORMATTING)
    missing = instance.by_type(ProblemType.MISSING_VALUE)

    assert [f.detection.single().column for f in formatting] == ["plz"]
    assert (formatting + missing) == instance.findings


def test_get_affected_cols_dedupes_across_problem_types():
    instance = SherpAIInstance()
    instance.add_finding(_finding(ProblemType.FORMATTING, "plz"))
    instance.add_finding(_finding(ProblemType.MISSING_VALUE, "plz"))
    instance.add_finding(_finding(ProblemType.MISSING_VALUE, "ustid"))

    cols = instance.get_affected_cols(ProblemType.FORMATTING, ProblemType.MISSING_VALUE)

    assert sorted(cols) == ["plz", "ustid"]


def test_get_affected_cols_ignores_unrequested_problem_types():
    instance = SherpAIInstance()
    instance.add_finding(_finding(ProblemType.MISPLACED, "ort"))

    assert instance.get_affected_cols(ProblemType.FORMATTING) == []


def test_sherpai_instance_has_no_is_empty_method():
    assert not hasattr(SherpAIInstance, "is_empty")


def test_sherpai_instance_round_trips_through_a_dataframe_column():
    instance = SherpAIInstance()
    instance.add_finding(_finding(ProblemType.FORMATTING, "plz"))
    df = pd.DataFrame({"plz": ["7017"], "SherpAISpace": [instance]})

    df = parse_dimensions_to_str(df)
    assert isinstance(df["SherpAISpace"].iloc[0], str)

    df = parse_dimensions_from_str(df)
    restored = df["SherpAISpace"].iloc[0]
    assert isinstance(restored, SherpAIInstance)
    assert restored.by_type(ProblemType.FORMATTING)[0].detection.single().column == "plz"


# --------------------------------------------------------------------------
# functions.py regression guards
# --------------------------------------------------------------------------


def test_get_pure_data_allow_list_excludes_iln():
    row = pd.Series({"plz": "70173", "iln": "12345678"})

    result = get_pure_data(row)

    assert "iln" not in result.index
    assert "plz" in result.index


# --------------------------------------------------------------------------
# FormattingRules / Prompts regression guards
# --------------------------------------------------------------------------


def test_formatting_rules_is_valid_and_get_pattern_still_work():
    assert FormattingRules.is_valid("plz", "70173")
    assert not FormattingRules.is_valid("plz", "7017")
    assert FormattingRules.get_pattern("plz") == r"^\d{5}$"


def test_prompts_has_the_renamed_misspelled_member():
    assert Prompts.DETECT_MISSPELLED_SYSTEM
    assert not hasattr(Prompts, "DETECT_FIX_MISSPELLED_SYSTEM")


# --------------------------------------------------------------------------
# PipelineTool defaults
# --------------------------------------------------------------------------


def test_pipeline_tool_base_defaults():
    class NoOpTool(PipelineTool):
        identity = _identity()

        def process_row(self, row, instance):
            return instance

    tool = NoOpTool()

    assert tool.batch_system_prompt is None
    assert tool.batch_max_tokens == 60


# --------------------------------------------------------------------------
# PipelineRunner internals
# --------------------------------------------------------------------------


def test_apply_accepted_writes_accepted_changes_onto_the_row():
    from sherpai_schemas.pipeline import PipelineRunner

    proposal = Proposal(identity=_identity(), changes=[FieldChange(column="plz", value="70173")])
    proposal.accept(reviewer="tester")
    finding = Finding(problem_type=ProblemType.FORMATTING, detection=proposal)
    instance = SherpAIInstance()
    instance.add_finding(finding)
    row = pd.Series({"plz": "7017"})

    updated = PipelineRunner._apply_accepted(row, instance)

    assert updated["plz"] == "70173"


def test_apply_accepted_ignores_proposals_that_are_not_accepted():
    from sherpai_schemas.pipeline import PipelineRunner

    proposal = Proposal(identity=_identity(), changes=[FieldChange(column="plz", value="70173")])
    proposal.mark_review_ready()
    finding = Finding(problem_type=ProblemType.FORMATTING, detection=proposal)
    instance = SherpAIInstance()
    instance.add_finding(finding)
    row = pd.Series({"plz": "7017"})

    updated = PipelineRunner._apply_accepted(row, instance)

    assert updated["plz"] == "7017"


def test_flush_batches_is_a_noop_without_a_batch_system_prompt():
    from sherpai_schemas.pipeline import PipelineRunner

    class NoOpTool(PipelineTool):
        identity = _identity()

        def process_row(self, row, instance):
            return instance

    proposal = Proposal(identity=_identity())
    proposal.mark_batching_ready("fix this")
    finding = Finding(problem_type=ProblemType.FORMATTING, detection=proposal)
    instance = SherpAIInstance()
    instance.add_finding(finding)

    runner = PipelineRunner(NoOpTool())
    runner._flush_batches(pd.Series([instance]))

    assert proposal.status == LifecycleStatus.BATCHING_READY
