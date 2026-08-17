"""Tests for the domain model: Proposal lifecycle, merge policy, instance, rules."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from conftest import make_accepted_correction, make_finding

from sherpai_schemas import (
    ChangeRole,
    FieldChange,
    FormattingRules,
    InvalidTransitionError,
    LatestAcceptedWins,
    LifecycleStatus,
    PipelineStage,
    ProblemType,
    Proposal,
    SherpAIInstance,
    ToolIdentity,
)
from sherpai_schemas.schemas import _TRANSITIONS

# --------------------------------------------------------------------------
# Proposal lifecycle
# --------------------------------------------------------------------------

# How to *reach* each status from a fresh DRAFTED proposal.
_PATHS = {
    LifecycleStatus.DRAFTED: [],
    LifecycleStatus.BATCHING_READY: ["batching"],
    LifecycleStatus.REVIEW_READY: ["review"],
    LifecycleStatus.ACCEPTED: ["review", "accept"],
    LifecycleStatus.REJECTED: ["review", "reject"],
}

# How to *attempt* a move to each status, whether or not it is legal.
_ATTEMPTS = {
    LifecycleStatus.BATCHING_READY: lambda p: p.mark_batching_ready("prompt"),
    LifecycleStatus.REVIEW_READY: lambda p: p.mark_review_ready(),
    LifecycleStatus.ACCEPTED: lambda p: p.accept(reviewer="tester"),
    LifecycleStatus.REJECTED: lambda p: p.reject(reviewer="tester", reason="nope"),
}

_STEPS = {
    "batching": lambda p: p.mark_batching_ready("prompt"),
    "review": lambda p: p.mark_review_ready(),
    "accept": lambda p: p.accept(reviewer="tester"),
    "reject": lambda p: p.reject(reviewer="tester", reason="nope"),
}


def advance_to(proposal: Proposal, status: LifecycleStatus) -> Proposal:
    for step in _PATHS[status]:
        _STEPS[step](proposal)
    assert proposal.status == status
    return proposal


def _legal_edges() -> list[tuple[LifecycleStatus, LifecycleStatus]]:
    return [(src, dst) for src, targets in _TRANSITIONS.items() for dst in targets]


def _illegal_edges() -> list[tuple[LifecycleStatus, LifecycleStatus]]:
    return [
        (src, dst)
        for src, targets in _TRANSITIONS.items()
        for dst in _ATTEMPTS
        if dst not in targets
    ]


@pytest.mark.parametrize(("source", "target"), _legal_edges())
def test_legal_transition_is_allowed(identity, source, target):
    proposal = advance_to(Proposal(identity=identity), source)

    _ATTEMPTS[target](proposal)

    assert proposal.status == target


@pytest.mark.parametrize(("source", "target"), _illegal_edges())
def test_illegal_transition_raises_and_leaves_status_untouched(identity, source, target):
    proposal = advance_to(Proposal(identity=identity), source)

    with pytest.raises(InvalidTransitionError):
        _ATTEMPTS[target](proposal)

    assert proposal.status == source


@pytest.mark.parametrize(
    "terminal", [LifecycleStatus.ACCEPTED, LifecycleStatus.REJECTED]
)
def test_terminal_statuses_have_no_outgoing_transitions(terminal):
    assert _TRANSITIONS[terminal] == frozenset()


def test_new_proposal_starts_drafted_without_decision(identity):
    proposal = Proposal(identity=identity)

    assert proposal.status == LifecycleStatus.DRAFTED
    assert proposal.decision is None
    assert proposal.pending_prompt is None


def test_mark_batching_ready_stores_prompt_and_review_ready_clears_it(identity):
    proposal = Proposal(identity=identity)

    proposal.mark_batching_ready("please fix Stuttgart")
    assert proposal.pending_prompt == "please fix Stuttgart"

    proposal.mark_review_ready()
    assert proposal.pending_prompt is None


def test_accept_records_reviewer_and_reason(identity):
    proposal = advance_to(Proposal(identity=identity), LifecycleStatus.REVIEW_READY)

    proposal.accept(reviewer="lasse", reason="looks right")

    assert proposal.decision.reviewer == "lasse"
    assert proposal.decision.reason == "looks right"
    assert proposal.can_apply() is True


def test_reject_records_decision_but_cannot_apply(identity):
    proposal = advance_to(Proposal(identity=identity), LifecycleStatus.REVIEW_READY)

    proposal.reject(reviewer="lasse", reason="hallucinated column")

    assert proposal.decision.reason == "hallucinated column"
    assert proposal.can_apply() is False


def test_transition_bumps_updated_at(identity):
    proposal = Proposal(identity=identity)
    before = proposal.updated_at

    proposal.mark_review_ready()

    assert proposal.updated_at >= before


# --------------------------------------------------------------------------
# Proposal.single / by_role
# --------------------------------------------------------------------------


def test_by_role_filters_to_the_requested_role(identity):
    proposal = Proposal(
        identity=identity,
        changes=[
            FieldChange(column="ort", role=ChangeRole.TARGET),
            FieldChange(column="zeile1", role=ChangeRole.SOURCE),
            FieldChange(column="plz", role=ChangeRole.CONTEXT),
        ],
    )

    assert [c.column for c in proposal.by_role(ChangeRole.TARGET)] == ["ort"]
    assert [c.column for c in proposal.by_role(ChangeRole.SOURCE)] == ["zeile1"]


def test_single_returns_the_only_change_of_that_role(identity):
    proposal = Proposal(
        identity=identity,
        changes=[
            FieldChange(column="ort", value="Stuttgart", role=ChangeRole.TARGET),
            FieldChange(column="zeile1", role=ChangeRole.SOURCE),
        ],
    )

    assert proposal.single().column == "ort"
    assert proposal.single(ChangeRole.SOURCE).column == "zeile1"


@pytest.mark.parametrize("count", [0, 2])
def test_single_raises_unless_exactly_one_match(identity, count):
    proposal = Proposal(
        identity=identity,
        changes=[
            FieldChange(column=f"col{i}", role=ChangeRole.TARGET) for i in range(count)
        ],
    )

    with pytest.raises(ValueError, match="Expected exactly one target change"):
        proposal.single()


# --------------------------------------------------------------------------
# LatestAcceptedWins
# --------------------------------------------------------------------------


def test_merge_policy_ignores_findings_without_a_correction(identity):
    findings = [make_finding(identity)]  # correction is None

    assert LatestAcceptedWins().resolve(findings) == {}


@pytest.mark.parametrize(
    "status", [LifecycleStatus.DRAFTED, LifecycleStatus.REVIEW_READY]
)
def test_merge_policy_ignores_corrections_that_are_not_accepted(identity, status):
    correction = advance_to(
        Proposal(identity=identity, changes=[FieldChange(column="ort", value="X")]),
        status,
    )
    findings = [make_finding(identity, correction=correction)]

    assert LatestAcceptedWins().resolve(findings) == {}


def test_merge_policy_ignores_rejected_corrections(identity):
    correction = Proposal(
        identity=identity, changes=[FieldChange(column="ort", value="X")]
    )
    correction.mark_review_ready()
    correction.reject(reviewer="tester", reason="wrong")
    findings = [make_finding(identity, correction=correction)]

    assert LatestAcceptedWins().resolve(findings) == {}


def test_merge_policy_keeps_the_latest_decision_per_column(identity, utc_now):
    older = make_accepted_correction(
        identity, column="ort", value="Alt", decided_at=utc_now - timedelta(hours=1)
    )
    newer = make_accepted_correction(
        identity, column="ort", value="Neu", decided_at=utc_now
    )
    findings = [
        make_finding(identity, correction=older),
        make_finding(identity, correction=newer),
    ]

    resolved = LatestAcceptedWins().resolve(findings)

    assert resolved["ort"].value == "Neu"


def test_merge_policy_is_order_independent(identity, utc_now):
    older = make_accepted_correction(
        identity, column="ort", value="Alt", decided_at=utc_now - timedelta(hours=1)
    )
    newer = make_accepted_correction(
        identity, column="ort", value="Neu", decided_at=utc_now
    )
    # newer first: the older one must not overwrite it on the second pass.
    findings = [
        make_finding(identity, correction=newer),
        make_finding(identity, correction=older),
    ]

    assert LatestAcceptedWins().resolve(findings)["ort"].value == "Neu"


def test_merge_policy_keeps_distinct_columns_side_by_side(identity, utc_now):
    findings = [
        make_finding(
            identity,
            correction=make_accepted_correction(
                identity, column="ort", value="Stuttgart", decided_at=utc_now
            ),
        ),
        make_finding(
            identity,
            correction=make_accepted_correction(
                identity, column="plz", value="70173", decided_at=utc_now
            ),
        ),
    ]

    resolved = LatestAcceptedWins().resolve(findings)

    assert set(resolved) == {"ort", "plz"}


# --------------------------------------------------------------------------
# Finding
# --------------------------------------------------------------------------


def test_finding_can_apply_only_with_an_accepted_correction(identity):
    assert make_finding(identity).can_apply() is False
    assert (
        make_finding(
            identity, correction=make_accepted_correction(identity)
        ).can_apply()
        is True
    )


# --------------------------------------------------------------------------
# SherpAIInstance
# --------------------------------------------------------------------------


def test_add_finding_appends_and_by_type_filters(identity):
    instance = SherpAIInstance()
    instance.add_finding(make_finding(identity, problem_type=ProblemType.MISPLACED))
    instance.add_finding(make_finding(identity, problem_type=ProblemType.FORMATTING))

    assert len(instance.findings) == 2
    assert len(instance.by_type(ProblemType.MISPLACED)) == 1
    assert instance.by_type(ProblemType.MISSING_VALUE) == []


def test_get_affected_cols_without_arguments_returns_empty(identity):
    instance = SherpAIInstance(findings=[make_finding(identity)])

    assert instance.get_affected_cols() == []


def test_get_affected_cols_unions_and_dedupes_across_findings(identity):
    instance = SherpAIInstance(
        findings=[
            make_finding(identity, detection_columns=("ort", "plz")),
            make_finding(identity, detection_columns=("plz", "land")),
        ]
    )

    # Built from a set, so order is not part of the contract.
    assert sorted(instance.get_affected_cols(ProblemType.MISPLACED)) == [
        "land",
        "ort",
        "plz",
    ]


def test_get_affected_cols_spans_multiple_problem_types(identity):
    instance = SherpAIInstance(
        findings=[
            make_finding(
                identity,
                problem_type=ProblemType.MISPLACED,
                detection_columns=("ort",),
            ),
            make_finding(
                identity,
                problem_type=ProblemType.FORMATTING,
                detection_columns=("plz",),
            ),
        ]
    )

    affected = instance.get_affected_cols(
        ProblemType.MISPLACED, ProblemType.FORMATTING
    )

    assert sorted(affected) == ["ort", "plz"]


def test_apply_solutions_writes_accepted_values_onto_the_row(
    identity, data_row, utc_now
):
    correction = make_accepted_correction(
        identity, column="ort", value="Karlsruhe", decided_at=utc_now
    )
    instance = SherpAIInstance(findings=[make_finding(identity, correction=correction)])

    updated = instance.apply_solutions(data_row)

    assert updated["ort"] == "Karlsruhe"


def test_apply_solutions_skips_columns_absent_from_the_row(identity, data_row, utc_now):
    correction = make_accepted_correction(
        identity, column="not_a_column", value="X", decided_at=utc_now
    )
    instance = SherpAIInstance(findings=[make_finding(identity, correction=correction)])

    updated = instance.apply_solutions(data_row)

    assert "not_a_column" not in updated.index


def test_apply_solutions_accepts_a_custom_merge_policy(identity, data_row):
    class AlwaysStuttgart:
        def resolve(self, findings):
            return {"ort": FieldChange(column="ort", value="Stuttgart")}

    instance = SherpAIInstance()

    updated = instance.apply_solutions(data_row, policy=AlwaysStuttgart())

    assert updated["ort"] == "Stuttgart"


def test_instance_survives_a_string_round_trip(identity, utc_now):
    correction = make_accepted_correction(
        identity, column="ort", value="Karlsruhe", decided_at=utc_now
    )
    original = SherpAIInstance(findings=[make_finding(identity, correction=correction)])

    restored = SherpAIInstance.parse_from_str(str(original))

    assert restored == original


def test_parse_from_str_treats_empty_input_as_an_empty_instance():
    assert SherpAIInstance.parse_from_str("") == SherpAIInstance()
    assert SherpAIInstance.parse_from_str("").findings == []


# --------------------------------------------------------------------------
# ToolIdentity
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "tool", "tier", "expected"),
    [
        (PipelineStage.DETECTION, "misplaced", 1, "detection_misplaced_tier1.yml"),
        (
            PipelineStage.CORRECTION,
            "validation_missing",
            1,
            "correction_validation_missing_tier1.yml",
        ),
        (PipelineStage.INTEGRATION, "ditto", 1, "integration_ditto_tier1.yml"),
    ],
)
def test_compose_name_matches_the_compose_collection_convention(
    stage, tool, tier, expected
):
    # These filenames are the contract with hostData/compose_collection/.
    assert ToolIdentity(stage=stage, tool=tool, tier=tier).compose_name() == expected


def test_tier_defaults_to_one():
    assert ToolIdentity(stage=PipelineStage.DETECTION, tool="missing").tier == 1


@pytest.mark.parametrize("tool", [pt.value for pt in ProblemType])
def test_as_problem_type_resolves_detection_and_correction_tools(tool):
    identity = ToolIdentity(stage=PipelineStage.DETECTION, tool=tool)

    assert identity.as_problem_type() == ProblemType(tool)


@pytest.mark.parametrize("tool", ["ditto", "duplicate_pairs"])
def test_as_problem_type_is_none_for_integration_tools(tool):
    identity = ToolIdentity(stage=PipelineStage.INTEGRATION, tool=tool)

    assert identity.as_problem_type() is None


# --------------------------------------------------------------------------
# FormattingRules
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("hybrid", "PERS_1_42"),
        ("klassifik", "10"),
        ("klassifik", "90"),
        ("nr", "4711"),
        ("plz", "70173"),
        ("typ", "3"),
        ("ustid", "DE123456789"),
        ("zeile1", "Hauptstrasse 12"),
        ("zeile1", "Hauptstrasse 12a"),
        ("ort", "Stuttgart"),
    ],
)
def test_is_valid_accepts_well_formed_erp_values(column, value):
    assert FormattingRules.is_valid(column, value) is True


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("hybrid", "PERS-1-42"),
        ("klassifik", "30"),
        ("nr", "12345678"),  # more than 7 digits
        ("plz", "7017"),  # four digits
        ("plz", "701730"),  # six digits
        ("typ", "4"),
        ("ustid", "DE12345678"),  # nine digits required
        ("ustid", "de123456789"),  # country code must be upper-case
    ],
)
def test_is_valid_rejects_malformed_erp_values(column, value):
    assert FormattingRules.is_valid(column, value) is False


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
def test_is_valid_rejects_missing_values(missing):
    assert FormattingRules.is_valid("plz", missing) is False


def test_is_valid_coerces_non_strings_before_matching():
    assert FormattingRules.is_valid("plz", 70173) is True
    assert FormattingRules.is_valid("nr", 4711) is True


def test_is_valid_passes_through_columns_without_a_rule():
    # No pattern attribute -> nothing to enforce, so anything goes.
    assert FormattingRules.is_valid("unknown_column", "whatever") is True


@pytest.mark.parametrize("column", ["PLZ", "Plz", "plz"])
def test_is_valid_is_case_insensitive_in_the_column_name(column):
    assert FormattingRules.is_valid(column, "70173") is True


@pytest.mark.parametrize(("column", "value"), [("steuernr", "anything"), ("iln", "")])
def test_catch_all_rules_accept_everything(column, value):
    assert FormattingRules.is_valid(column, value) is True


def test_get_pattern_returns_the_raw_regex_string():
    assert FormattingRules.get_pattern("plz") == r"^\d{5}$"
    assert FormattingRules.get_pattern("PLZ") == r"^\d{5}$"


def test_get_pattern_returns_none_for_unknown_columns():
    assert FormattingRules.get_pattern("unknown_column") is None


def test_get_pattern_ignores_non_pattern_attributes():
    # getattr would find the method itself; the isinstance guard rejects it.
    assert FormattingRules.get_pattern("is_valid") is None


# --- Known bugs: these assert the intended behavior and fail until it is met -


@pytest.mark.known_bug
@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("land", "DE!!!"),
        ("ort", "Stuttgart <script>"),
        ("name1", "Gebauer GmbH \x00"),
    ],
)
def test_rules_reject_trailing_garbage(column, value):
    """A value must match its pattern end to end, not just at the start.

    BUG: is_valid uses re.match, which anchors only at position 0. The land,
    ort and name1 patterns carry no trailing `$`, so anything after a valid
    prefix is accepted -- "DE!!!" passes as a country code.
    FIX: use pattern.fullmatch(str(value)) in FormattingRules.is_valid
    (sherpai_schemas/schemas.py:437). Every already-passing case in
    test_is_valid_accepts_well_formed_erp_values stays valid under fullmatch.
    """
    assert FormattingRules.is_valid(column, value) is False


@pytest.mark.parametrize("value", ["!!!DE", " Stuttgart"])
def test_rules_reject_leading_garbage(value):
    # re.match does anchor at position 0, so the asymmetry above is one-sided.
    assert FormattingRules.is_valid("land", value) is False


def test_datetime_fields_are_timezone_aware(identity):
    proposal = Proposal(identity=identity)

    assert proposal.created_at.tzinfo is not None
    assert proposal.created_at.tzinfo.utcoffset(proposal.created_at) == timedelta(0)
    assert isinstance(proposal.created_at, datetime)
    assert proposal.created_at <= datetime.now(timezone.utc)
