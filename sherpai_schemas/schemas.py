"""All schemas and classes used for organization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, StrEnum
from typing import Protocol

import pandas as pd
from pydantic import BaseModel, Field


class Fix(BaseModel):
    column: str = Field(..., description="Name of the column being corrected")
    corrected_value: str = Field(..., description="The corrected value, as a string")
    reason: str = Field(
        ..., description="Brief explanation for why this correction was made"
    )


class LlmResponse(BaseModel):
    fixes: list[Fix]


class ProblemType(str, Enum):
    """The category of data-quality issue a Finding represents.

    - incomplete: An existing abbreviation in the row
    - misplaced: A value is set in the wrong column of the csv
    - formatting: The data has inconsistent formatting or wrong standards
    - misspelled: Probable wrong spelling in data
    - missing_value: A value is missing from the column
    - validation: Contradicting outside information spotted
    """

    INCOMPLETE = "incomplete"
    MISPLACED = "misplaced"
    FORMATTING = "formatting"
    MISSPELLED = "misspelled"
    MISSING_VALUE = "missing_value"
    VALIDATION = "validation"


class PipelineStage(str, Enum):
    """Where in the pipeline a tool runs."""

    DETECTION = "detection"
    CORRECTION = "correction"
    INTEGRATION = "integration"


class ToolIdentity(BaseModel):
    """Identifies one pipeline tool/container.

    Constructible 1:1 from one instructions.json entry
    ({"pool": ..., "tool": ..., "tier": ...}).
    """

    stage: PipelineStage
    tool: str
    tier: int = 1

    def compose_name(self) -> str:
        """Single source of truth for the pool_tool_tierN.yml naming convention."""
        return f"{self.stage.value}_{self.tool}_tier{self.tier}.yml"

    def as_problem_type(self) -> ProblemType | None:
        """ProblemType(tool) for detection/correction tools; None for integration
        tools (ditto, duplicate_pairs), which have no ProblemType counterpart.
        """
        try:
            return ProblemType(self.tool)
        except ValueError:
            return None


class ChangeRole(str, Enum):
    """What a FieldChange means within a Proposal."""

    TARGET = "target"
    SOURCE = "source"
    CONTEXT = "context"


class FieldChange(BaseModel):
    """One column+value touched by a detection or correction."""

    column: str
    value: str | int | float | None = None
    role: ChangeRole = ChangeRole.TARGET


class LifecycleStatus(str, Enum):
    """Where a Proposal sits between being drafted and being reviewed."""

    DRAFTED = "drafted"
    BATCHING_READY = "batching_ready"
    REVIEW_READY = "review_ready"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


_TRANSITIONS: dict[LifecycleStatus, frozenset[LifecycleStatus]] = {
    LifecycleStatus.DRAFTED: frozenset({LifecycleStatus.BATCHING_READY, LifecycleStatus.REVIEW_READY}),
    LifecycleStatus.BATCHING_READY: frozenset({LifecycleStatus.REVIEW_READY}),
    LifecycleStatus.REVIEW_READY: frozenset({LifecycleStatus.ACCEPTED, LifecycleStatus.REJECTED}),
    LifecycleStatus.ACCEPTED: frozenset(),
    LifecycleStatus.REJECTED: frozenset(),
}


class InvalidTransitionError(RuntimeError):
    """Raised when a Proposal's requested status isn't reachable from its current one."""


class Decision(BaseModel):
    """Immutable human review outcome; unset until status reaches ACCEPTED/REJECTED."""

    reviewer: str
    reason: str = ""
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Proposal(BaseModel):
    """A single tool's proposed detection or correction, plus its review lifecycle."""

    identity: ToolIdentity
    changes: list[FieldChange] = Field(default_factory=list)
    reason: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: LifecycleStatus = LifecycleStatus.DRAFTED
    pending_prompt: str | None = None
    decision: Decision | None = None

    def single(self, role: ChangeRole = ChangeRole.TARGET) -> FieldChange:
        """The one FieldChange with the given role; raises unless there's exactly one."""
        matches = self.by_role(role)
        if len(matches) != 1:
            msg = f"Expected exactly one {role.value} change, found {len(matches)}"
            raise ValueError(msg)
        return matches[0]

    def by_role(self, role: ChangeRole) -> list[FieldChange]:
        return [change for change in self.changes if change.role == role]

    def mark_batching_ready(self, prompt: str) -> None:
        self._transition(LifecycleStatus.BATCHING_READY)
        self.pending_prompt = prompt

    def mark_review_ready(self) -> None:
        self._transition(LifecycleStatus.REVIEW_READY)
        self.pending_prompt = None

    def accept(self, reviewer: str, reason: str = "") -> None:
        self._transition(LifecycleStatus.ACCEPTED)
        self.decision = Decision(reviewer=reviewer, reason=reason)

    def reject(self, reviewer: str, reason: str) -> None:
        self._transition(LifecycleStatus.REJECTED)
        self.decision = Decision(reviewer=reviewer, reason=reason)

    def can_apply(self) -> bool:
        """True iff this proposal has been accepted by a reviewer."""
        return self.status == LifecycleStatus.ACCEPTED

    def _transition(self, target: LifecycleStatus) -> None:
        if target not in _TRANSITIONS[self.status]:
            msg = f"Cannot move Proposal from {self.status.value} to {target.value}"
            raise InvalidTransitionError(msg)
        self.status = target
        self.updated_at = datetime.now(timezone.utc)


class Finding(BaseModel):
    """One detected problem in a data row, plus its proposed correction (if any)."""

    problem_type: ProblemType
    detection: Proposal
    correction: Proposal | None = None

    def can_apply(self) -> bool:
        return self.correction is not None and self.correction.can_apply()


class MergePolicy(Protocol):
    def resolve(self, findings: list[Finding]) -> dict[str, FieldChange]: ...


class LatestAcceptedWins:
    """Default MergePolicy: among findings whose correction is ACCEPTED, keep the
    FieldChange with the latest Proposal.decision.decided_at per column.
    """

    def resolve(self, findings: list[Finding]) -> dict[str, FieldChange]:
        latest: dict[str, tuple[datetime, FieldChange]] = {}

        for finding in findings:
            if not finding.can_apply():
                continue

            decided_at = finding.correction.decision.decided_at
            for change in finding.correction.changes:
                if change.column not in latest or decided_at > latest[change.column][0]:
                    latest[change.column] = (decided_at, change)

        return {col: change for col, (_, change) in latest.items()}


_DEFAULT_MERGE_POLICY = LatestAcceptedWins()


class SherpAIInstance(BaseModel):
    """All findings identified in a data row."""

    findings: list[Finding] = Field(default_factory=list)

    def __str__(self) -> str:
        """Convert SherpAIInstance into json-format"""
        return self.model_dump_json()

    @staticmethod
    def parse_from_str(label: str) -> SherpAIInstance:
        """Convert a JSON string back into a SherpAIInstance."""
        if not label:
            return SherpAIInstance()
        return SherpAIInstance.model_validate_json(label)

    def by_type(self, problem_type: ProblemType) -> list[Finding]:
        """All findings of a given problem type."""
        return [finding for finding in self.findings if finding.problem_type == problem_type]

    def add_finding(self, finding: Finding) -> None:
        self.findings.append(finding)

    def get_affected_cols(self, *problem_types: ProblemType) -> list[str]:
        """Get all cols touched by detections of any of the given problem types."""
        if not problem_types:
            return []

        affected_cols: set[str] = set()
        for problem_type in problem_types:
            for finding in self.by_type(problem_type):
                affected_cols.update(change.column for change in finding.detection.changes)

        return list(affected_cols)

    def apply_solutions(self, data_row: pd.Series, policy: MergePolicy = _DEFAULT_MERGE_POLICY) -> pd.Series:
        """Update current data with the latest accepted corrections."""
        for col, change in policy.resolve(self.findings).items():
            if col in data_row.index:
                data_row[col] = change.value

        return data_row


class Prompts(StrEnum):
    """Contains all prompts of the problem identification, fixAIs."""

    DETECT_MISPLACED_SYSTEM = """
        You are a data validation expert. Your task is to find values placed in the wrong columns. The correct schema is: {\"hybrid\": \"PERS_#_######\", \"typ\": #, \"nr\": ######, \"klassifik\": \"#\", \"name1\": \"Company/Person\", \"zeile1\": \"Address\", \"plz\": \"Postal Code\", \"ort\": \"City\", \"land\": \"Country\", \"ustid\": \"########\", \"steuernr\": \"########\", \"iln\": \"########\"}"}.
        If you find misplacements, output a JSON object containing the columns needed to be switched!
        """
    DETECT_MISSPELLED_SYSTEM = """
        # Role
        You are a German Data Quality Specialist. Your task is to normalize and spell-check German address data.

        # Instructions
        Check every word for spelling errors, incorrect capitalization, or letter switches etc.

        # Output Format
        Provide ONLY the output as a JSON **with changes**. Dont give any extra explainations! Dont just repeat already given values!

        # Example
        Input: {"hybrid": "PERS_1_12, "name1": "Tehno Gmbh", "zeile1": "Beriner Str. 12", "city": "Berln", "zip": "10115"}
        Output: {{
                    "fixes": [
                        {{
                            "column": "zeile1",
                            "corrected_value": "Berliner Str. 12",
                            "reason": "Berlin is written with an L."
                        }},
                        {{
                            "column: "city",
                            "corrected_value": "Berlin",
                            "reason": "Berlin is written with an I."
                        }}
                    ]
                }}
        Output: {"zeile1": "Berliner Straße 12", "city": "Berlin"}

        # Your Turn
        Process the provided input data now.

        # Input Data
    """
    FIX_INCOMPLETE_SYSTEM = """
    # Role
    You are a German Data Quality Specialist. Your task is to write out any abbreviations!

    # Instructions
    You receive an string with abbreviations. Write out any other abbreviation and return a JSON with corrected_value as its only key-value pair!
    Ignore standardized abbreviations like Co KG or Inc and leave data identifiers as is!

    # Examples
    Input: The value  "Manufaktur u. Produktion Dachmann" of column name1
    Output: {"column": "name1", "corrected_value": "Manufaktur und Produktion Dachmann", reason: "'u.' hat die Bedeutung 'und'."}

    Input: The value "NY" of column ort
    Output: {"column": "ort", "corrected_value": "New York", reason: "NY is a real abbreviation for New York."}

    # Input
    """
    FIX_INCOMPLETE_USER = """
    The value"{col_value}" of column {col_name}
    """

    FIX_FORMATTING_SYSTEM = """
    # Role
    You are a data formatting expert. Your task is to fix the formatting of data if possible!

    # Instructions
    You receive a data point where the formatting is **not** correct. You should fix the formatting by adhering to the given regex.
    Check if the current data can be adapted to the needed format.
    If so, return a JSON object with the fixed data and a boolean value if it was fixable or not!

    # Examples
    Input: {{"column_name": "date", "column_value": "01.03.2025", "format": "\\d{{2}}-\\d{{2}}-\\d{{4}}"}}
    Output: {{"column_name": "date", "column_value": "01-03-2025", "reason: "Current data could be applied to given format."}}

    Input: {{"column_name": "date", "column_value": "2025", "format": "\\d{{2}}-\\d{{2}}-\\d{{4}}"}}
    Output: {{"column_name": "date", "column_value": null, "reason": "Missing data in the original data to fullfill given format."}}

    Input: Input: {{"column_name": "date", "column_value": "Jan. 23rd, 2022", "format": "\\d{{2}}-\\d{{2}}-\\d{{4}}"}}
    Output: {{"column_name": "date", "column_value": "23-01-2022", "realson": "Date format was in written form and was transfomred into schema."}}

    # Your Turn
    Process this given input data:

    # Input data
    """
    FIX_FORMATTING_USER = """{{"column_name": "{col_name}", "column_value": {col_value}", "format": "{col_rule}"}}"""
    FIX_MISPLACED_SYSTEM = """You are a data-validation expert correcting mistakenly placed values in columns."""
    FIX_MISPLACED_USER = """A value from column "{missing_col}" was mistakenly placed inside
        the value "{overfilled_value}" of column "{overfilled_col}".

        Your task:
        1. Extract the correct value for column "{missing_col}" from the text "{overfilled_value}".
        2. Determine the cleaned value for column "{overfilled_col}" with the extracted part removed.
        3. Return a fix for **both** columns.
        4. Output **only** valid JSON in this exact format:
        {{
            "fixes": [
                {{
                    "column": "{missing_col}",
                    "corrected_value": "<extracted_value>",
                    "reason": "<brief explanation for this correction>"
                }},
                {{
                    "column": "{overfilled_col}",
                    "corrected_value": "<cleaned_value>",
                    "reason": "<brief explanation for this correction>"
                }}
            ]
        }}

        Rules:
        - The "fixes" list must contain exactly two entries, one per column above, in that order.
        - "reason" must be a short, single-sentence explanation — no extra commentary elsewhere.
        - Do not include any explanation or extra text outside the JSON object.
        - If you are uncertain, make the most reasonable inference from the provided value.
        """
    EXTRACT_ADDRESS_SYSTEM = """You extract addresses from google search snippets. The correct schema is: {\"street\": \"street and street nr\",\"city\": \"city\",\"zip\": \"#####\",\"country\": \"country\"}. If no address is found or the address does not make sense, return an empty JSON object "{}" with no commentary. Respons strictly in JSON!"""
    EXTRACT_KLASSIFIK_SYSTEM = """
    # Role
    You are a enterprise identification specialist. Your task is to identify enterprises from normal individual names.

    # Instructions
    You receive a name of a company, person. Your job is to identify if this name belongs to a company or a person!
    Return your guess with the following identifiers: COMPANY=10, PERSON=20, UNKNOWN=90 in JSON!
    Return **ONLY** valid JSON with your prediction and a very short explaination!

    # Examples
    Input: "Dirk Wreiniger GmbH"
    Output: {{"prediction": 10, "reason": "Because GmbH is in the name"}}

    Input: "Trikton Ltd."
    Output: {{"prediction": 10, "reason": "Because Ltd. is in the name"}}

    Input: "Tom Yarkson"
    Output: {{"prediction": 20, "reason": "Just a normal name"}}

    Input: "Wrench"
    Output: {{"prediction": 90, "reason": "Unidentifiable"}}

    # Your Turn
    Process this given input data:

    # Input data
    """


@dataclass(frozen=True)
class FormattingRules:
    """Class holding the regex rules as pre-compiled patterns."""

    hybrid: re.Pattern = re.compile(r"^PERS_\d_\d+$")
    iln: re.Pattern = re.compile(r".*")
    klassifik: re.Pattern = re.compile(r"^(10|20|90)$")
    land: re.Pattern = re.compile(r"[A-ZÄÖÜa-zäöüß.-]+")
    name1: re.Pattern = re.compile(r"[A-ZÄÖÜa-zäöüß.\s-]+")
    nr: re.Pattern = re.compile(r"^\d{1,7}$")
    ort: re.Pattern = re.compile(r"[A-ZÄÖÜa-zäöüß.\s-]+")
    plz: re.Pattern = re.compile(r"^\d{5}$")
    steuernr: re.Pattern = re.compile(r".*")
    typ: re.Pattern = re.compile(r"^[123]$")
    ustid: re.Pattern = re.compile(r"^[A-Z]{2}\d{9}$")
    zeile1: re.Pattern = re.compile(
        r"^[A-ZÄÖÜa-zäöüß.\s-]+\s+\d+(\s*[/-]\s*\d+|[a-zA-Z])?\s*$"
    )

    @staticmethod
    def get_pattern(column: str) -> str | None:
        """Retrieves the raw regex string for a specific column."""
        attr = getattr(FormattingRules, column.lower(), None)
        return attr.pattern if isinstance(attr, re.Pattern) else None

    @staticmethod
    def is_valid(column: str, value: any) -> bool:
        if value is None or pd.isna(value):
            return False

        pattern = getattr(FormattingRules, column.lower(), None)
        if pattern and isinstance(pattern, re.Pattern):
            return bool(pattern.match(str(value)))

        return True
