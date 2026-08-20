"""All schemas and classes used for organization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

import pandas as pd
from enum import StrEnum


class Fix(BaseModel):
    column: str = Field(..., description="Name of the column being corrected")
    corrected_value: str = Field(..., description="The corrected value, as a string")
    reason: str = Field(
        ..., description="Brief explanation for why this correction was made"
    )


class LlmResponse(BaseModel):
    fixes: list[Fix]


class PipelineStage(StrEnum):
    """Which stage of the pipeline a tool belongs to."""

    DETECTION = "detection"
    CORRECTION = "correction"
    INTEGRATION = "integration"


class ProblemType(StrEnum):
    """The kind of data-quality problem a Finding represents."""

    FORMATTING = "formatting"
    INCOMPLETE = "incomplete"
    MISPLACED = "misplaced"
    MISSING_VALUE = "missing_value"
    MISSPELLED = "misspelled"
    VALIDATION = "validation"


class ChangeRole(StrEnum):
    """Which side of a change a FieldChange represents.

    Most Proposals carry a single, unambiguous FieldChange -- those default to
    TARGET, so Proposal.single() finds them without callers ever having to set
    role= explicitly. SOURCE only shows up for tools (like "misplaced") that
    need to describe two columns at once.
    """

    TARGET = "target"
    SOURCE = "source"


class LifecycleStatus(str, Enum):
    """Where a Proposal sits in its review/batching lifecycle."""

    PENDING = "pending"
    BATCHING_READY = "batching_ready"
    REVIEW_READY = "review_ready"
    ACCEPTED = "accepted"


_TOOL_TO_PROBLEM_TYPE: dict[str, ProblemType] = {
    "formatting": ProblemType.FORMATTING,
    "incomplete": ProblemType.INCOMPLETE,
    "misplaced": ProblemType.MISPLACED,
    "missing": ProblemType.MISSING_VALUE,
    "misspelled": ProblemType.MISSPELLED,
    "validation": ProblemType.VALIDATION,
}


class ToolIdentity(BaseModel):
    """Identifies a single pipeline tool: which stage, which tool, which tier."""

    stage: PipelineStage
    tool: str
    tier: int

    def compose_name(self) -> str:
        """The compose service name this tool runs under, e.g. 'detection_formatting_tier1.yml'."""
        return f"{self.stage.value}_{self.tool}_tier{self.tier}.yml"

    def as_problem_type(self) -> ProblemType | None:
        """The ProblemType this tool's own tool name maps to, or None if it spans several."""
        return _TOOL_TO_PROBLEM_TYPE.get(self.tool)


class FieldChange(BaseModel):
    """A single column/value pair proposed by a detection or correction."""

    column: str
    value: Any = None
    role: ChangeRole = ChangeRole.TARGET


class Proposal(BaseModel):
    """A detection or correction: which tool made it, what it wants to change, and its review state."""

    identity: ToolIdentity
    changes: list[FieldChange] = Field(default_factory=list)
    reason: str = ""
    status: LifecycleStatus = LifecycleStatus.PENDING
    pending_prompt: str = ""
    reviewed_by: str = ""

    def mark_review_ready(self) -> None:
        self.status = LifecycleStatus.REVIEW_READY

    def mark_batching_ready(self, prompt: str) -> None:
        self.status = LifecycleStatus.BATCHING_READY
        self.pending_prompt = prompt

    def single(self, role: ChangeRole | None = None) -> FieldChange:
        """Return the sole change for the given role (default TARGET), or raise."""
        effective_role = role if role is not None else ChangeRole.TARGET
        matches = [change for change in self.changes if change.role == effective_role]
        if len(matches) != 1:
            msg = f"Expected exactly one {effective_role.value} change, found {len(matches)}"
            raise ValueError(msg)
        return matches[0]

    def accept(self, reviewer: str) -> None:
        self.status = LifecycleStatus.ACCEPTED
        self.reviewed_by = reviewer


class Finding(BaseModel):
    """One detected problem: what kind, how it was detected, and (once available) how to fix it."""

    problem_type: ProblemType
    detection: Proposal
    correction: Proposal | None = None


class SherpAIInstance(BaseModel):
    """All findings identified in a data row."""

    findings: list[Finding] = Field(default_factory=list)

    def add_finding(self, finding: Finding) -> None:
        self.findings.append(finding)

    def by_type(self, problem_type: ProblemType) -> list[Finding]:
        """All findings of the given problem type."""
        return [finding for finding in self.findings if finding.problem_type == problem_type]

    def get_affected_cols(self, *problem_types: ProblemType) -> list[str]:
        """Columns already flagged by a detection of any of the given problem types."""
        wanted = set(problem_types)
        seen: set[str] = set()
        cols: list[str] = []
        for finding in self.findings:
            if finding.problem_type not in wanted:
                continue
            for change in finding.detection.changes:
                if change.column not in seen:
                    seen.add(change.column)
                    cols.append(change.column)
        return cols

    def __str__(self) -> str:
        """Convert SherpAiInstance into json-format"""
        return self.model_dump_json()

    @staticmethod
    def parse_from_str(label: str) -> SherpAIInstance:
        """Convert a stringified SherpAIInstance back into an object."""
        if not label:
            return SherpAIInstance()
        return SherpAIInstance.model_validate_json(label)


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
    You are a enterprise identification specialist. Your task is to identify enterprises from normal individual names from the column "klassifik".

    # Instructions
    You receive a name of a company, person. Your job is to identify if this name belongs to a company or a person!
    Return your guess with the following identifiers: COMPANY=10, PERSON=20, UNKNOWN=90 in JSON!
    Return **ONLY** valid JSON with your prediction and a very short explaination!

    # Examples
    Input: "Dirk Wreiniger GmbH"
    Output: {{"column": "klassifik , "corrected_value": 10, "reason": "Because GmbH is in the name"}}

    Input: "Trikton Ltd."
    Output: {{"column": "klassifik , "corrected_value": 10, "reason": "Because Ltd. is in the name"}}

    Input: "Tom Yarkson"
    Output: {{"column": "klassifik , "corrected_value": 20, "reason": "Just a normal name"}}

    Input: "Wrench"
    Output: {{"column": "klassifik , "corrected_value": 90, "reason": "Unidentifiable"}}

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
