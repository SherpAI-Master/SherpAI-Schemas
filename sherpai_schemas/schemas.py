"""All schemas and classes used for organization."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field, fields
from enum import Enum
from datetime import datetime, timezone
from typing import Optional
import json
from pydantic import BaseModel, Field

import pandas as pd
from enum import StrEnum


class ToolID(Enum):
    CORRECTION_FORMATTING_TIER1 = 1
    CORRECTION_INCOMPLETE_TIER1 = 2
    CORRECTION_MISPLACED_TIER1 = 3
    CORRECTION_MISSPELLED_TIER1 = 4
    CORRECTION_VALIDATION_MISSING_TIER1 = 5
    DETECTION_FORMATTING_TIER1 = 6
    DETECTION_INCOMPLETE_TIER1 = 7
    DETECTION_MISPLACED_TIER1 = 8
    DETECTION_MISSING_TIER1 = 9
    DETECTION_MISSPELLED_TIER1 = 10
    DETECTION_VALIDATION_TIER1 = 11
    INTEGRATION_DITTO_TIER1 = 12
    INTEGRATION_DUPLICATION_PAIRS_TIER1 = 13


class ProblemID(Enum):
    INCOMPLETE = 1
    MISPLACED = 2
    FORMATTING = 3
    MISSPELLED = 4
    MISSING_VALUE = 5
    VALIDATION = 6


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Acceptance(BaseModel):
    value: bool = False
    reason: str = ""
    user: str = ""
    time_stamp: datetime = Field(default_factory=_now)


class ToolUse(BaseModel):
    value: str = ""
    reason: str = ""
    used_tool: ToolID | None = None
    time_stamp: datetime = Field(default_factory=_now)
    accepted: Acceptance | None = None   # not reviewed yet


class Pair(BaseModel):
    affected_col: str = ""
    problem: ToolUse | None = None
    solution: ToolUse | None = None
    
class SherpAIInstance(BaseModel):
    """Identified problems in a data row.

    Here, the attribute name is the problem type and the lists contain the affected rows
    - incomplete: An existing abbreviation in the row                           1
    - misplaced: A value is set in the wrong column of the csv                  2
    - formatting: The data has inconsistent formatting or wrong standards       3
    - misspelled: Probable wrong spelling in data                               4
    - missing_value: A value is missing from the column                         5
    - validation: Contradicting outside information spotted                     6
    """

    incomplete: list[Pair] = field(default_factory=list)
    misplaced: list[Pair] = field(default_factory=list)
    formatting: list[Pair] = field(default_factory=list)
    misspelled: list[Pair] = field(default_factory=list)
    missing_value: list[Pair] = field(default_factory=list)
    validation: list[Pair] = field(default_factory=list)

    def __str__(self) -> str:
        """Convert SherpAiInstance into json-format"""
        return self.model_dump_json()
    
    @staticmethod
    def parse_from_str(label: str) -> SherpAIInstance:
        """Convert ProblemID string back into a Identified problem object."""
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
        Input: {"hybrid": "PERS_1_12, "name1": "Tehno Gmbh", "zeile1": "Beriner str. 12", "city": "Berln", "zip": "10115"}
        Output: {"zeile1": "Berliner Straße 12", "city": "Berlin"}

        # Your Turn
        Process the provided input data now.

        # Input Data
    """
    FIX_INCOMPLETE_SYSTEM = """
    # Role
    You are a German Data Quality Specialist. Your task is to write out any abbreviations!

    # Instructions
    You receive an string with abbreviations. Write out any other abbreviation and return the completed string with double quotes!
    Ignore standardized abbreviations like Co KG or Inc.

    # Examples
    Input: The value  "Manufaktur u. Produktion Dachmann" of column name1
    Output: "Manufaktur und Produktion Dachmann"

    Input: The value "Aluminiumwerk Hr. Meier" of column name1
    Output: "Aluminiumwerk Herr Meier"

    # Input
    """
    FIX_INCOMPLETE_USER="""
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
    Input: {{"format": "\\d{{2}}-\\d{{2}}-\\d{{4}}", "data": "01.03.2025"}}
    Output: {{"data": "01-03-2025", "fixable": true}}

    Input: {{"format": "\\d{{2}}-\\d{{2}}-\\d{{4}}", "data": "2025"}}
    Output: {{"data": "2025", "fixable": false}}

    Input: {{"format": "\\d{{2}}-\\d{{2}}-\\d{{4}}", "data": "Jan. 23rd, 2022"}}
    Output: {{"data": "23-01-2022", "fixable": true}}

    # Your Turn
    Process this given input data:

    # Input data
    """
    FIX_FORMATTING_USER="""{{"format": "{col_rule}", "data": "{col_value}"}}"""
    FIX_MISPLACED_SYSTEM = """You are a data-validation expert correcting mistakenly placed values in columns."""
    FIX_MISPLACED_USER = """A value from column "{missing_col}" was mistakenly placed inside 
        the value "{overfilled_value}" of column "{overfilled_col}".

        Your task:
        1. Extract the correct value for column "{missing_col}" from the text "{overfilled_value}".
        2. Return the corrected values for both columns.
        3. Output **only** valid JSON in this exact format:
        {{
            "{missing_col}": "<extracted_value>",
            "{overfilled_col}": "<cleaned_value>"
        }}

        Rules:
        - Do not include any explanation or extra text.
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
    zeile1: re.Pattern = re.compile(r"^[A-ZÄÖÜa-zäöüß.\s-]+\s+\d+(\s*[/-]\s*\d+|[a-zA-Z])?\s*$")

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