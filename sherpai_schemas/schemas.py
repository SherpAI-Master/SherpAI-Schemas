"""All schemas and classes used for organization."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field, fields
from enum import Enum

import pandas as pd
from enum import StrEnum


class ProblemID(Enum):
    """Enumerates all possible problem categories with unique IDs."""

    INCOMPLETE = 1
    MISPLACED = 2
    FORMATTING = 3
    MISSPELLED = 4
    MISSING_VALUE = 5
    VALIDATION = 6


@dataclass
class ProblemInstance:
    """Identified problems in a data row.

    Here, the attribute name is the problem type and the lists contain the affected rows
    - incomplete: An existing abbreviation in the row                           1
    - misplaced: A value is set in the wrong column of the csv                  2
    - formatting: The data has inconsistent formatting or wrong standards       3
    - misspelled: Probable wrong spelling in data                               4
    - missing_value: A value is missing from the column                         5
    - validation: Contradicting outside information spotted                     6
    """

    incomplete: list[str] = field(default_factory=list)
    misplaced: list[str] = field(default_factory=list)
    formatting: list[str] = field(default_factory=list)
    misspelled: list[str] = field(default_factory=list)
    missing_value: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Compact string representation like 3[col1,col2]5[col3,col2]7[col1,col3]."""
        return "".join(
            f"{pid.value}{getattr(self, pid.name.lower())}" for pid in ProblemID if getattr(self, pid.name.lower())
        )

    @staticmethod
    def parse_from_str(label: str) -> ProblemInstance:
        """Convert ProblemID string back into a Identified problem object.

        :param label: ProbelmIDs field of data
        :type label: str
        :return: Identified Problems object
        :rtype: ProblemInstance
        """
        ident_probs = ProblemInstance()
        pattern = r"(\d)(\[.+?\])"
        problems = re.findall(pattern, label)

        for prob_id, affected_cols in problems:
            attr = ProblemID(int(prob_id)).name.lower()
            setattr(ident_probs, attr, ast.literal_eval(affected_cols))

        return ident_probs


@dataclass
class Fix:
    """Base class for fixes with reason."""

    value: str | int | None = None
    reason: str | None = None

    def has_value(self) -> bool:
        """Check if Fix has a meaningful value."""
        return self.value is not None


@dataclass
class SolutionInstance:
    """Proposed fixes for each column of a row."""

    hybrid: Fix = field(default_factory=Fix)
    typ: Fix = field(default_factory=Fix)
    nr: Fix = field(default_factory=Fix)
    klassifik: Fix = field(default_factory=Fix)
    name1: Fix = field(default_factory=Fix)
    zeile1: Fix = field(default_factory=Fix)
    plz: Fix = field(default_factory=Fix)
    ort: Fix = field(default_factory=Fix)
    land: Fix = field(default_factory=Fix)
    ustid: Fix = field(default_factory=Fix)
    steuernr: Fix = field(default_factory=Fix)

    def is_empty(self) -> bool:
        """Return True if all Fix fields are empty."""
        return all(not getattr(self, f.name).has_value() for f in fields(self))

    def combine(self, other: SolutionInstance) -> None:
        """Combine two Proposals. Keep self values if present; otherwise take from other."""
        if not isinstance(other, SolutionInstance):
            msg = "Not type of SolutionInstance!"
            raise TypeError(msg)

        for f in fields(self):
            current_fix: Fix = getattr(self, f.name)
            other_fix: Fix = getattr(other, f.name)
            if not current_fix.has_value() and other_fix.has_value():
                setattr(self, f.name, other_fix)

    def apply_proposal(self, row: pd.Series) -> pd.Series:
        """Apply proposal to a df row.

        :param row: Current row where changes are injected into
        :type row: pd.Series
        :return: New row with changes implemented
        :rtype: pd.Series
        """
        if self.is_empty():
            return row

        for f in fields(self):
            fix_obj = getattr(self, f.name)
            if fix_obj.has_value() and f.name in row.index:
                row[f.name] = fix_obj.value

        # maybe reset with: proposal row["SolutionInstance"] = SolutionInstance()
        return row

    def __str__(self) -> str:
        """Pretty-print all fields using only Fix.value."""
        final_string = ""
        for f in fields(self):
            field_fix: Fix = getattr(self, f.name)
            if field_fix.value or field_fix.reason:
                final_string += f"{f.name}['{field_fix.value}','{field_fix.reason}']"
        return final_string

    @staticmethod
    def parse_from_str(label: str) -> SolutionInstance:
        """Parse SolutionInstance from string into python object.

        :param data: SolutionInstance in dict format (from a stringified JSONL file)
        :type data: dict
        :return: Python SolutionInstance objects
        :rtype: SolutionInstance
        """
        fix_proposal = SolutionInstance()
        pattern = r"([a-zA-Z0-9_]+)\[([^,\]]+),([^\]]+)\]"  # col_name1[value, reasons]
        fixes = re.findall(pattern, label)
        found_fixes = {col_name: (value, reason) for col_name, value, reason in fixes}

        for f in fields(SolutionInstance):
            if f.name in found_fixes:
                final_value = found_fixes[f.name][0] or None
                final_reason = found_fixes[f.name][1] or None
                fix = Fix(value=ast.literal_eval(final_value or ""), reason=final_reason)
                setattr(fix_proposal, f.name, fix)
            else:
                setattr(fix_proposal, f.name, Fix(value=None, reason=None))

        return fix_proposal


@dataclass
class MetaDataInstance:
    """Preservation of process events and orders."""

    def __str__(self):
        pass

    @staticmethod
    def parse_from_str(label: str) -> MetaDataInstance:
        pass


class Prompts(StrEnum):
    """Contains all prompts of the problem identification, fixAIs."""

    DETECT_MISPLACED_SYSTEM = """
        You are a data validation expert. Your task is to find values placed in the wrong columns. The correct schema is: {\"hybrid\": \"PERS_#_######\", \"typ\": #, \"nr\": ######, \"klassifik\": \"#\", \"name1\": \"Company/Person\", \"zeile1\": \"Address\", \"plz\": \"Postal Code\", \"ort\": \"City\", \"land\": \"Country\", \"ustid\": \"########\", \"steuernr\": \"########\", \"iln\": \"########\"}"}.
        If you find misplacements, output a JSON object containing the columns needed to be switched!
        """
    DETECT_MISSPELLED = """
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
    FIX_INCOMPLETE = """
    # Role
    You are a German Data Quality Specialist. Your task is to write out any abbreviations!

    # Instructions
    You receive an string with abbreviations. Write out any other abbreviation and return the completed string with double quotes!

    # Input
    """
    FIX_FORMATTING = """
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
    {{"format": "{col_rule}", "data": "{col_value}"}}
    """
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
    EXTRACT_ADDRESS = """You extract addresses from google search snippets. The correct schema is: {\"street\": \"street and street nr\",\"city\": \"city\",\"zip\": \"#####\",\"country\": \"country\"}. If no address is found or the address does not make sense, return an empty JSON object "{}" with no commentary. Respons strictly in JSON!"""
    EXTRACT_KLASSIFIK = """
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

    GENERATE_REGEX = """
    System Role: You are an expert Data Engineer and Regular Expression specialist. 
    Your goal is to analyze data patterns and create precise, efficient, and robust Regular Expressions (Regex).
    Task: I will provide you with 100 sample values extracted from a single data column. 
    You must analyze these samples and generate a single Python-compatible Regular Expression that matches all valid entries in the column while minimizing false positives.

    Requirements for the Output:
    Provide the raw Regex pattern in python format: r"<PATTERN>"
    Handle potential whitespace if the data suggests it (e.g., \\s*).
    Prioritize character classes (e.g., \\d, [A-Z]) over literal matches unless the data is strictly constant.

    Constraints:
    The regex must match at least 98% of the provided valid examples.
    Avoid "wildcard" solutions like .* and be as specific as the data allows.
    Rarly use specific character, just when it actually always appears.

    Input Data:
    """
    