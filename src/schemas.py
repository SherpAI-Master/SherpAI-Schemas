"""All schemas and classes used for organization."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field, fields
from enum import Enum

import pandas as pd


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
