"""Tests for the pre-/post-processing helpers in functions.py."""

import pandas as pd
import pytest
from conftest import ERP_COLUMNS, make_finding

from sherpai_schemas import (
    ProblemType,
    SherpAIInstance,
    get_pure_data,
    parse_dimensions_from_str,
    parse_dimensions_to_str,
    smart_cast,
)

# --------------------------------------------------------------------------
# smart_cast
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("['ort>zeile1']", ["ort>zeile1"]),
        ("[]", []),
        ("{'a': 1}", {"a": 1}),
        ("42", 42),
        ("'Stuttgart'", "Stuttgart"),
    ],
)
def test_smart_cast_parses_python_literals(raw, expected):
    assert smart_cast(raw, return_on_fail=[]) == expected


def test_smart_cast_rewrites_json_booleans():
    assert smart_cast('{"a": true, "b": false}', return_on_fail={}) == {
        "a": True,
        "b": False,
    }


def test_smart_cast_returns_the_fallback_on_unparsable_input():
    assert smart_cast("not python at all", return_on_fail=[]) == []
    assert smart_cast("['unclosed'", return_on_fail={"x": 1}) == {"x": 1}


def test_smart_cast_returns_the_raw_value_when_the_fallback_is_none():
    # `return_on_fail if return_on_fail is not None else value` -- a None
    # fallback means "hand the caller the original string back".
    assert smart_cast("not python at all", return_on_fail=None) == "not python at all"


@pytest.mark.parametrize("value", [123, ["already", "a", "list"], None])
def test_smart_cast_passes_non_strings_straight_through(value):
    assert smart_cast(value, return_on_fail=[]) == value


def test_smart_cast_on_empty_string_uses_the_fallback():
    assert smart_cast("", return_on_fail=[]) == []


# --- Characterization test: current behavior, not desired behavior ---------


def test_smart_cast_corrupts_words_containing_true_or_false():
    # FIXME: the boolean rewrite is a bare re.sub with no word boundary, so it
    # fires inside ordinary words too -- "construe" becomes "consTrue". Adding
    # \b anchors would fix it; flip this assertion when that happens.
    assert smart_cast("['construe']", return_on_fail=[]) == ["consTrue"]


# --------------------------------------------------------------------------
# get_pure_data
# --------------------------------------------------------------------------


def test_get_pure_data_drops_the_pipeline_bookkeeping_column(data_row):
    row = data_row.copy()
    row["SherpAISpace"] = str(SherpAIInstance())

    assert "SherpAISpace" not in get_pure_data(row).index


def test_get_pure_data_keeps_the_erp_columns(data_row):
    pure = get_pure_data(data_row)

    assert set(pure.index) == set(ERP_COLUMNS)
    assert pure["ort"] == "Stuttgart"


def test_get_pure_data_returns_the_allow_list_order_not_the_row_order():
    scrambled = pd.Series({"ort": "Stuttgart", "hybrid": "PERS_1_42", "typ": "1"})

    # The order is what reaches the LLM as JSON, so it is part of the contract.
    assert list(get_pure_data(scrambled).index) == ["hybrid", "typ", "ort"]


def test_get_pure_data_tolerates_missing_columns():
    sparse = pd.Series({"ort": "Stuttgart"})

    assert list(get_pure_data(sparse).index) == ["ort"]


def test_get_pure_data_on_a_row_with_no_known_columns_is_empty():
    assert get_pure_data(pd.Series({"unrelated": 1})).empty


def test_get_pure_data_excludes_iln():
    # iln has a FormattingRules pattern but is deliberately not in the
    # allow-list, so it never reaches the LLM.
    row = pd.Series({"ort": "Stuttgart", "iln": "4012345678901"})

    assert "iln" not in get_pure_data(row).index


# --------------------------------------------------------------------------
# parse_dimensions_from_str / parse_dimensions_to_str
# --------------------------------------------------------------------------


def test_parse_dimensions_from_str_hydrates_instances():
    frame = pd.DataFrame({"ort": ["Stuttgart"], "SherpAISpace": [str(SherpAIInstance())]})

    hydrated = parse_dimensions_from_str(frame)

    assert isinstance(hydrated.loc[0, "SherpAISpace"], SherpAIInstance)


def test_parse_dimensions_to_str_stringifies_instances():
    frame = pd.DataFrame({"ort": ["Stuttgart"], "SherpAISpace": [SherpAIInstance()]})

    flattened = parse_dimensions_to_str(frame)

    assert isinstance(flattened.loc[0, "SherpAISpace"], str)


def test_dimension_round_trip_preserves_findings(identity):
    instance = SherpAIInstance(
        findings=[make_finding(identity, problem_type=ProblemType.FORMATTING)]
    )
    frame = pd.DataFrame({"SherpAISpace": [instance]})

    restored = parse_dimensions_from_str(parse_dimensions_to_str(frame))

    assert restored.loc[0, "SherpAISpace"] == instance


def test_parse_dimensions_from_str_handles_empty_cells():
    frame = pd.DataFrame({"SherpAISpace": ["", str(SherpAIInstance())]})

    hydrated = parse_dimensions_from_str(frame)

    assert all(isinstance(cell, SherpAIInstance) for cell in hydrated["SherpAISpace"])
    assert hydrated.loc[0, "SherpAISpace"].findings == []


def test_parse_dimensions_leaves_other_columns_alone(data_row):
    frame = pd.DataFrame([{**ERP_COLUMNS, "SherpAISpace": str(SherpAIInstance())}])

    hydrated = parse_dimensions_from_str(frame)

    assert hydrated.loc[0, "ort"] == "Stuttgart"
    assert hydrated.loc[0, "plz"] == "70173"
