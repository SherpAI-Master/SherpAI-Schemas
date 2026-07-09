# Data pre-, or post-processing functions for SherpAI Schemas.

import pandas as pd
import re
import ast

from typing import Any

from .schemas import SherpAIInstance


def parse_dimensions_from_str(df: pd.DataFrame):
    """Turn stringified data dimension instances back into their python objects.

    :param df: DataFrame containing all added data dimensions
    """
    df["SherpAISpace"] = df["SherpAISpace"].apply(SherpAIInstance.parse_from_str)
    return df


def parse_dimensions_to_str(df: pd.DataFrame):
    """Turn data dimension python objects into their stringified versions.

    :param df: DataFrame containing all added data dimensions
    """
    df["SherpAISpace"] = df["SherpAISpace"].map(str)
    return df


def get_pure_data(data_row: pd.Series) -> pd.Series:
    """Remove all added data dimensions from pd.Series.

    :param data_row: Row of a pd.DataFrame
    :return: Altered pd.Series without 3 added data dimensions.
    """
    allowed_columns = [
        "hybrid",
        "typ",
        "nr",
        "klassifik",
        "name1",
        "zeile1",
        "plz",
        "ort",
        "land",
        "ustid",
        "steuernr",
    ]

    existing_allowed = [col for col in allowed_columns if col in data_row.index]
    return data_row[existing_allowed]


def smart_cast(value: str, return_on_fail: Any) -> any:
    """Turn LLM response into python literals.

    :param value: LLM response
    :param return_on_fail: Default object when failed
    """
    if not isinstance(value, str):
        print(f"Warning: Input not string{value}")
        return value
    try:
        python_value = re.sub("true", "True", value)
        python_value = re.sub("false", "False", python_value)
        return ast.literal_eval(python_value)
    except (ValueError, SyntaxError):
        return return_on_fail if return_on_fail is not None else value
