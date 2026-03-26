# Data pre-, or post-processing functions for SherpAI Schemas.

import pandas as pd
from dataclasses import fields

from .schemas import ProblemInstance, SolutionInstance, MetaDataInstance


def parse_dimensions_from_str(df: pd.DataFrame):
    """Turn stringified data dimension instances back into their python objects.

    :param df: DataFrame containing all added data dimensions
    """
    df["ProblemSpace"] = df["ProblemSpace"].apply(ProblemInstance.parse_from_str)
    df["SolutionSpace"] = df["SolutionSpace"].apply(SolutionInstance.parse_from_str)
    df["MetaDataSpace"] = df["MetaDataSpace"].apply(MetaDataInstance.parse_from_str)

    return df


def parse_dimensions_to_str(df: pd.DataFrame):
    """Turn data dimension python objects into their stringified versions.

    :param df: DataFrame containing all added data dimensions
    """
    df["ProblemSpace"] = df["ProblemSpace"].map(str)
    df["SolutionSpace"] = df["SolutionSpace"].map(str)
    df["MetaDataSpace"] = df["MetaDataSpace"].map(str)

    return df


def get_pure_data(data_row: pd.Series) -> pd.Series:
    """Remove all added data dimensions from pd.Series.

    :param data_row: Row of a pd.DataFrame
    :return: Altered pd.Series without 3 added data dimensions.
    """
    allowed_columns = [f.name for f in fields(SolutionInstance)]
    existing_allowed = [col for col in allowed_columns if col in data_row.index]
    return data_row[existing_allowed]
