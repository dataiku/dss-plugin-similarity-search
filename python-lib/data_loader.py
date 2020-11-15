# -*- coding: utf-8 -*-
"""Module to load data into a format usable by Nearest Neighbor Search algorithms"""

import json
import logging
from typing import List, AnyStr, Tuple
from time import perf_counter

import pandas as pd
import numpy as np


class DataLoader:
    """Data loading class to convert numeric/vector data from pandas DataFrames into numpy.arrays"""

    MAX_VECTOR_LENGTH = 2 ** 16  # hardcoded limit to keep vectors size under 65536

    def __init__(self, unique_id_column: AnyStr, feature_columns: List[AnyStr]):
        self.unique_id_column = unique_id_column
        self.feature_columns = feature_columns

    @staticmethod
    def load_vector_from_string(string: AnyStr) -> np.array:
        """Attempt to load a stringified vector into a numpy.array"""
        parsed = json.loads(string)
        if isinstance(parsed, list):
            vector = np.array(parsed).astype(np.float32)
            return vector
        else:
            raise ValueError(f"string '{string}' is not a list")

    def _validate_df(self, df: pd.DataFrame):
        """Make sure that the DataFrame is not empty and has a unique ID column"""
        if len(df.index) == 0:
            raise ValueError("Input dataset is empty")
        if not df[self.unique_id_column].is_unique:
            raise ValueError(f"Values in the unique ID column '{self.unique_id_column}' should be unique")
        for column in self.feature_columns:
            if df[column].isnull().values.any():
                raise ValueError(f"Empty values in column '{column}'")

    def convert_df_to_vectors(self, df: pd.DataFrame, verbose: bool = True) -> Tuple[np.array, np.array]:
        """Convert a DataFrame into the vector format required by Similarity Search algorithms"""
        start = perf_counter()
        self._validate_df(df)
        if verbose:
            logging.info(
                f"Loading dataframe of {len(df.index)} rows and "
                + f"{len(self.feature_columns)} column(s) into vector format..."
            )
        vector_ids = df[self.unique_id_column].values
        vectors = np.empty(shape=(len(df.index), self.MAX_VECTOR_LENGTH))
        i = 0
        for column in self.feature_columns:
            column_is_vector = df[column].dtype == "object" and df[column].str.startswith("[").all()
            if column_is_vector:
                try:
                    column_array = np.stack(df[column].apply(self.load_vector_from_string), axis=0)
                    vectors[:, i : (i + column_array.shape[1])] = column_array  # noqa
                    i += column_array.shape[1]
                except ValueError as e:
                    raise ValueError(f"Invalid vector data in column '{column}': {e}")
            else:
                try:
                    vectors[:, i] = df[column].values.astype(np.float32)
                    i += 1
                except ValueError as e:
                    raise ValueError(f"Invalid numeric data in column '{column}': {e}")
        vectors = np.ascontiguousarray(vectors[:, :i], dtype=np.float32)
        if verbose:
            logging.info(
                f"Loading dataframe into vector format: array of dimensions {vectors.shape} "
                + f"loaded in {perf_counter() - start:.2f} seconds.",
            )
        return (vector_ids, vectors)
