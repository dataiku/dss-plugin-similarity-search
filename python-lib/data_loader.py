# -*- coding: utf-8 -*-
"""Module to load data into a format usable by Nearest Neighbor Search algorithms"""

import json
import logging
from typing import List, AnyStr, Tuple
from time import perf_counter

import pandas as pd
import numpy as np


class DataLoader:
    """Data loading class to convert numeric/array data from pandas DataFrames into numpy.arrays"""

    MAX_ARRAY_LENGTH = 2 ** 16  # hardcoded limit to keep array size under 65536

    def __init__(self, unique_id_column: AnyStr, feature_columns: List[AnyStr]):
        self.unique_id_column = unique_id_column
        self.feature_columns = feature_columns

    @staticmethod
    def load_array_from_string(string: AnyStr) -> np.array:
        """Attempt to load a stringified list into a numpy.array"""
        parsed = json.loads(string)
        if isinstance(parsed, list):
            array = np.array(parsed).astype(np.float32)
            return array
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

    def convert_df_to_arrays(self, df: pd.DataFrame, verbose: bool = True) -> Tuple[np.array, np.array]:
        """Convert a DataFrame into the array format required by Similarity Search algorithms"""
        start = perf_counter()
        self._validate_df(df)
        if verbose:
            logging.info(
                f"Loading dataframe of {len(df.index)} rows and "
                + f"{len(self.feature_columns)} column(s) into array format..."
            )
        array_ids = df[self.unique_id_column].values
        array_length = self._count_array_length(df)
        arrays = self._load_arrays_from_df(df, array_length)
        if verbose:
            logging.info(
                f"Loading dataframe into array format: dimensions {arrays.shape} "
                + f"loaded in {perf_counter() - start:.2f} seconds.",
            )
        return (array_ids, arrays)

    def _load_arrays_from_df(self, df: pd.DataFrame, array_length: int) -> np.array:
        """Concatenate numeric/array columns of DataFrame into a single numpy.array"""
        arrays = np.empty(shape=(len(df.index), array_length))  # pre-allocate empty array of fixed size
        i = 0
        for column in self.feature_columns:
            column_is_array = df[column].dtype == "object" and df[column].str.startswith("[").all()
            if column_is_array:
                try:
                    column_array = np.stack(df[column].apply(self.load_array_from_string), axis=0)
                    arrays[:, i : (i + column_array.shape[1])] = column_array  # noqa
                    i += column_array.shape[1]
                except ValueError as e:
                    raise ValueError(f"Invalid array data in column '{column}': {e}")
            else:
                try:
                    arrays[:, i] = df[column].values.astype(np.float32)
                    i += 1
                except ValueError as e:
                    raise ValueError(f"Invalid numeric data in column '{column}': {e}")
        arrays = np.ascontiguousarray(arrays[:, :i], dtype=np.float32)
        return arrays

    def _count_array_length(self, df: pd.DataFrame) -> int:
        """Compute the concatenated array length of a DataFrame with multiple numeric/array columns"""
        first_array = self._load_arrays_from_df(df.head(1), self.MAX_ARRAY_LENGTH)
        array_length = first_array.shape[1]
        logging.info(f"Concatenated array length: {array_length}")
        return array_length
