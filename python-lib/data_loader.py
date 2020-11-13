# -*- coding: utf-8 -*-
"""Module to load data into a format usable by models"""

import json
import logging
from typing import List, AnyStr, Tuple
from time import time

import pandas as pd
import numpy as np


class DataLoader:

    MAX_VECTOR_LENGTH = 2 ** 16  # hardcoded limit to keep vectors size under 65536

    def __init__(self, primary_key_column: AnyStr, feature_columns: List[AnyStr]):
        self.primary_key_column = primary_key_column
        self.feature_columns = feature_columns

    @staticmethod
    def load_vector_from_string(string: AnyStr):
        parsed = json.loads(string)
        if isinstance(parsed, list):
            vector = np.array(parsed).astype(np.float)
            return vector
        else:
            raise ValueError(f"string '{string}' is not a list")

    def _validate_df(self, df: pd.DataFrame):
        if len(df.index) == 0:
            raise ValueError("Input dataset is empty")
        if not df[self.primary_key_column].is_unique:
            raise ValueError(f"Values in the primary key column '{self.primary_key_column}' should be unique")
        for column in self.feature_columns:
            if df[column].isnull().values.any():
                raise ValueError(f"Empty values in column '{column}'")

    def load_df(self, df: pd.DataFrame) -> Tuple[np.array, np.array]:
        """Load a dataframe into the vector format required by Similarity Search algorithms"""
        start = time()
        self._validate_df(df)
        logging.info(
            f"Loading dataframe of {len(df.index)} rows and "
            + f"{len(self.feature_columns)} column(s) into vector format..."
        )
        vector_ids = df[self.primary_key_column].values
        vectors = np.empty(shape=(len(df.index), self.MAX_VECTOR_LENGTH))
        i = 0
        for column in sorted(self.feature_columns):
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
                    vectors[:, i] = df[column].values.astype(np.float)
                    i += 1
                except ValueError as e:
                    raise ValueError(f"Invalid numeric data in column '{column}': {e}")
        vectors = vectors[:, :i]
        logging.info(
            f"Loading dataframe into vector format: array of dimensions {vectors.shape} "
            + f"loaded in {time() - start:.2f} seconds.",
        )
        return (vector_ids, vectors)
