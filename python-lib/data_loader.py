# -*- coding: utf-8 -*-
"""Module to load data into a format usable by models"""

from typing import List, AnyStr

import pandas as pd

from nns_utils import reshape_values


class DataLoader:

    MAX_VECTOR_LENGTH = 2 ** 16  # hardcoded limit for vectors larger than 65536

    def __init__(self, input_df: pd.DataFrame, unique_id_column: AnyStr, feature_columns: List[AnyStr]):
        self.input_df = input_df
        self.unique_id_column = unique_id_column
        self.feature_columns = feature_columns

    def load(self):
        """TODO"""
        if len(df[self._unique_column].unique()) == len(df[self._unique_column].tolist()):
            df = df[df[self._embedding_column].notnull()]
            vectors = reshape_values(df[self._embedding_column])
            names = df[self._unique_column].tolist()
            return names, vectors
        else:
            raise ValueError("The values in column {} are not unique".format(self._unique_column))
