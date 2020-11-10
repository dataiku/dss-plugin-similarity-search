# -*- coding: utf-8 -*-
"""Module to load data using the Dataiku API"""

from typing import List, AnyStr

import dataiku

from nns_utils import reshape_values


class DataLoader:
    def __init__(self, dataset: dataiku.Dataset, unique_id_column: AnyStr, feature_columns: List[AnyStr]):
        self.dataset = dataset
        self.unique_id_column = unique_id_column
        self.feature_columns = feature_columns

    def load(self):
        """TODO"""
        df = self._dataset_name.get_dataframe(columns=[self._unique_column, self._embedding_column])
        if len(df[self._unique_column].unique()) == len(df[self._unique_column].tolist()):
            df = df[df[self._embedding_column].notnull()]
            vectors = reshape_values(df[self._embedding_column])
            names = df[self._unique_column].tolist()
            return names, vectors
        else:
            raise ValueError("The values in column {} are not unique".format(self._unique_column))
