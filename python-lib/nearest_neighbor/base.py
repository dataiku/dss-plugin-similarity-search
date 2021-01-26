# -*- coding: utf-8 -*-
"""Module to wrap all Nearest Neighbor Search algorithms"""

from typing import AnyStr, Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loader import DataLoader


class NearestNeighborSearch:
    """Base class for all Nearest Neighbor Search algorithms"""

    INDEX_FILE_NAME = "index.nns"
    CONFIG_FILE_NAME = "config.json"
    ARRAY_IDS_FILE_NAME = "vector_ids.npz"
    ARRAYS_FILE_NAME = "vectors.npz"
    INPUT_COLUMN_NAME = "input_id"
    NEIGHBOR_COLUMN_NAME = "neighbor_id"
    DISTANCE_COLUMN_NAME = "distance"
    COLUMN_DESCRIPTIONS = {
        INPUT_COLUMN_NAME: "Unique ID from the input dataset",
        NEIGHBOR_COLUMN_NAME: "Neighbor ID from the pre-computed index",
        DISTANCE_COLUMN_NAME: "Distance with neighbor",
    }

    def __new__(cls, *args, **kwargs):
        """Determine the appropriate algorithm based on the arguments"""
        algorithm = kwargs.get("algorithm")
        if algorithm == "annoy":
            from nearest_neighbor.annoy import Annoy  # noqa

            for i in cls.__subclasses__():
                if i.__name__ == "Annoy":
                    return super().__new__(i)
        elif algorithm == "faiss":
            from nearest_neighbor.faiss import Faiss  # noqa

            for i in cls.__subclasses__():
                if i.__name__ == "Faiss":
                    return super().__new__(i)
        else:
            raise NotImplementedError(f"Algorithm '{algorithm}' is not available")

    def __init__(self, num_dimensions: int, **kwargs):
        self.num_dimensions = num_dimensions

    def get_config(self) -> Dict:
        """Config required to reload the index after initial build"""
        raise NotImplementedError("Get config method not implemented")

    def build_save_index(self, arrays: np.array, index_path: AnyStr) -> None:
        """Add arrays (a.k.a. vectors) to the index and save to disk"""
        raise NotImplementedError("Index building and saving method not implemented")

    def load_index(self, index_file_path: AnyStr) -> None:
        """Load pre-computed index from disk into memory"""
        raise NotImplementedError("Index loading method not implemented")

    def find_neighbors_array(self, arrays: np.array, num_neighbors: int = 5) -> List[List[Tuple]]:
        """Find nearest neighbors of each arrays (a.k.a. vectors) and return pairs of (index, distance)"""
        raise NotImplementedError("Find neighbors method not implemented")

    def find_neighbors_df(
        self,
        df: pd.DataFrame,
        unique_id_column: AnyStr,
        feature_columns: List[AnyStr],
        index_array_ids: np.array,
        num_neighbors: int = 5,
        **kwargs,
    ) -> pd.DataFrame:
        """Find nearest neighbors in a raw pandas DataFrame and format results into a new DataFrame"""
        output_df = pd.DataFrame()
        output_df[self.INPUT_COLUMN_NAME] = df[unique_id_column]
        data_loader = DataLoader(unique_id_column, feature_columns)
        (array_ids, arrays) = data_loader.convert_df_to_array(df, verbose=False)
        if arrays.shape[1] != self.num_dimensions:
            raise ValueError(
                "Incompatible number of dimensions: "
                + f"{self.num_dimensions} in index, {arrays.shape[1]} in feature column(s)"
            )
        output_df["index_distance_pairs"] = self.find_neighbors_array(arrays, num_neighbors)
        output_df = output_df.explode("index_distance_pairs")
        output_df[self.NEIGHBOR_COLUMN_NAME] = output_df["index_distance_pairs"].apply(lambda x: int(x[0]))
        output_df[self.DISTANCE_COLUMN_NAME] = output_df["index_distance_pairs"].apply(lambda x: float(x[1]))
        output_df[self.NEIGHBOR_COLUMN_NAME] = (
            output_df[self.NEIGHBOR_COLUMN_NAME].astype(int).apply(lambda i: index_array_ids[i])
        )  # lookup the original array ids
        del output_df["index_distance_pairs"]
        return output_df
