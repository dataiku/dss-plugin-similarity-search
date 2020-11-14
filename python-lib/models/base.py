# -*- coding: utf-8 -*-

from typing import AnyStr, Dict, List

import numpy as np
import pandas as pd

from data_loader import DataLoader


class SimilaritySearchAlgorithm:
    """Base class for all Similarity Search algorithms"""

    INDEX_FILE_NAME = "index.nns"
    CONFIG_FILE_NAME = "config.json"
    VECTOR_IDS_FILE_NAME = "vector_ids.npz"
    VECTORS_FILE_NAME = "vectors.npz"
    QUERY_COLUMN_NAME = "query_id"
    NEIGHBOR_COLUMN_NAME = "neighbor_id"
    COLUMN_DESCRIPTIONS = {
        QUERY_COLUMN_NAME: "Unique ID from the input dataset",
        NEIGHBOR_COLUMN_NAME: "Neighbor ID from the pre-computed index",
    }

    def __new__(cls, *args, **kwargs):
        """Determine based on the arguments the appropriate algorithm"""
        algorithm = kwargs.get("algorithm")
        if algorithm == "annoy":
            from models.annoy import AnnoyAlgorithm  # noqa

            for i in cls.__subclasses__():
                if i.__name__ == "AnnoyAlgorithm":
                    return super().__new__(i)
        elif algorithm == "faiss":
            from models.faiss import FaissAlgorithm  # noqa

            for i in cls.__subclasses__():
                if i.__name__ == "FaissAlgorithm":
                    return super().__new__(i)
        else:
            raise NotImplementedError(f"Algorithm '{algorithm}' is not available")

    def __init__(self, num_dimensions: int, **kwargs):
        self.num_dimensions = num_dimensions

    def get_config(self) -> Dict:
        """Config required to reload the index after initial build"""
        raise NotImplementedError("Get config method not implemented")

    def build_save_index(self, vector_ids: np.array, vectors: np.array, file_path: AnyStr) -> None:
        """Add vectors to the index and save to disk"""
        raise NotImplementedError("Index building and saving method not implemented")

    def load_index(self, index_file_path: AnyStr) -> None:
        """Load pre-computed index from disk into memory"""
        raise NotImplementedError("Index loading method not implemented")

    def find_neighbors_vector(self, vectors: np.array, num_neighbors: int = 5) -> List:
        """Search for nearest neighbors of each vector and return their indices"""
        raise NotImplementedError("Find neighbors method not implemented")

    def find_neighbors_df(
        self,
        df: pd.DataFrame,
        unique_id_column: AnyStr,
        feature_columns: List[AnyStr],
        num_neighbors: int = 5,
        **kwargs,
    ) -> pd.DataFrame:
        """Algorithm-specific method to specify for each implementation"""
        output_df = pd.DataFrame()
        output_df[self.QUERY_COLUMN_NAME] = df[unique_id_column]
        data_loader = DataLoader(unique_id_column, feature_columns)
        (vector_ids, vectors) = data_loader.load_df(df, verbose=False)
        if vectors.shape[1] != self.num_dimensions:
            raise ValueError(
                "Incompatible number of dimensions: "
                + f"{self.num_dimensions} in index, {vectors.shape[1]} in input dataset"
            )
        output_df[self.NEIGHBOR_COLUMN_NAME] = self.find_neighbors_vector(vectors, num_neighbors)
        return output_df
