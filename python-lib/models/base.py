# -*- coding: utf-8 -*-

from typing import AnyStr, Dict

import numpy as np


class NearestNeighborSearchAlgorithm:
    """Base class for all Nearest Neighbor Search algorithms"""

    INDEX_FILE_NAME = "index.nns"
    CONFIG_FILE_NAME = "config.json"
    VECTOR_IDS_FILE_NAME = "vector_ids.npz"
    VECTORS_FILE_NAME = "vectors.npz"

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

    def __str__(self) -> AnyStr:
        return self.name

    def build_save_index(self, vector_ids: np.array, vectors: np.array, file_path: AnyStr) -> Dict:
        """Algorithm-specific method to specify for each implementation"""
        raise NotImplementedError("Index building and saving method not implemented")

    def load_index(self, index_file_path: AnyStr) -> None:
        """Algorithm-specific method to specify for each implementation"""
        raise NotImplementedError("Index loading method not implemented")

    def lookup_neighbors(self, vectors: np.array, num_neighbors: int = 5) -> np.array:
        """Algorithm-specific method to specify for each implementation"""
        raise NotImplementedError("Index building and saving method not implemented")
