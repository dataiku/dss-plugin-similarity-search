# -*- coding: utf-8 -*-

import numpy as np
from typing import AnyStr

import faiss

from models.base import NearestNeighborSearchAlgorithm
from plugin_utils import time_logging


class FaissAlgorithm(NearestNeighborSearchAlgorithm):
    """Wrapper class for the Faiss Nearest Neighbor Search algorithm"""

    def __init__(self, num_dimensions: int, **kwargs):
        self.num_dimensions = num_dimensions
        self.index_type = kwargs.get("faiss_index_type")
        self.config = {"model": "faiss", "num_dimensions": self.num_dimensions, "index_type": self.index_type}
        if self.index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.num_dimensions)
        elif self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.num_dimensions)
        elif self.index_type == "IndexLSH":
            self.lsh_num_bits = int(kwargs.get("faiss_lsh_num_bits", 4))
            self.index = faiss.IndexLSH(self.num_dimensions, self.lsh_num_bits)
            self.config["lsh_num_bits"] = self.lsh_num_bits
        else:
            raise NotImplementedError(f"Faiss index '{self.index_type}' not implemented'")

    @time_logging(log_message="Building Faiss index on file")
    def build_save_index(self, vectors: np.array, file_path: AnyStr) -> None:
        """Initialize index on disk, add vectors by batch and save to disk"""
        if self.index.is_trained:
            self.index.add(vectors)
        else:
            raise NotImplementedError("Faiss training methods not implemented")
        faiss.write_index(self.index, file_path)

    def load_index(self, file_path: AnyStr) -> None:
        """Load saved index into memory"""
        self.index.load(file_path)

    def lookup_neighbors(self, vectors: np.array, num_neighbors: int = 5) -> np.array:
        """No bulk lookup supported by the library so it has to be done in loop"""
        nns = []
        for vector in vectors:
            nns.append(self.index.get_nns_by_vector(vector, num_neighbors))
        return np.array(nns)
