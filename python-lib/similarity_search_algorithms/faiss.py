# -*- coding: utf-8 -*-

import numpy as np
from typing import AnyStr, Dict, List

import faiss

from similarity_search_algorithms.base import SimilaritySearchAlgorithm
from plugin_utils import time_logging


class Faiss(SimilaritySearchAlgorithm):
    """Wrapper class for the Faiss Similarity Search algorithm"""

    def __init__(self, num_dimensions: int, **kwargs):
        super().__init__(num_dimensions)
        self.faiss_index_type = kwargs.get("faiss_index_type")
        self.faiss_lsh_num_bits = int(kwargs.get("faiss_lsh_num_bits", 4))
        if self.faiss_index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(self.num_dimensions)
        elif self.faiss_index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.num_dimensions)
        elif self.faiss_index_type == "IndexLSH":
            self.index = faiss.IndexLSH(self.num_dimensions, self.faiss_lsh_num_bits)
        else:
            raise NotImplementedError(f"Faiss index '{self.index_type}' not implemented'")

    def __str__(self):
        return "faiss"

    def get_config(self) -> Dict:
        return {
            "algorithm": self.__str__(),
            "num_dimensions": self.num_dimensions,
            "faiss_index_type": self.faiss_index_type,
            "faiss_lsh_num_bits": self.faiss_lsh_num_bits,
        }

    @time_logging(log_message="Building index and saving to disk")
    def build_save_index(self, vectors: np.array, index_path: AnyStr) -> None:
        if self.index.is_trained:
            self.index.add(vectors)
        else:
            raise NotImplementedError("Faiss training methods not implemented")
        faiss.write_index(self.index, index_path)

    @time_logging(log_message="Loading pre-computed index from disk")
    def load_index(self, file_path: AnyStr) -> None:
        self.index = faiss.read_index(file_path)

    def find_neighbors_vector(self, vectors: np.array, num_neighbors: int = 5) -> List:
        (_, neighbors) = self.index.search(vectors, num_neighbors)
        return neighbors.tolist()
