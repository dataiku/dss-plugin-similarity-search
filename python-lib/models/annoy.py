# -*- coding: utf-8 -*-

import numpy as np
from typing import AnyStr, Dict

import annoy
from tqdm import tqdm

from models.base import SimilaritySearchAlgorithm
from plugin_utils import time_logging


class AnnoyAlgorithm(SimilaritySearchAlgorithm):
    """Wrapper class for the Annoy Similarity Search algorithm"""

    def __init__(self, num_dimensions: int, **kwargs):
        self.num_dimensions = num_dimensions
        self.annoy_metric = kwargs.get("annoy_metric")
        self.annoy_num_trees = int(kwargs.get("annoy_num_trees", 10))
        self.index = annoy.AnnoyIndex(self.num_dimensions, metric=self.annoy_metric)

    def __str__(self):
        return "annoy"

    def get_config(self) -> Dict:
        return {
            "algorithm": self.__str__(),
            "num_dimensions": self.num_dimensions,
            "annoy_metric": self.annoy_metric,
            "annoy_num_trees": self.annoy_num_trees,
        }

    @time_logging(log_message=f"Building index and saving to disk")
    def build_save_index(self, vectors: np.array, file_path: AnyStr) -> None:
        """Initialize index on disk, add each item one-by-one and save to disk"""
        self.index.on_disk_build(file_path)
        for i, vector in enumerate(tqdm(vectors, mininterval=1.0)):
            self.index.add_item(i, vector.tolist())
        self.index.build(n_trees=self.annoy_num_trees)

    @time_logging(log_message="Loading pre-computed index from disk")
    def load_index(self, file_path: AnyStr) -> None:
        """Load saved index into memory"""
        self.index.load(file_path)

    def lookup_neighbors(self, vectors: np.array, num_neighbors: int = 5) -> np.array:
        """No bulk lookup supported by the library so it has to be done in loop"""
        nns = []
        for vector in vectors:
            nns.append(self.index.get_nns_by_vector(vector, num_neighbors))
        return np.array(nns)
