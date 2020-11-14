# -*- coding: utf-8 -*-

import numpy as np
from typing import AnyStr

import annoy
from tqdm import tqdm

from models.base import NearestNeighborSearchAlgorithm
from plugin_utils import time_logging


class AnnoyAlgorithm(NearestNeighborSearchAlgorithm):
    """Wrapper class for the Annoy Nearest Neighbor Search algorithm"""

    def __init__(self, num_dimensions: int, **kwargs):
        self.num_dimensions = num_dimensions
        self.metric = kwargs.get("annoy_metric")
        self.index = annoy.AnnoyIndex(self.num_dimensions, metric=self.metric)
        self.num_trees = int(kwargs.get("annoy_num_trees", 10))
        self.config = {
            "model": "annoy",
            "num_dimensions": self.num_dimensions,
            "metric": self.metric,
            "num_trees": self.num_trees,
        }

    @time_logging(log_message="Building Annoy index on file")
    def build_save_index(self, vectors: np.array, file_path: AnyStr) -> None:
        """Initialize index on disk, add each item one-by-one and save to disk"""
        self.index.on_disk_build(file_path)
        for i, vector in enumerate(tqdm(vectors, mininterval=1.0)):
            self.index.add_item(i, vector.tolist())
        self.index.build(n_trees=self.num_trees)

    def load_index(self, file_path: AnyStr) -> None:
        """Load saved index into memory"""
        self.index.load(file_path)

    def lookup_neighbors(self, vectors: np.array, num_neighbors: int = 5) -> np.array:
        """No bulk lookup supported by the library so it has to be done in loop"""
        nns = []
        for vector in vectors:
            nns.append(self.index.get_nns_by_vector(vector, num_neighbors))
        return np.array(nns)
