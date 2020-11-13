# -*- coding: utf-8 -*-

import numpy as np
from typing import AnyStr, Dict

import annoy

from models.base import NearestNeighborSearchAlgorithm
from plugin_utils import time_logging


class AnnoyAlgorithm(NearestNeighborSearchAlgorithm):
    """Wrapper class for the Annoy Nearest Neighbor Search algorithm"""

    def __init__(self, **kwargs):
        self.metric = kwargs.get("annoy_metric")
        self.num_trees = int(kwargs.get("annoy_num_trees", 10))
        self.config = {
            "model": self.__str__(),
            "metric": self.metric,
            "num_trees": self.num_trees,
        }  # may be modified by `self.build_save_index`

    def __str__(self):
        return "annoy"

    @time_logging(log_message="Building index on file")
    def build_save_index(self, vectors: np.array, file_path: AnyStr) -> Dict:
        """Initialize index on disk, add each item one-by-one and save to disk"""
        index_config = self.config
        index_config["num_dimensions"] = vectors.shape[1]
        index = annoy.AnnoyIndex(index_config["num_dimensions"], metric=self.metric)
        index.on_disk_build(file_path)
        for i, vector in enumerate(vectors):
            index.add_item(i, vector.tolist())
        index.build(n_trees=self.num_trees)
        return index_config

    def load_index(self, index_file_path: AnyStr):
        self.index = annoy.AnnoyIndex(self.num_dimensions, metric=self.metric)
        self.index.load(index_file_path)

    def lookup_neighbors(self, vectors: np.array, num_neighbors: int = 5) -> np.array:
        """No bulk lookup supported by the library so it has to be done in loop"""
        nns = []
        for vector in vectors:
            nns.append(self.index.get_nns_by_vector(vector, num_neighbors))
        return np.array(nns)
