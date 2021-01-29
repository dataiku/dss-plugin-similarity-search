# -*- coding: utf-8 -*-
"""Module for the Annoy Nearest Neighbor Search algorithm"""

import logging

import numpy as np
from typing import AnyStr, Dict, List, Tuple

import annoy
from tqdm import tqdm

from nearest_neighbor.base import NearestNeighborSearch
from utils import time_logging


class Annoy(NearestNeighborSearch):
    """Wrapper class for the Annoy Nearest Neighbor Search algorithm"""

    def __init__(self, num_dimensions: int, **kwargs):
        super().__init__(num_dimensions)
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

    @time_logging(log_message="Building index and saving to disk")
    def build_save_index(self, arrays: np.array, index_path: AnyStr) -> None:
        self.index.on_disk_build(index_path)
        for i, array in enumerate(tqdm(arrays, mininterval=1.0)):
            self.index.add_item(i, array.tolist())
        self.index.build(n_trees=self.annoy_num_trees)
        logging.info(f"Index file path: {index_path}")

    @time_logging(log_message="Loading pre-computed index")
    def load_index(self, file_path: AnyStr) -> None:
        self.index.load(file_path)

    def find_neighbors_array(self, arrays: np.array, num_neighbors: int = 5) -> List[List[Tuple]]:
        output = []
        for array in arrays:
            (neighbors, distances) = self.index.get_nns_by_vector(array, num_neighbors, include_distances=True)
            output.append(list(zip(neighbors, distances)))
        return output
