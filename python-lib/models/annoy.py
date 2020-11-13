# -*- coding: utf-8 -*-

from typing import AnyStr, Dict:

import annoy
from models.base import NearestNeighborSearchAlgorithm

import json
import numpy as np
import os


class AnnoyAlgorithm(NearestNeighborSearchAlgorithm):
    """Wrapper class for the Annoy Nearest Neighbor Search algorithm"""

    def __init__(self, **kwargs):
        self.metric = kwargs.get("annoy_metric")
        self.n_trees = int(kwargs.get("annoy_n_trees", 10))
        self.num_dimensions = int(kwargs.get("num_dimensions", 0))
        self.index = None  # will be loaded by `load_index`

    def __str__(self):
        return "annoy"

    def build_index(self, vector_ids: np.array, vectors: np.array, folder_path: AnyStr):
        """
        Create requested index from params.
        Normalise vectors and add vectors to index (no bulk method so one by one).
        Generate configuration file and save alongside the index.
        """
        folder_path = folder.get_path()

        index = annoy.AnnoyIndex(self.num_dimensions, metric=self.metric)
        index.on_disk_build(os.path.join(folder_path, "index.nns"))

        names_dict = {}
        for i, vector in enumerate(vectors):
            index.add_item(i, vector.tolist())
            names_dict[i] = names[i]
        index.build(self._n_trees)

        config = self._create_config(names_dict)
        with open(os.path.join(folder_path, "config.json"), "w") as fp:
            json.dump(config, fp)

    def _get_config(self, vector_ids: np.array):
        """
        Generated needed config to be able to load and query the index.
        """
        config = {
            "model": self.__str__(),
            "index_file": "index.nns",
            "annoy_metric": self.metric,
            "num_dimensions": self.num_dimensions,
            "n_trees": self.n_trees,
            "vector_ids": names_dict,
        }
        return config

    def load_index(self, index_file_path: AnyStr):
        self.index = annoy.AnnoyIndex(self.num_dimensions, metric=self.metric)
        self.index.load(index_file_path)

    def lookup_neighbors(self, vectors, number_of_neighbors=5) -> np.array:
        """No bulk lookup supported by the library so it has to be done in loop"""
        nns = []
        for vector in vectors:
            nns.append(self.index.get_nns_by_vector(vector, number_of_neighbors))
        return np.array(nns)

