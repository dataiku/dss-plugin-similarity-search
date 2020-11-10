# -*- coding: utf-8 -*-
import annoy
from models.base import NearestNeighborSearchModel

import json
import numpy as np
import os


class AnnoyAlgo(NearestNeighborSearchModel):
    def __init__(self, *args, **kwargs):
        self._metric = kwargs.get("annoy_metric")
        self._search_k = None
        self._n_trees = int(kwargs.get("annoy_n_trees", 10))
        self._dims = int(kwargs.get("dims"))

    def fit_and_save(self, names, vectors, folder):
        """
        Create requested index from params.
        Normalise vectors and add vectors to index (no bulk method so one by one).
        Generate configuration file and save alongside the index.
        """
        folder_path = folder.get_path()

        index = annoy.AnnoyIndex(self._dims, metric=self._metric)
        index.on_disk_build(os.path.join(folder_path, "index.nns"))

        names_dict = {}
        for i, x in enumerate(vectors):
            index.add_item(i, x.tolist())
            names_dict[i] = names[i]
        index.build(self._n_trees)

        config = self._create_config(names_dict)
        with open(os.path.join(folder_path, "config.json"), "w") as fp:
            json.dump(config, fp)

    def _create_config(self, names_dict):
        """
        Generated needed config to be able to load and query the index.
        """
        return json.dumps(
            {
                "model": self.__str__(),
                "index_file": "index.nns",
                "annoy_metric": self._metric,
                "dims": self._dims,
                "n_trees": self._n_trees,
                "names_dict": names_dict,
            }
        )

    def _load(self, index_file_path):
        self.index = annoy.AnnoyIndex(self._dims, metric=self._metric)
        self.index.load(index_file_path)
        return self

    def _find_near_neighbors(self, vectors, number_of_neighbors=5):
        """
        No bulk lookup supported by the library so it has to be done in loop.
        """
        nns = []
        for vector in vectors:
            nns.append(self.index.get_nns_by_vector(vector, number_of_neighbors))
        return np.array(nns)

    def __str__(self):
        return "annoy"
