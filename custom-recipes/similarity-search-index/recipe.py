# -*- coding: utf-8 -*-
"""Nearest Neighbor Indexing recipe script"""

from dku_plugin_param_loading import load_indexing_recipe_params
from models.base import NearestNeighborSearchAlgorithm

params = load_indexing_recipe_params()
search_algorithm = NearestNeighborSearchAlgorithm(**params)

search_algorithm.build_index(vector_ids=params["vector_ids"], vector=params["vectors"])
search_algorithm.save_index(folder_path=params["output_folder_path"])
