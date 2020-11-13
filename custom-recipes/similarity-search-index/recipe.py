# -*- coding: utf-8 -*-
"""Nearest Neighbor Indexing recipe script"""

from dku_plugin_param_loading import load_indexing_recipe_params
from data_loader import DataLoader
from models.base import NearestNeighborSearchAlgorithm

params = load_indexing_recipe_params()
data_loader = DataLoader(params["primary_key_column"], params["feature_columns"])

(vector_ids, vectors) = data_loader.load_df(params["input_df"])

search_algorithm = NearestNeighborSearchAlgorithm()

search_algorithm.build_index(vector_ids=params["vector_ids"], vector=params["vectors"])
search_algorithm.save_index(folder_path=params["output_folder_path"])
