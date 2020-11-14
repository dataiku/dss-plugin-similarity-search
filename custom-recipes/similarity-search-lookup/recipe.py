# -*- coding: utf-8 -*-
"""Nearest Neighbor Search recipe script"""

import os

from dku_plugin_param_loading import load_lookup_recipe_params
from data_loader import DataLoader
from models.base import SimilaritySearchAlgorithm

# Load parameters
params = load_lookup_recipe_params()

# Load pre-computed index config
index_config = params["index_folder"].read_json(SimilaritySearchAlgorithm.CONFIG_FILE_NAME)
algorithm = SimilaritySearchAlgorithm(**index_config)
saved_index_path = os.path.join(params["index_folder_path"], SimilaritySearchAlgorithm.INDEX_FILE_NAME)
algorithm.load_index(saved_index_path)

# Load data into vector format for indexing
data_loader = DataLoader(params["unique_id_column"], params["feature_columns"])
(vector_ids, vectors) = data_loader.load_df(params["input_df"])

# TODO
print("foo")
