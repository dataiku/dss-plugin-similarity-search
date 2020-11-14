# -*- coding: utf-8 -*-
"""Nearest Neighbor Search recipe script"""

import os

from dku_plugin_param_loading import load_search_recipe_params
from models.base import SimilaritySearchAlgorithm
from dku_io_utils import process_dataset_chunks, set_column_descriptions

# Load parameters
params = load_search_recipe_params()

# Load pre-computed index
index_config = params["index_folder"].read_json(SimilaritySearchAlgorithm.CONFIG_FILE_NAME)
algorithm = SimilaritySearchAlgorithm(**index_config)
saved_index_path = os.path.join(params["index_folder_path"], SimilaritySearchAlgorithm.INDEX_FILE_NAME)
algorithm.load_index(saved_index_path)

# Find nearest neighbors in input dataset
process_dataset_chunks(func=algorithm.find_neighbors_df, **params)

# Add column descriptions to the output dataset
set_column_descriptions(params["output_dataset"], algorithm.COLUMN_DESCRIPTIONS)
