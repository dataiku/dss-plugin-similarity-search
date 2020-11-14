# -*- coding: utf-8 -*-
"""Nearest Neighbor Search recipe script"""

from dku_plugin_param_loading import load_search_recipe_params
from similarity_search_algorithms.base import SimilaritySearchAlgorithm
from dku_io_utils import (
    download_file_from_folder_to_tmp,
    load_array_from_folder,
    process_dataset_chunks,
    set_column_descriptions,
)

# Load parameters
params = load_search_recipe_params()

# Load pre-computed index and vector ids
index_config = params["index_folder"].read_json(SimilaritySearchAlgorithm.CONFIG_FILE_NAME)
algorithm = SimilaritySearchAlgorithm(**index_config)
with download_file_from_folder_to_tmp(algorithm.INDEX_FILE_NAME, params["index_folder"]) as tmp:
    algorithm.load_index(tmp.name)
index_vector_ids = load_array_from_folder(algorithm.VECTOR_IDS_FILE_NAME, params["index_folder"])

# Find nearest neighbors in input dataset
process_dataset_chunks(func=algorithm.find_neighbors_df, index_vector_ids=index_vector_ids, **params)

# Add column descriptions to the output dataset
set_column_descriptions(params["output_dataset"], algorithm.COLUMN_DESCRIPTIONS)
