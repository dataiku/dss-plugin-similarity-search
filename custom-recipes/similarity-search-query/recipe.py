# -*- coding: utf-8 -*-
"""Find Nearest Neighbors recipe script"""

from dku_param_loading import load_search_recipe_params
from nearest_neighbor.base import NearestNeighborSearch
from dku_io_utils import (
    download_file_from_folder_to_tmp,
    load_array_from_folder,
    process_dataset_chunks,
    set_column_descriptions,
)

# Load parameters
params = load_search_recipe_params()

# Load pre-computed index and vector ids
index_config = params["index_folder"].read_json(NearestNeighborSearch.CONFIG_FILE_NAME)
nearest_neighbor = NearestNeighborSearch(**index_config)
with download_file_from_folder_to_tmp(nearest_neighbor.INDEX_FILE_NAME, params["index_folder"]) as tmp:
    nearest_neighbor.load_index(tmp.name)
index_vector_ids = load_array_from_folder(nearest_neighbor.VECTOR_IDS_FILE_NAME, params["index_folder"])

# Find nearest neighbors in input dataset
process_dataset_chunks(func=nearest_neighbor.find_neighbors_df, index_vector_ids=index_vector_ids, **params)

# Add column descriptions to the output dataset
set_column_descriptions(params["output_dataset"], nearest_neighbor.COLUMN_DESCRIPTIONS)
