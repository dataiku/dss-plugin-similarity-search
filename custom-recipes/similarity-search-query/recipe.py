# -*- coding: utf-8 -*-
"""Find Nearest Neighbors recipe script"""

import os

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

# Load pre-computed index and arra ids
config_file_path = os.path.join(params["folder_partition_root"], NearestNeighborSearch.CONFIG_FILE_NAME)
index_config = params["index_folder"].read_json(config_file_path)
nearest_neighbor = NearestNeighborSearch(**index_config)
index_file_path = os.path.join(params["folder_partition_root"], nearest_neighbor.INDEX_FILE_NAME)
with download_file_from_folder_to_tmp(index_file_path, params["index_folder"]) as tmp:
    nearest_neighbor.load_index(tmp.name)
array_ids_file_path = os.path.join(params["folder_partition_root"], nearest_neighbor.ARRAY_IDS_FILE_NAME)
index_array_ids = load_array_from_folder(array_ids_file_path, params["index_folder"])

# Find nearest neighbors in input dataset
process_dataset_chunks(func=nearest_neighbor.find_neighbors_df, index_array_ids=index_array_ids, **params)

# Add column descriptions to the output dataset
set_column_descriptions(params["output_dataset"], nearest_neighbor.COLUMN_DESCRIPTIONS)
