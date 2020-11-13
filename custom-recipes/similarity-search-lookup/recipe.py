# -*- coding: utf-8 -*-
"""Nearest Neighbor Search recipe script"""

from dku_plugin_param_loading import load_lookup_recipe_params
from models.base import NearestNeighborSearchAlgorithm
from dku_io_utils import process_dataset_chunks

params = load_lookup_recipe_params()
search_algorithm = NearestNeighborSearchAlgorithm(**params)

search_algorithm.load_index(params["index_file_path"])

process_dataset_chunks(
    input_dataset=params["input_dataset"],
    output_dataset=params["output_dataset"],
    func=search_algorithm.lookup_df,
    num_neighbors=params["num_neighbors"],
)
