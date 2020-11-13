# -*- coding: utf-8 -*-
"""Nearest Neighbor Indexing recipe script"""

from dku_plugin_param_loading import load_indexing_recipe_params
from data_loader import DataLoader
from models.base import NearestNeighborSearchAlgorithm

params = load_indexing_recipe_params()
data_loader = DataLoader(params["primary_key_column"], params["feature_columns"])

(vector_ids, vectors) = data_loader.load_df(params["input_df"])

algorithm = NearestNeighborSearchAlgorithm(**params)
print(algorithm)
index_config = algorithm.build_save_index(
    vector_ids=vector_ids, vectors=vectors, folder_path=params["output_folder_path"]
)
params["output_folder"].write_json(filename=algorithm.CONFIG_FILE_NAME, obj=index_config)
