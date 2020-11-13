# -*- coding: utf-8 -*-
"""Nearest Neighbor Indexing recipe script"""

from tempfile import NamedTemporaryFile

import numpy as np

from dku_plugin_param_loading import load_indexing_recipe_params
from data_loader import DataLoader
from models.base import NearestNeighborSearchAlgorithm

# Load recipe parameters
params = load_indexing_recipe_params()

# Load data into vector format for indexing
data_loader = DataLoader(params["primary_key_column"], params["feature_columns"])
(vector_ids, vectors) = data_loader.load_df(params["input_df"])

# Build index and save index file to output folder
algorithm = NearestNeighborSearchAlgorithm(**params)
with NamedTemporaryFile() as tmp:
    index_config = algorithm.build_save_index(vectors=vectors, file_path=tmp.name)
    params["output_folder"].upload_stream(algorithm.INDEX_FILE_NAME, tmp)

# Save vector data and indexing config to guarantee reproducibility
with NamedTemporaryFile() as tmp:
    np.savez_compressed(tmp, vectors=vectors, vector_ids=vectors)
    tmp.seek(0)  # Oh, take me back to the start
    params["output_folder"].upload_stream(algorithm.VECTORS_FILE_NAME, tmp)
    params["output_folder"].write_json(algorithm.CONFIG_FILE_NAME, index_config)
