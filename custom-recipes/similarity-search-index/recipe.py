# -*- coding: utf-8 -*-
"""Nearest Neighbor Indexing recipe script"""

from tempfile import NamedTemporaryFile

from dku_plugin_param_loading import load_indexing_recipe_params
from data_loader import DataLoader
from models.base import NearestNeighborSearchAlgorithm
from dku_io_utils import save_array

# Load recipe parameters
params = load_indexing_recipe_params()

# Load data into vector format for indexing
data_loader = DataLoader(params["unique_id_column"], params["feature_columns"])
(vector_ids, vectors) = data_loader.load_df(params["input_df"])

# Build index and save index file to output folder
algorithm = NearestNeighborSearchAlgorithm(num_dimensions=vectors.shape[1], **params)
with NamedTemporaryFile() as tmp:
    algorithm.build_save_index(vectors=vectors, file_path=tmp.name)
    params["output_folder"].upload_stream(algorithm.INDEX_FILE_NAME, tmp)

# Save vector data and indexing config to guarantee reproducibility
save_array(array=vector_ids, path=algorithm.VECTOR_IDS_FILE_NAME, folder=params["output_folder"])
save_array(array=vectors, path=algorithm.VECTORS_FILE_NAME, folder=params["output_folder"])
params["output_folder"].write_json(algorithm.CONFIG_FILE_NAME, algorithm.config)
