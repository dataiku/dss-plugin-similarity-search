# -*- coding: utf-8 -*-
"""Nearest Neighbor Indexing recipe script"""

from tempfile import NamedTemporaryFile

from dku_plugin_param_loading import load_indexing_recipe_params
from data_loader import DataLoader
from similarity_search_algorithms.base import SimilaritySearchAlgorithm
from dku_io_utils import save_array_to_folder

# Load parameters
params = load_indexing_recipe_params()

# Load data into vector format for indexing
columns = [params["unique_id_column"]] + params["feature_columns"]
input_df = params["input_dataset"].get_dataframe(columns=columns, infer_with_pandas=False)
data_loader = DataLoader(params["unique_id_column"], params["feature_columns"])
(vector_ids, vectors) = data_loader.convert_df_to_vectors(input_df)

# Build index and save index file to output folder
algorithm = SimilaritySearchAlgorithm(num_dimensions=vectors.shape[1], **params)
with NamedTemporaryFile() as tmp:
    algorithm.build_save_index(vectors=vectors, index_path=tmp.name)
    params["index_folder"].upload_stream(algorithm.INDEX_FILE_NAME, tmp)

# Save vector data and indexing config to guarantee reproducibility
save_array_to_folder(array=vector_ids, path=algorithm.VECTOR_IDS_FILE_NAME, folder=params["index_folder"])
save_array_to_folder(array=vectors, path=algorithm.VECTORS_FILE_NAME, folder=params["index_folder"])
params["index_folder"].write_json(algorithm.CONFIG_FILE_NAME, algorithm.get_config())
