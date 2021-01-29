# -*- coding: utf-8 -*-
"""Build Nearest Neighbor Search index recipe script"""

import os
from tempfile import NamedTemporaryFile

from dku_param_loading import load_indexing_recipe_params
from data_loader import DataLoader
from nearest_neighbor.base import NearestNeighborSearch
from dku_io_utils import save_array_to_folder

# Load parameters
params = load_indexing_recipe_params()

# Load data into vector format for indexing
columns = [params["unique_id_column"]] + params["feature_columns"]
input_df = params["input_dataset"].get_dataframe(columns=columns, infer_with_pandas=False)
data_loader = DataLoader(params["unique_id_column"], params["feature_columns"])
(vector_ids, vectors) = data_loader.convert_df_to_vectors(input_df)

# Build index and save index file to output folder
nearest_neighbor = NearestNeighborSearch(num_dimensions=vectors.shape[1], **params)
with NamedTemporaryFile() as tmp:
    nearest_neighbor.build_save_index(vectors=vectors, index_path=tmp.name)
    index_file_path = os.path.join(params["folder_partition_root"], nearest_neighbor.INDEX_FILE_NAME)
    params["index_folder"].upload_stream(index_file_path, tmp)

# Save vector data and indexing config to guarantee reproducibility
vector_ids_file_path = os.path.join(params["folder_partition_root"], nearest_neighbor.VECTOR_IDS_FILE_NAME)
vectors_file_path = os.path.join(params["folder_partition_root"], nearest_neighbor.VECTORS_FILE_NAME)
config_file_path = os.path.join(params["folder_partition_root"], nearest_neighbor.CONFIG_FILE_NAME)
save_array_to_folder(array=vector_ids, path=vector_ids_file_path, folder=params["index_folder"])
save_array_to_folder(array=vectors, path=vectors_file_path, folder=params["index_folder"])
config = {**nearest_neighbor.get_config(), **{k: v for k, v in params.items() if k in {"feature_columns", "expert"}}}
params["index_folder"].write_json(config_file_path, config)
