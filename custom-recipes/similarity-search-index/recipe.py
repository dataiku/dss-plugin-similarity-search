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

# Load data into array format for indexing
columns = [params["unique_id_column"]] + params["feature_columns"]
input_df = params["input_dataset"].get_dataframe(columns=columns, infer_with_pandas=False)
data_loader = DataLoader(params["unique_id_column"], params["feature_columns"])
(array_ids, arrays) = data_loader.convert_df_to_array(input_df)

# Build index and save index file to output folder
nearest_neighbor = NearestNeighborSearch(num_dimensions=arrays.shape[1], **params)
with NamedTemporaryFile() as tmp:
    nearest_neighbor.build_save_index(arrays=arrays, index_path=tmp.name)
    index_file_path = os.path.join(params["folder_partition_root"], nearest_neighbor.INDEX_FILE_NAME)
    params["index_folder"].upload_stream(index_file_path, tmp)

# Save arrays and indexing config to guarantee reproducibility
array_ids_file_path = os.path.join(params["folder_partition_root"], nearest_neighbor.ARRAY_IDS_FILE_NAME)
arrays_file_path = os.path.join(params["folder_partition_root"], nearest_neighbor.ARRAYS_FILE_NAME)
config_file_path = os.path.join(params["folder_partition_root"], nearest_neighbor.CONFIG_FILE_NAME)
save_array_to_folder(array=array_ids, path=array_ids_file_path, folder=params["index_folder"])
save_array_to_folder(array=arrays, path=arrays_file_path, folder=params["index_folder"])
config = {**nearest_neighbor.get_config(), **{k: v for k, v in params.items() if k in {"feature_columns", "expert"}}}
params["index_folder"].write_json(config_file_path, config)
