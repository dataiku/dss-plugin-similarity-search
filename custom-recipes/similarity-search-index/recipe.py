# -*- coding: utf-8 -*-
"""Nearest Neighbor Indexing recipe script"""

from dku_plugin_param_loading import load_plugin_config_indexing
from models.base import NearestNeighborSearchModel

input_dataset_name = get_input_names()[0]
input_dataset = dataiku.Dataset(input_dataset_name)
output_folder_name = get_output_names()[0]
output_folder = dataiku.Folder(output_folder_name)
output_folder_path = output_folder.get_path()

data_loader = DataLoader(input_dataset, get_recipe_config()["unique_id"], get_recipe_config()["embedding_column"])
unique_ids, vectors = data_loader.load()

params = get_recipe_config()
params["dims"] = vectors.shape[1]

model = NearestNeighborSearchModel(**params)
model.fit_and_save(unique_ids, vectors, output_folder_path)
