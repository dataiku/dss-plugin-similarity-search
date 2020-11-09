import dataiku
from dataiku.customrecipe import get_input_names, get_output_names, get_recipe_config
from data_loader import DataLoader
from models.base import BaseNNS

input_dataset_name = get_input_names()[0]
input_dataset = dataiku.Dataset(input_dataset_name)
output_folder_name = get_output_names()[0]
output_folder = dataiku.Folder(output_folder_name)

images = DataLoader(input_dataset, get_recipe_config()["unique_id"], get_recipe_config()["embedding_column"])
names, vectors = images._load()

params = get_recipe_config()
params["dims"] = vectors.shape[1]

algo = BaseNNS(**params)
algo.fit_and_save(names, vectors, output_folder)
