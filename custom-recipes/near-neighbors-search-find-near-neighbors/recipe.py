import dataiku
from dataiku.customrecipe import get_input_names, get_output_names, get_recipe_config
from models.base import BaseNNS
from nns_utils import reshape_values, reshape_nns
import os
import pandas as pd

input_dataset_name = get_input_names()[0]
input_dataset = dataiku.Dataset(input_dataset_name)

index_folder_name = get_input_names()[1]
index_folder = dataiku.Folder(index_folder_name)

output_dataset_name = get_output_names()[0]
output_dataset = dataiku.Dataset(output_dataset_name)

embedding_column = get_recipe_config()['embedding_column']
unique_id = get_recipe_config()['unique_id']
number_of_neighbors = int(get_recipe_config()['number_of_neighbors'])

config_file_path = os.path.join(index_folder.get_path(), 'config.json')


if os.path.isfile(config_file_path):
    with open(config_file_path) as json_file:
        config = json.loads(json.load(json_file))
        index_file_path = os.path.join(index_folder.get_path(), config.get('index_file'))
else:
    raise ValueError("The configuration file cannot be found.")


algo = BaseNNS(**config)
algo._load(index_file_path)

with output_dataset.get_writer() as writer:

    set_schema = True
    for df in input_dataset.iter_dataframes(columns=[unique_id, embedding_column],
                                            chunksize=1000):
        df = df.dropna()
        vectors = reshape_values(df[embedding_column])
        nns = algo._find_near_neighbors(vectors, number_of_neighbors)
        nns_df = reshape_nns(nns=nns, query_names=df[unique_id], index_names=pd.Series(config.get('names_dict')))
        if set_schema:
            output_dataset.write_schema_from_dataframe(nns_df)
            set_schema = False

        writer.write_dataframe(nns_df)
