import pandas as pd
import os.path

from data_loader import DataLoader
from tempfile import NamedTemporaryFile
from nearest_neighbor.base import NearestNeighborSearch


def test_build_save_index():

    params = {'unique_id_column': 'images',
              'feature_columns': ['prediction'],
              'algorithm': 'annoy',
              'expert': True,
              'annoy_metric': 'angular',
              'annoy_num_trees': 10}

    # Load data into array format for indexing
    columns = [params["unique_id_column"]] + params["feature_columns"]
    input_df = pd.read_csv('./tests/resources/caltech_embeddings.csv')
    # Restrict to selected columns
    input_df = input_df[columns]
    data_loader = DataLoader(params["unique_id_column"], params["feature_columns"])
    (array_ids, arrays) = data_loader.convert_df_to_arrays(input_df)
    nearest_neighbor = NearestNeighborSearch(num_dimensions=arrays.shape[1], **params)
    with NamedTemporaryFile() as tmp:
        nearest_neighbor.build_save_index(arrays=arrays, index_path=tmp.name)
        assert os.path.isfile(tmp.name)


def test_find_neighbors_df():

    params = {'unique_id_column': 'images',
              'feature_columns': ['prediction'],
              'algorithm': 'annoy',
              'expert': True,
              'annoy_metric': 'angular',
              'annoy_num_trees': 10}

    index_config = {'algorithm': 'annoy',
                    'num_dimensions': 2048,
                    'annoy_metric': 'angular',
                    'annoy_num_trees': 10,
                    'feature_columns': ['prediction'],
                    'expert': True}

    # Load data into array format for indexing
    columns = [params["unique_id_column"]] + params["feature_columns"]
    input_df = pd.read_csv('./tests/resources/caltech_embeddings.csv')
    input_df = input_df[columns]
    data_loader = DataLoader(params["unique_id_column"], params["feature_columns"])
    (array_ids, arrays) = data_loader.convert_df_to_arrays(input_df)
    nearest_neighbor = NearestNeighborSearch(num_dimensions=arrays.shape[1], **params)
    with NamedTemporaryFile() as tmp:
        nearest_neighbor.build_save_index(arrays=arrays, index_path=tmp.name)
        params = {'unique_id_column': 'images', 'feature_columns': ['prediction'], 'num_neighbors': 5}
        nearest_neighbor = NearestNeighborSearch(**index_config)
        nearest_neighbor.load_index(tmp.name)
        # Find nearest neighbors in input dataset
        df = nearest_neighbor.find_neighbors_df(input_df, **params, index_array_ids=array_ids)
        actual = sorted(list(df[df['input_id'] == '34719_ostrich.jpg']['neighbor_id']))
        expected = ['107505_ostrich.jpg', '185189_ostrich.jpg', '213657_ostrich.jpg', '229350_ostrich.jpg', '34719_ostrich.jpg']
    assert len(actual) == len(expected)
    assert all([actual_item == expected_item for actual_item, expected_item in zip(actual, expected)])
