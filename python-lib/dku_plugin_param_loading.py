# -*- coding: utf-8 -*-
"""Module with functions to load and validate plugin parameters using the Dataiku API"""

import logging
from typing import Dict

import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role


class PluginParamValidationError(ValueError):
    """Custom exception raised when the the plugin parameters chosen by the user are invalid"""

    pass


def load_indexing_recipe_params() -> Dict:
    """Load and validate parameters of the Nearest Neighbor Indexing recipe

    Returns:
        Dictionary of parameter names (key) and values

    Raises:
        PluginParamValidationError: If a parameter is not valid

    """
    params = {}
    # Input parameters
    input_dataset_names = get_input_names_for_role("input_dataset")
    if len(input_dataset_names) == 0:
        raise PluginParamValidationError("Please specify input folder")
    input_dataset = dataiku.Dataset(input_dataset_names[0])
    input_dataset_columns = [p["name"] for p in input_dataset.read_schema()]
    # Output parameters
    output_folder_names = get_output_names_for_role("output_folder")
    if len(output_folder_names) == 0:
        raise PluginParamValidationError("Please specify output folder")
    params["output_folder"] = dataiku.Folder(output_folder_names[0])
    params["output_folder_path"] = params["output_folder"].get_path()
    if not params["output_folder_path"]:
        raise PluginParamValidationError("Output folder must be on the local filesystem")
    # Recipe configuration parameters
    recipe_config = get_recipe_config()
    params["unique_id_column"] = recipe_config.get("unique_id_column")
    if params["unique_id_column"] not in input_dataset_columns:
        raise PluginParamValidationError(f"Invalid unique ID column: {params['unique_id_column']}")
    params["feature_columns"] = recipe_config.get("feature_columns", [])
    if not set(params["feature_columns"]).issubset(set(input_dataset_columns)):
        raise PluginParamValidationError(f"Invalid feature column(s): {params['feature_columns']}")
    params["input_df"] = input_dataset.get_dataframe(
        columns=[params["unique_id_column"]] + params["feature_columns"], infer_with_pandas=False
    )
    params["algorithm"] = recipe_config.get("algorithm")
    if params["algorithm"] not in {"annoy", "faiss"}:
        raise PluginParamValidationError(f"Invalid algorithm: {params['algorithm']}")
    params["expert"] = bool(recipe_config.get("expert"))
    if params["algorithm"] == "annoy":
        params["annoy_metric"] = recipe_config.get("annoy_metric")
        if params["annoy_metric"] not in {"angular", "euclidean", "manhattan", "hamming", "dot"}:
            raise PluginParamValidationError(f"Invalid Annoy distance metric: {params['annoy_metric']}")
        params["annoy_num_trees"] = recipe_config.get("annoy_num_trees")
        if not isinstance(params["annoy_num_trees"], int):
            raise PluginParamValidationError(f"Invalid number of trees: {params['annoy_num_trees']}")
        if params["annoy_num_trees"] < 1:
            raise PluginParamValidationError("Number of trees must be above 1")
    elif params["algorithm"] == "faiss":
        params["faiss_index_type"] = recipe_config.get("faiss_index_type")
        if params["faiss_index_type"] not in {"IndexFlatL2", "IndexFlatIP", "IndexLSH"}:
            raise PluginParamValidationError(f"Invalid FAISS index type: {params['faiss_index_type']}")
        params["faiss_lsh_num_bits"] = recipe_config.get("faiss_lsh_num_bits")
        if not isinstance(params["faiss_lsh_num_bits"], int):
            raise PluginParamValidationError(f"Invalid number of LSH bits: {params['faiss_lsh_num_bits']}")
        if params["faiss_lsh_num_bits"] < 4:
            raise PluginParamValidationError("Number of LSH bits must be above 4")
    printable_params = {k: v for k, v in params.items() if k != "input_df"}
    logging.info(f"Validated plugin recipe parameters: {printable_params}")
    return params


def load_lookup_recipe_params() -> Dict:
    """Load and validate parameters of the Nearest Neighbor Search recipe

    Returns:
        Dictionary of parameter names (key) and values

    Raises:
        PluginParamValidationError: If a parameter is not valid

    """
    pass
