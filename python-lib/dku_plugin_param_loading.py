# -*- coding: utf-8 -*-
"""Module with functions to load and validate plugin parameters using the Dataiku API"""

import logging
from typing import Dict
from enum import Enum

import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role


class RecipeID(Enum):
    """Enum class to identify each recipe"""

    SIMILARITY_SEARCH_INDEX = "Nearest Neighbor Indexing"
    SIMILARITY_SEARCH_QUERY = "Nearest Neighbor Search"


class PluginParamValidationError(ValueError):
    """Custom exception raised when the the plugin parameters chosen by the user are invalid"""

    pass


def load_input_output_params(recipe_id: RecipeID) -> Dict:
    """Load and validate input/output parameters for both indexing and search recipes

    Returns:
        Dictionary of parameter names (key) and values

    Raises:
        PluginParamValidationError: If a parameter is not valid

    """
    # Input dataset
    params = {}
    input_dataset_names = get_input_names_for_role("input_dataset")
    if len(input_dataset_names) == 0:
        raise PluginParamValidationError("Please specify input folder")
    params["input_dataset"] = dataiku.Dataset(input_dataset_names[0])
    input_dataset_columns = [p["name"] for p in params["input_dataset"].read_schema()]
    # Index folder
    if recipe_id == RecipeID.SIMILARITY_SEARCH_INDEX:
        output_folder_names = get_output_names_for_role("index_folder")
        if len(output_folder_names) == 0:
            raise PluginParamValidationError("Please specify index folder as output")
        params["index_folder"] = dataiku.Folder(output_folder_names[0])
    elif recipe_id == RecipeID.SIMILARITY_SEARCH_QUERY:
        input_folder_names = get_input_names_for_role("index_folder")
        if len(input_folder_names) == 0:
            raise PluginParamValidationError("Please specify index folder as input")
        params["index_folder"] = dataiku.Folder(input_folder_names[0])
    params["index_folder_path"] = params["index_folder"].get_path()
    if not params["index_folder_path"]:
        raise PluginParamValidationError("Index folder must be on the local filesystem")
    # Output dataset - only for search recipe
    if recipe_id == RecipeID.SIMILARITY_SEARCH_QUERY:
        output_dataset_names = get_output_names_for_role("output_dataset")
        if len(output_dataset_names) == 0:
            raise PluginParamValidationError("Please specify output dataset")
        params["output_dataset"] = dataiku.Dataset(output_dataset_names[0])
    # Recipe input parameters
    recipe_config = get_recipe_config()
    params["unique_id_column"] = recipe_config.get("unique_id_column")
    if params["unique_id_column"] not in input_dataset_columns:
        raise PluginParamValidationError(f"Invalid unique ID column: {params['unique_id_column']}")
    params["feature_columns"] = recipe_config.get("feature_columns", [])
    if not set(params["feature_columns"]).issubset(set(input_dataset_columns)):
        raise PluginParamValidationError(f"Invalid feature column(s): {params['feature_columns']}")
    printable_params = {k: v for k, v in params.items() if k not in {"input_dataset", "index_folder", "output_dataset"}}
    logging.info(f"Validated input/output parameters: {printable_params}")
    return params


def load_indexing_recipe_params() -> Dict:
    """Load and validate parameters of the Nearest Neighbor Indexing recipe

    Returns:
        Dictionary of parameter names (key) and values

    Raises:
        PluginParamValidationError: If a parameter is not valid

    """
    logging.info("Validating Nearest Neighbor Indexing parameters...")
    input_output_params = load_input_output_params(RecipeID.SIMILARITY_SEARCH_INDEX)
    # Recipe modeling parameters
    modeling_params = {}
    recipe_config = get_recipe_config()
    modeling_params["algorithm"] = recipe_config.get("algorithm")
    if modeling_params["algorithm"] not in {"annoy", "faiss"}:
        raise PluginParamValidationError(f"Invalid algorithm: {modeling_params['algorithm']}")
    modeling_params["expert"] = bool(recipe_config.get("expert"))
    if modeling_params["algorithm"] == "annoy":
        modeling_params["annoy_metric"] = recipe_config.get("annoy_metric")
        if modeling_params["annoy_metric"] not in {"angular", "euclidean", "manhattan", "hamming", "dot"}:
            raise PluginParamValidationError(f"Invalid Annoy distance metric: {modeling_params['annoy_metric']}")
        modeling_params["annoy_num_trees"] = recipe_config.get("annoy_num_trees")
        if not isinstance(modeling_params["annoy_num_trees"], int):
            raise PluginParamValidationError(f"Invalid number of trees: {modeling_params['annoy_num_trees']}")
        if modeling_params["annoy_num_trees"] < 1:
            raise PluginParamValidationError("Number of trees must be above 1")
    elif modeling_params["algorithm"] == "faiss":
        modeling_params["faiss_index_type"] = recipe_config.get("faiss_index_type")
        if modeling_params["faiss_index_type"] not in {"IndexFlatL2", "IndexFlatIP", "IndexLSH"}:
            raise PluginParamValidationError(f"Invalid FAISS index type: {modeling_params['faiss_index_type']}")
        modeling_params["faiss_lsh_num_bits"] = recipe_config.get("faiss_lsh_num_bits")
        if not isinstance(modeling_params["faiss_lsh_num_bits"], int):
            raise PluginParamValidationError(f"Invalid number of LSH bits: {modeling_params['faiss_lsh_num_bits']}")
        if modeling_params["faiss_lsh_num_bits"] < 4:
            raise PluginParamValidationError("Number of LSH bits must be above 4")
    logging.info(f"Validated modeling parameters: {modeling_params}")
    return {**input_output_params, **modeling_params}


def load_search_recipe_params() -> Dict:
    """Load and validate parameters of the Nearest Neighbor Search recipe

    Returns:
        Dictionary of parameter names (key) and values

    Raises:
        PluginParamValidationError: If a parameter is not valid

    """
    logging.info("Validating Nearest Neighbor Search parameters...")
    input_output_params = load_input_output_params(RecipeID.SIMILARITY_SEARCH_QUERY)
    # Recipe lookup parameters
    lookup_params = {}
    recipe_config = get_recipe_config()
    lookup_params["num_neighbors"] = recipe_config.get("num_neighbors")
    if not isinstance(lookup_params["num_neighbors"], int):
        raise PluginParamValidationError(f"Invalid number of neighbors: {lookup_params['num_neighbors']}")
    if lookup_params["num_neighbors"] < 1 or lookup_params["num_neighbors"] > 1000:
        raise PluginParamValidationError("Number of neighbors must be between 1 and 1000")
    logging.info(f"Validated lookup parameters: {lookup_params}")
    return {**input_output_params, **lookup_params}
