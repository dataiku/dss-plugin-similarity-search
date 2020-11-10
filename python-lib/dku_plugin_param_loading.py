# -*- coding: utf-8 -*-
"""Module with functions to load and validate plugin parameters using the Dataiku API"""

import logging
import os
from typing import Dict

import dataiku
from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role,
    get_recipe_resource,
)

from data_loader import DataLoader


class PluginParamValidationError(ValueError):
    """Custom exception raised when the the plugin parameters chosen by the user are invalid"""

    pass


def load_plugin_config_indexing() -> Dict:
    """Load and validate parameters of the Nearest Neighbor Indexing recipe

    Returns:
        Dictionary of parameter names (key) and values

    Raises:
        PluginParamValidationError: If a parameter is not valid

    """
    input_dataset_names = get_input_names_for_role("input_dataset")
    if len(input_dataset_names) == 0:
        raise PluginParamValidationError("Please specify input folder")
    input_dataset = dataiku.Dataset(input_dataset_names[0])
    output_folder_name = get_output_names()[0]
    output_folder = dataiku.Folder(output_folder_name)
    params = get_recipe_config()


def load_plugin_config_search() -> Dict:
    """Load and validate parameters of the Nearest Neighbor Search recipe

    Returns:
        Dictionary of parameter names (key) and values

    Raises:
        PluginParamValidationError: If a parameter is not valid

    """
    pass
