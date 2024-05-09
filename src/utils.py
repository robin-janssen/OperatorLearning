from __future__ import annotations

import os
import time
from datetime import datetime
import yaml

import streamlit as st
import torch


# TODO complete type hints


def read_yaml_config(model_path):
    config_path = model_path.replace(".pth", ".yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_date_based_directory(base_dir=".", subfolder="models"):
    """
    Create a directory based on the current date (dd-mm format) inside a specified subfolder of the base directory.

    :param base_dir: The base directory where the subfolder and date-based directory will be created.
    :param subfolder: The subfolder inside the base directory to include before the date-based directory.
    :return: The path of the created date-based directory within the specified subfolder.
    """
    # Get the current date in dd-mm format
    current_date = datetime.now().strftime("%m-%d")
    full_path = os.path.join(base_dir, subfolder, current_date)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def save_plot_counter(filename, directory="plots"):
    # Initialize filename and counter
    filepath = os.path.join(directory, filename)
    filebase, fileext = filename.split(".")
    counter = 1

    # Check if the file exists and modify the filename accordingly
    while os.path.exists(filepath):
        filename = f"{filebase}_{counter}.{fileext}"
        filepath = os.path.join(directory, filename)
        counter += 1

    return filepath


def get_or_create_placeholder(key):
    """Get an existing placeholder or create a new one if it doesn't exist."""
    if key not in st.session_state:
        with st.container():
            placeholder = st.empty()
            st.session_state[key] = placeholder
    return st.session_state[key]


def get_project_path(relative_path):
    """
    Construct the absolute path to a project resource (data or model) based on a relative path.

    :param relative_path: A relative path to the resource, e.g., "data/dataset100" or "models/02-28/model.pth".
    :return: The absolute path to the resource.
    """
    import os

    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.realpath(__file__))

    # Navigate up to the parent directory of 'src' and then to the specified relative path
    project_resource_path = os.path.join(current_script_dir, "..", relative_path)

    # Normalize the path to resolve any '..' components
    project_resource_path = os.path.normpath(project_resource_path)

    return project_resource_path


def list_pth_files(directory: str = "models") -> list[str]:
    """
    List all .pth files in the specified directory without their extensions.

    Args:
        directory (str): The directory to search for .pth files.

    Returns:
        list[str]: A list of filenames without the .pth extension.
    """
    files = os.listdir(directory)
    pth_files = [file[:-4] for file in files if file.endswith(".pth")]
    return pth_files


def set_random_seed(gpu_id: int = 0):
    # Combine the current time with the GPU ID to create a unique seed
    seed = int(time.time()) + gpu_id
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you're using CUDA


def check_streamlit(use_streamlit: bool | None = None) -> bool:
    """
    Function to check whether python code is run within streamlit
    If use_streamlit is None, this will be interpreted as a request to check whether the code is run within streamlit.
    If use_streamlit is not None, this will be interpreted as a request to set the use_streamlit variable to the provided value.

    Args:
        use_streamlit (bool | None): Whether to use streamlit or not. If None, the function will return whether the code is run within streamlit.

    Returns:
        bool: Whether to use streamlit or not.
    """
    if use_streamlit is None:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx

            return get_script_run_ctx() is not None
        except ModuleNotFoundError:
            return False
    else:
        return use_streamlit
