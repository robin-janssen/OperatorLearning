from __future__ import annotations

import os
import yaml
from datetime import datetime
import streamlit as st
import numpy as np


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


def load_chemical_data(data_folder, file_extension=".dat", separator=" "):
    """
    Load chemical data from a directory containing multiple files.

    :param data_folder: The directory containing the data files.
    :param file_extension: The file extension of the data files.
    :return: A list of numpy arrays containing the data from each file.
    """
    # Get a list of all relevant files in the directory
    dataset_path = get_project_path(data_folder)
    all_files = os.listdir(dataset_path)
    files = [file for file in all_files if file.endswith(file_extension)]
    num_files = len(files)
    files.sort()

    # Load one file to see the data shape
    data = np.loadtxt(os.path.join(dataset_path, files[0]), delimiter=separator)
    data_shape = data.shape

    # Create an array to store all the data
    all_data = np.zeros((num_files, *data_shape))

    # Iterate over all the files and load the data
    for i, file in enumerate(files):
        if file.endswith(file_extension):
            data = np.loadtxt(os.path.join(dataset_path, file), delimiter=separator)
            all_data[i] = data

    return all_data


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
