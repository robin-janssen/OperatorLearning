from __future__ import annotations

import os
from typing import Type
import torch
import yaml
from datetime import datetime
import streamlit as st
import time
import functools
import numpy as np

from deeponet import DeepONet, MultiONet, MultiONetB, MultiONetT


# TODO complete type hints
def save_model(
    model: Type[DeepONet | MultiONet],
    model_name,
    hyperparameters,
    subfolder="models",
    train_loss: np.ndarray | None = None,
    test_loss: np.ndarray | None = None,
):
    """
    Save the trained model and hyperparameters.

    :param model: The trained model.
    :param hyperparameters: Dictionary containing hyperparameters.
    :param base_dir: Base directory for saving the model.
    """
    # Create a directory based on the current date
    model_dir = create_date_based_directory(subfolder=subfolder)

    # Save the model state dict
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)

    # Save hyperparameters as a YAML file
    hyperparameters_path = os.path.join(model_dir, f"{model_name}.yaml")
    with open(hyperparameters_path, "w") as file:
        yaml.dump(hyperparameters, file)

    if train_loss is not None and test_loss is not None:
        # Save the losses as a numpy file
        losses_path = os.path.join(model_dir, f"{model_name}_losses.npz")
        np.savez(losses_path, train_loss=train_loss, test_loss=test_loss)

    print(f"Model, losses and hyperparameters saved to {model_dir}")


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


def time_execution(func):
    """
    Decorator to time the execution of a function and store the duration
    as an attribute of the function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.duration = end_time - start_time
        print(f"{func.__name__} executed in {wrapper.duration:.2f} seconds.")
        return result

    wrapper.duration = None
    return wrapper


def load_chemical_data(data_dir, file_extension=".dat", separator=" "):
    """
    Load chemical data from a directory containing multiple files.

    :param data_dir: The directory containing the data files.
    :param file_extension: The file extension of the data files.
    :return: A list of numpy arrays containing the data from each file.
    """
    # Get a list of all relevant files in the directory
    all_files = os.listdir(data_dir)
    files = [file for file in all_files if file.endswith(file_extension)]
    num_files = len(files)
    files.sort()

    # Load one file to see the data shape
    data = np.loadtxt(os.path.join(data_dir, files[0]), delimiter=separator)
    data_shape = data.shape

    # Create an array to store all the data
    all_data = np.zeros((num_files, *data_shape))

    # Iterate over all the files and load the data
    for i, file in enumerate(files):
        if file.endswith(file_extension):
            data = np.loadtxt(os.path.join(data_dir, file), delimiter=separator)
            all_data[i] = data

    return all_data


def make_multionet(
    branch_input_size,
    branch_hidden_size,
    branch_hidden_layers,
    trunk_input_size,
    trunk_hidden_size,
    trunk_hidden_layers,
    output_size,
    N_outputs,
    architecture,
):
    """
    Load a MultiONet model based on the specified architecture.

    :param architecture: The architecture of the MultiONet model.
    :param branch_input_size: The number of input neurons for the branch network.
    :param hidden_size: The number of neurons in the hidden layers.
    :param branch_hidden_layers: The number of hidden layers in the branch network.
    :param trunk_input_size: The number of input neurons for the trunk network.
    :param hidden_size: The number of neurons in the hidden layers.
    :param trunk_hidden_layers: The number of hidden layers in the trunk network.
    :param output_size: The number of output neurons.
    :param N_outputs: The number of outputs.
    :param device: The device to use for the model.
    :return: The MultiONet model.
    """
    if architecture == "both":
        deeponet = MultiONet(
            branch_input_size,
            branch_hidden_size,
            branch_hidden_layers,
            trunk_input_size,
            trunk_hidden_size,
            trunk_hidden_layers,
            output_size,
            N_outputs,
        )
    elif architecture == "branch":
        deeponet = MultiONetB(
            branch_input_size,
            branch_hidden_size,
            branch_hidden_layers,
            trunk_input_size,
            trunk_hidden_size,
            trunk_hidden_layers,
            output_size,
            N_outputs,
        )
    elif architecture == "trunk":
        deeponet = MultiONetT(
            branch_input_size,
            branch_hidden_size,
            branch_hidden_layers,
            trunk_input_size,
            trunk_hidden_size,
            trunk_hidden_layers,
            output_size,
            N_outputs,
        )
    return deeponet
