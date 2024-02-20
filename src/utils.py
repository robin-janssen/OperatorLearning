import os
import torch
import yaml
from datetime import datetime
import streamlit as st
import time
import functools


def save_model(model, model_name, hyperparameters, subfolder="models"):
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
    hyperparameters_path = os.path.join(model_dir, f"{model_name}_config.yaml")
    with open(hyperparameters_path, "w") as file:
        yaml.dump(hyperparameters, file)

    print(f"Model and hyperparameters saved to {model_dir}")


def create_date_based_directory(base_dir=".", subfolder="models"):
    """
    Create a directory based on the current date (dd-mm format) inside a specified subfolder of the base directory.

    :param base_dir: The base directory where the subfolder and date-based directory will be created.
    :param subfolder: The subfolder inside the base directory to include before the date-based directory.
    :return: The path of the created date-based directory within the specified subfolder.
    """
    # Get the current date in dd-mm format
    current_date = datetime.now().strftime("%d-%m")
    full_path = os.path.join(base_dir, subfolder, current_date)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


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
