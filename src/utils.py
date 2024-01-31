import os
import torch
import yaml
from datetime import datetime

def save_model(model, model_name, hyperparameters, base_dir="models"):
    """
    Save the trained model and hyperparameters.

    :param model: The trained model.
    :param hyperparameters: Dictionary containing hyperparameters.
    :param base_dir: Base directory for saving the model.
    """
    # Create a directory path with the current date
    date_str = datetime.now().strftime("%d-%m")
    model_dir = os.path.join(base_dir, date_str)

    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Save the model state dict
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)

    # Save hyperparameters as a YAML file
    hyperparameters_path = os.path.join(model_dir, "hyperparameters.yaml")
    with open(hyperparameters_path, 'w') as file:
        yaml.dump(hyperparameters, file)

    print(f"Model and hyperparameters saved to {model_dir}")

