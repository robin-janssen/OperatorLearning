# This script utilizes multiprocessing to train multiple models on individual GPUs.

import multiprocessing
import numpy as np

# from torchinfo import summary

from data import create_dataloader_chemicals, load_chemical_data

# from plotting import (
#     plot_chemical_examples,
#     plot_chemicals_comparative,
#     plot_losses,
#     plot_chemical_results,
#     plot_chemical_errors,
#     plot_relative_errors_over_time,
# )
from training import (
    train_multionet_chemical_remote,
    save_model,
)

# from utils import read_yaml_config


def train_model_on_gpu(args, gpu_id):
    """
    Wrapper function to train a model on a specific GPU.
    :param args: Namespace of arguments needed for training.
    :param gpu_id: The GPU ID to use for training.
    """
    device = f"cuda:{gpu_id}"
    print(f"Training on {device}")

    data = load_chemical_data(args.data_path)
    train_data, test_data = data[:500], data[500:550]
    timesteps = np.arange(data.shape[1])

    dataloader_train = create_dataloader_chemicals(
        train_data, timesteps, fraction=1, batch_size=32, shuffle=True
    )

    dataloader_test = create_dataloader_chemicals(
        test_data, timesteps, fraction=1, batch_size=32, shuffle=False
    )

    masses = None

    multionet, train_loss, test_loss = train_multionet_chemical_remote(
        dataloader_train=dataloader_train,
        masses=masses,
        branch_input_size=args.branch_input_size,
        trunk_input_size=args.trunk_input_size,
        hidden_size=args.hidden_size,
        branch_hidden_layers=args.branch_hidden_layers,
        trunk_hidden_layers=args.trunk_hidden_layers,
        output_neurons=args.output_neurons,
        N_outputs=args.N_outputs,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        test_loader=dataloader_test,
        N_sensors=args.branch_input_size,
        N_timesteps=timesteps.shape[0],
        schedule=args.schedule,
        architecture=args.architecture,
        device=device,
        device_id=gpu_id,
        regularization_factor=args.regularization_factor,
        massloss_factor=args.massloss_factor,
    )

    save_model(
        multionet,
        f"multionet_chemical_{gpu_id}",
        {
            "branch_input_size": args.branch_input_size,
            "trunk_input_size": args.trunk_input_size,
            "hidden_size": args.hidden_size,
            "branch_hidden_layers": args.branch_hidden_layers,
            "trunk_hidden_layers": args.trunk_hidden_layers,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "N_timesteps": args.N_timesteps,
            "fraction": args.fraction,
            "num_samples_train": train_data.shape[0],
            "num_samples_test": test_data.shape[0],
            "output_neurons": args.output_neurons,
            "N_outputs": args.N_outputs,
            "schedule": args.schedule,
            "regularization_factor": args.regularization_factor,
            "massloss_factor": args.massloss_factor,
            "device": device,
            "device_id": gpu_id,
            "train_duration": train_multionet_chemical_remote.train_duration,
            "architecture": args.architecture,
        },
        train_loss=train_loss,
        test_loss=test_loss,
    )


def run(args):
    """
    Launches multiple training processes on specified GPUs.
    :param args: Namespace of arguments needed for training.
    :param gpu_ids: A tuple of GPU IDs on which to train models.
    """
    processes = []
    for gpu_id in args.device_ids:
        p = multiprocessing.Process(
            target=train_model_on_gpu,
            args=(
                args,
                gpu_id,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
