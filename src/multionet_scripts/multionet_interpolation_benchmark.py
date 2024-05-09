# This script implements an OPTUNA study to investigate the performance of a MultiONet model
# in interpolating and extrapolating the osu dataset.

import numpy as np
import matplotlib.pyplot as plt

from data import (
    create_dataloader_chemicals,
)

from training import (
    OChemicalTrainConfig,
    test_deeponet,
    load_multionet,
)

from plotting import (
    plot_losses,
    plot_relative_errors_over_time,
    plot_chemical_results,
    plot_generalization_errors,
)

from utils import list_pth_files


def run(args):

    # Use this to toggle between interpolation and extrapolation
    interpolate = True

    config = OChemicalTrainConfig()
    args.vis = True
    config.device = args.device
    test_data = np.load("data/osu_data/test_data.npy")
    timesteps = np.linspace(0, 99, 100)
    model_dir = "models/interpolation" if interpolate else "models/extrapolation"

    dataloader = create_dataloader_chemicals(
        test_data,
        timesteps,
        fraction=1,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Find model paths based on provided folder
    model_names = list_pth_files(model_dir)
    model_names = sorted(model_names, key=lambda x: int(x.split("_")[-1]))

    model_errors = []
    metrics = []

    for i, model_name in enumerate(model_names):
        model_path = f"{model_dir}/{model_name}"
        print(f"Evaluating model {i + 1}/{len(model_names)}: {model_path}")
        metric = int(model_name.split("_")[-1])
        metrics.append(metric)

        multionet, train_loss, test_loss = load_multionet(
            config, config.device, model_path
        )

        average_error, predictions, ground_truth = test_deeponet(
            multionet,
            dataloader,
            N_timesteps=config.N_timesteps,
            reshape=True,
        )

        print(f"Average prediction error: {average_error:.3E}")
        model_errors.append(average_error)

        errors = np.abs(predictions - ground_truth)
        relative_errors = errors / np.abs(ground_truth)

        if args.vis:
            phrase = "Interpolation interval" if interpolate else "Extrapolation cutoff"
            plot_losses(
                (train_loss, test_loss),
                ("Train loss", "Test loss"),
                f"Losses (MultiONet on Osu Data, {phrase}: {metric})",
                save=True,
            )

            plot_relative_errors_over_time(
                relative_errors,
                f"Relative errors over time (MultiONet on Osu Data, {phrase}: {metric})",
                save=True,
            )

            plot_chemical_results(
                predictions=predictions,
                ground_truth=ground_truth,
                # names=extracted_chemicals,
                num_chemicals=10,
                title=f"Predictions of MultiONet on Osu Data ({phrase}: {metric})",
                save=True,
            )

            plt.close("all")

    if args.vis:
        metrics = np.array(metrics)
        model_errors = np.array(model_errors)
        plot_generalization_errors(
            metrics, model_errors, interpolate=interpolate, save=True
        )
