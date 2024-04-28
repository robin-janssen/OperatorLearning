# This script implements an OPTUNA study to investigate the performance of a MultiONet model in interpolating data.
# Both the branca and osu datasets will be investigated.

import numpy as np

from data import (
    create_dataloader_chemicals,
    # analyze_branca_data,
    # prepare_branca_data,
)

from training import (
    BChemicalTrainConfig,
    # train_multionet_chemical,
    # save_model,
    test_deeponet,
    load_multionet,
)

from plotting import (
    # plot_chemical_examples,
    # plot_chemicals_comparative,
    plot_relative_errors_over_time,
    plot_chemical_results,
    plot_losses,
)


def run(args):

    config = BChemicalTrainConfig()
    args.vis = False
    config.device = args.device
    test_data = np.load("dummy_data.npy")
    timesteps = np.linspace(0, 15, 16)
    model_path = "dummy_model"

    dataloader = create_dataloader_chemicals(
        test_data,
        timesteps,
        fraction=1,
        batch_size=config.batch_size,
        shuffle=False,
    )
    multionet, train_loss, test_loss = load_multionet(config, config.device, model_path)

    average_error, predictions, ground_truth = test_deeponet(
        multionet,
        dataloader,
        N_timesteps=config.N_timesteps,
        reshape=True,
    )

    print(f"Average prediction error: {average_error:.3E}")

    errors = np.abs(predictions - ground_truth)
    relative_errors = errors / np.abs(ground_truth)

    if args.vis:
        plot_losses(train_loss, test_loss, "Branca", "losses_branca.png")

        plot_relative_errors_over_time(
            relative_errors,
            "Relative errors over time (MultiONet1 for Chemicals)",
            save=True,
        )

        plot_chemical_results(
            predictions=predictions,
            ground_truth=ground_truth,
            # names=extracted_chemicals,
            num_chemicals=10,
            model_names="MultiONet1",
            save=True,
        )
