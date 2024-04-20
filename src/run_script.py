import argparse
import importlib
import sys
from pathlib import Path


def run_script(args):
    """
    Dynamically import a script based on its file path or module path and run its `run` function with provided arguments.
    Handles both absolute/relative file paths and module-like paths.

    :param args: Namespace of arguments including the script path.
    """
    # Extract the script path from the provided arguments
    script_path = args.script

    # Check if the script_path is a module path (contains dots but no slashes)
    if "." in script_path and not any(sep in script_path for sep in ("/", "\\")):
        # It's a module path; we directly use it
        module_name, function_name = script_path.rsplit(".", 1)
    else:
        # It's a file path; we convert it to a module path
        file_path = Path(script_path).resolve()

        # Determine the root path (src directory) relative to this script
        root_path = Path(__file__).resolve().parent

        # Calculate the relative module path
        try:
            relative_module_path = file_path.relative_to(root_path).with_suffix("")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Convert file path to module path format
        module_path = str(relative_module_path).replace("/", ".").replace("\\", ".")
        module_name, function_name = f"{module_path}.run".rsplit(".", 1)

    try:
        module = importlib.import_module(module_name)
        function_to_run = getattr(module, function_name)
        function_to_run(args)
    except (ImportError, AttributeError) as e:
        print(f"Error importing or running the script: {e}")
        sys.exit(1)


def run_script_2(args):
    """
    Dynamically import a script based on its file path and run its `run` function with provided arguments.
    Handles both absolute and relative file paths.

    :param file_path: The file path to the script, which can be absolute or relative.
    :param args: Parsed arguments to pass to the script's `run` function.
    """
    # Resolve the absolute path from the provided file path
    file_path = args.script
    print(sys.argv)
    script_path = Path(file_path).resolve()

    # Determine the root path (src directory) relative to this script
    root_path = Path(__file__).resolve().parent

    # If the provided path is already absolute, calculate the relative module path directly
    # Otherwise, treat it as relative to the 'src' directory
    if script_path.is_absolute():
        relative_module_path = script_path.relative_to(root_path).with_suffix("")
    else:
        # For a relative path, form the absolute path assuming 'src' as the base directory
        relative_module_path = (
            (root_path / file_path).resolve().relative_to(root_path).with_suffix("")
        )

    # Replace OS-specific path separators with '.' to form a valid Python module path
    module_path = str(relative_module_path).replace("/", ".").replace("\\", ".")

    # Append '.run' to target the run function directly
    script_module_name = f"{module_path}.run"

    try:
        module_name, function_name = script_module_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        function_to_run = getattr(module, function_name)
        function_to_run(args)
    except (ImportError, AttributeError) as e:
        print(f"Error importing or running the script: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run scripts for DeepONet operations.")
    parser.add_argument(
        "--script",
        type=str,
        # default="multionet_scripts.multionet_pchemicals_tests.run",
        # default="deeponet_scripts.deeponet_spectra_fc.run",
        default="multionet_scripts.multionet_bchemicals_tests.run",
        help="Path to the script to run, e.g., 'deeponet_scripts.deeponet_training.run'",
    )

    # Arguments for the script example provided
    parser.add_argument(
        "--train",
        action="store_true",
        default=True,
        help="Whether to train the model.",
    )
    parser.add_argument(
        "--vis", action="store_true", default=True, help="Whether to visualize data."
    )
    parser.add_argument(
        "--use_mass_conservation",
        action="store_true",
        default=False,
        help="Use mass conservation in the loss function.",
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Path to a pretrained model file.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/dataset1000",
        help="Path to the data file.",
    )
    parser.add_argument(
        "--branch_input_size",
        type=int,
        default=29,
        help="Input size for the branch network.",
    )
    parser.add_argument(
        "--trunk_input_size",
        type=int,
        default=1,
        help="Input size for the trunk network.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=100,
        help="Number of hidden units in each layer.",
    )
    parser.add_argument(
        "--branch_hidden_layers",
        type=int,
        default=2,
        help="Number of hidden layers in the branch network.",
    )
    parser.add_argument(
        "--trunk_hidden_layers",
        type=int,
        default=2,
        help="Number of hidden layers in the trunk network.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=500, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate for training."
    )
    parser.add_argument(
        "--fraction", type=float, default=1, help="Fraction of data to use."
    )
    parser.add_argument(
        "--output_neurons",
        type=int,
        default=290,
        help="Number of neurons in the output layer.",
    )
    parser.add_argument(
        "--n_outputs", type=int, default=29, help="Number of outputs of the model."
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="both",
        choices=["both", "branch", "trunk"],
        help="Architecture of the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        # choices=["cpu", "mps", "cuda"],
        help="Device to use for training.",
    )

    parser.add_argument(
        "--device_ids",
        type=int,
        default=(3, 4, 5),
        help="Device IDs to use for training.",
    )

    args = parser.parse_args()
    run_script(args)


if __name__ == "__main__":
    main()
