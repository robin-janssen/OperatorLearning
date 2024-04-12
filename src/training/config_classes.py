from dataclasses import dataclass
from typing import Optional
from optuna import Trial


@dataclass
class PChemicalTrainConfig:
    masses: Optional[list[float]] = None
    branch_input_size: int = 216
    trunk_input_size: int = 1
    hidden_size: int = 767
    branch_hidden_layers: int = 4
    trunk_hidden_layers: int = 6
    output_neurons: int = 4320
    N_outputs: int = 216
    num_epochs: int = 20
    learning_rate: float = 9.396e-06
    schedule: bool = False
    N_sensors: int = 216
    N_timesteps: int = 128
    architecture: str = "both"
    pretrained_model_path: Optional[str] = None
    device: str = "cpu"
    use_streamlit: bool = False
    optuna_trial: Trial | None = None
    regularization_factor: float = 0.0
    massloss_factor: float = 0.0
    batch_size: int = 256


@dataclass
class SpectraTrainConfig:
    branch_input_size: int = 92
    trunk_input_size: int = 2
    hidden_size: int = 100
    branch_hidden_layers: int = 3
    trunk_hidden_layers: int = 3
    output_neurons: int = 100
    N_outputs: int = 1
    num_epochs: int = 2
    learning_rate: float = 3e-4
    schedule: bool = False
    N_sensors: int = 92
    N_timesteps: int = 11
    pretrained_model_path: Optional[str] = None
    device: str = "mps"
    use_streamlit: bool = False
    optuna_trial: Trial | None = None
    regularization_factor: float = 0.0
    batch_size: int = 256
