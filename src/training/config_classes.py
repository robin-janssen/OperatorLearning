from dataclasses import dataclass
from typing import Optional
from optuna import Trial


@dataclass
class PChemicalTrainConfig:
    masses: Optional[list[float]] = None
    branch_input_size: int = 216
    trunk_input_size: int = 1
    hidden_size: int = 2000
    branch_hidden_layers: int = 5
    trunk_hidden_layers: int = 5
    output_neurons: int = 2160
    N_outputs: int = 216
    num_epochs: int = 20
    learning_rate: float = 1e-5
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
    batch_size: int = 64
