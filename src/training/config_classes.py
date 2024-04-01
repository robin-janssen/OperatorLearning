from dataclasses import dataclass
from typing import Optional
from torch.utils.data import DataLoader
from optuna import Trial


@dataclass
class PChemicalTrainConfig:
    data_loader: DataLoader
    masses: Optional[list[float]] = None
    branch_input_size: int = 216
    trunk_input_size: int = 1
    hidden_size: int = 100
    branch_hidden_layers: int = 3
    trunk_hidden_layers: int = 3
    output_neurons: int = 432
    N_outputs: int = 216
    num_epochs: int = 10
    learning_rate: float = 1e-3
    schedule: bool = False
    test_loader: Optional[DataLoader] = None
    N_sensors: int = 216
    N_timesteps: int = 128
    architecture: str = "both"
    pretrained_model_path: Optional[str] = None
    device: str = "cpu"
    use_streamlit: bool = True
    optuna_trial: Trial | None = None
    regularization_factor: float = 0.0
    massloss_factor: float = 0.0
