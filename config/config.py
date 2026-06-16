
from dataclasses import dataclass, field


@dataclass
class Config:
    epochs: int = 30
    batch_size: int = 64
    seq_len: int = 10
    learning_rate: float = 0.001
    models_folder: str = "./models"
    save_models: bool = True
    model_name: str = "LSTM_autoencoder_for_FDI_10.pth"
    normalization_model: str = "min_max_scaler_AE_for_FDI.save"

    path_for_data = "C:\\Users\\kvasi\\OneDrive - purdue.edu\\projects\\Autonomous Control System\\data\\real_dataset1_final.csv"

    # columns: list = field(default_factory=lambda: ["Unnamed: 0", "nfd-1-cps", "nfd-1-cr", "rr-active-state",
    #                       "rr-position", "ss1-active-state", "ss1-position", "ss2-active-state", "ss2-position"])

    columns: list = field(default_factory=lambda: ["Unnamed: 0", "nfd-1-cps", "nfd-3-pwr", "nfd-4-flux",
                                                   "rr-active-state", "rr-position", "ss1-active-state", "ss1-position",
                                                   "ss2-active-state", "ss2-position"])  #

# Define the LSTM Autoencoder architecture
