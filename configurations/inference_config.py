

from dataclasses import dataclass, field
from deep_models.models import TimeseriesDataset


@dataclass
class Config:

    csv_file: str = "./outputs/low_power_fdi_1_avg.csv"
    # csv_file: str = "./outputs/test_data.csv"

    epochs: int = 30
    batch_size: int = 64
    seq_len: int = 10
    learning_rate: float = 0.001
    models_folder: str = "./models"
    save_models: bool = True
    # "lstm_autoencoder_for_FDI_20.pth"
    model_name: str = "LSTM_autoencoder_for_FDI_10_v4.pth"
    normalization_model: str = "min_max_scaler_AE_for_FDI.save"
    # input dim is the length of columns being used
    input_dim: int = 10

    # Power_Cycle_4794407.csv
    # path_for_abnormal_data = "./datasets/Power_Cycle_5130104.csv"
    path_for_normalization_summary: str = "C:\\Users\\kvasi\\OneDrive - purdue.edu\\projects\\Autonomous Control System\\data\\global_feature_min_max_summary.csv"
    path_for_abnormal_data: str = "./datasets/low power/low_power_signal_1.csv"
    columns: list = field(default_factory=lambda: ["nfd-1-cps", "nfd-2-log", "nfd-3-pwr", "nfd-4-flux",
                                                   "rr-active-state", "rr-position", "ss1-active-state",
                                                   "ss1-position", "ss2-active-state", "ss2-position"])

    def save(self, path):
        '''  Save the configuration settings to a JSON file.   '''
        import json
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, path):
        '''  Load the configuration settings from a JSON file.   '''
        import json

        with open(path, 'r') as f:
            config_dict = json.load(f)

        self.__dict__.update(config_dict)
