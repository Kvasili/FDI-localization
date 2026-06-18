'''

    This code contains  the configuration settings for the FDI localization model and a DataPreparer 
    class that provides methods for loading, splitting, and preprocessing the data, 
    including min-max normalization based on pre-computed min and max values for each feature.

'''


from dataclasses import dataclass, field
from deep_models.models import TimeseriesDataset


@dataclass
class Config:
    epochs: int = 15
    batch_size: int = 64
    seq_len: int = 10
    learning_rate: float = 0.001
    models_folder: str = "./models"
    save_models: bool = True
    model_name: str = "test_model.pth"
    # model_name: str = "LSTM_autoencoder_for_FDI_10_v5.pth"
    path_for_normalization_summary: str = "C:\\Users\\kvasi\\OneDrive - purdue.edu\\projects\\Autonomous Control System\\data\\global_feature_min_max_summary.csv"
    path_for_data: str = "C:\\Users\\kvasi\\OneDrive - purdue.edu\\projects\\Autonomous Control System\\data\\real_dataset1_final.csv"
    path_for_training: str = "C:\\Users\\kvasi\\OneDrive - purdue.edu\\projects\\Autonomous Control System\\data\\Full cycles and Startups\\Power_Cycle_with_Startup\\training"
    path_for_validation: str = "C:\\Users\\kvasi\\OneDrive - purdue.edu\\projects\\Autonomous Control System\\data\\Full cycles and Startups\\Power_Cycle_with_Startup\\validation"

    # columns: list = field(default_factory=lambda: ["Unnamed: 0", "nfd-1-cps", "nfd-1-cr", "rr-active-state",
    #                       "rr-position", "ss1-active-state", "ss1-position", "ss2-active-state", "ss2-position"])

    columns: list = field(default_factory=lambda: ["nfd-1-cps", "nfd-2-log", "nfd-3-pwr", "nfd-4-flux",
                                                   "rr-active-state", "rr-position", "ss1-active-state", "ss1-position", "ss2-active-state", "ss2-position"])

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
