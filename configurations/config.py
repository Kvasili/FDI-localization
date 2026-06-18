'''

    This code contains  the configuration settings for the FDI localization model and a DataPreparer 
    class that provides methods for loading, splitting, and preprocessing the data, 
    including min-max normalization based on pre-computed min and max values for each feature.

'''

from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from deep_models.models import TimeseriesDataset
import pandas as pd
import numpy as np


@dataclass
class Config:
    epochs: int = 2
    batch_size: int = 64
    seq_len: int = 10
    learning_rate: float = 0.001
    models_folder: str = "./models"
    save_models: bool = True
    # model_name: str = "LSTM_autoencoder_for_FDI_10.pth"
    model_name: str = "test_v2.pth"
    normalization_model: str = "min_max_scaler_AE_for_FDI.save"

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


class DataPreparer:

    def __init__(self, config):
        self.config = config

    def load_data(self, path_for_data, features, percentage):
        df = pd.read_csv(path_for_data)
        df.dropna(inplace=True)
        df = df.loc[:, features]
        length = len(df)
        if 0 < percentage <= 1.0:
            number_of_rows = int(percentage * length)
            return df[:number_of_rows]
        else:
            raise ValueError(
                "values in percentage should be in the range (0, 1]")

    def split_data(self, df, test_size=0.10):
        split_index = int(len(df) * test_size)
        return df[:split_index], df[split_index:]

    def to_sequences(self, df, seq_size=10):
        """
        Convert data to sequences of given size.
            Data format is expected to be a pandas DataFrame.

        Returns a numpy array of shape (num_sequences, seq_size, num_features).

            """
        sequences = []
        for i in range(len(df) - seq_size + 1):
            sequences.append(df[i:i + seq_size])
        return np.array(sequences)

    def to_sequences_(self, df, seq_size, Index_col="Unnamed: 0"):
        """
        Creates sequences from the data by checking if the index is consecutive.
        Only sequences with consecutive indices are included.

        Data format is expected to be a pandas DataFrame with the first column as index and the rest as features.
        """
        data_values = []
        index_values = df.iloc[:, 0].astype(int).values  # Ensure integer type

        for i in range(len(df) - seq_size + 1):
            seq_index = index_values[i:i + seq_size]
            # Check if all indices are consecutive
            if np.all(np.diff(seq_index) == 1):
                seq = df.iloc[i:(i + seq_size)]
                data_values.append(seq.drop(columns=[Index_col]).values)

        return np.array(data_values)

    def create_dataloader(self, sequences, batch_size, shuffle=False):
        '''  Convert sequences to PyTorch tensors and create a DataLoader for training or validation.   

        Parameters:
            sequences (numpy array): The input data sequences to be converted to tensors.
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data in the DataLoader (default: False).
        Returns:
            DataLoader: A PyTorch DataLoader containing the prepared tensors for training or validation.
        '''

        dataset = TimeseriesDataset(sequences)
        loader = DataLoader(
            dataset, batch_size, shuffle=shuffle)

        return loader

    def min_max_normalizer(self, df, feature_list, min_max_csv_path, mode="normalize"):
        """
        Normalize selected features in a DataFrame using Min-Max scaling,
        based on precomputed min-max values from a CSV file.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the features to be normalized.
            feature_list (list): A list of features to normalize.
            min_max_csv: Path to the CSV file containing the min and max values for each feature.
            mode (str): The mode of operation ("normalize" or "denormalize").

        Returns:
            pd.DataFrame: A DataFrame with the same structure as `df`, but with normalized values for selected features.
        """
        # Load the min-max values from CSV
        min_max_df = pd.read_csv(min_max_csv_path)

        # Convert to dictionary for fast lookup
        min_vals = dict(zip(min_max_df["Feature"], min_max_df["Min"]))
        max_vals = dict(zip(min_max_df["Feature"], min_max_df["Max"]))

        # Create a copy of the original DataFrame
        normalized_df = df.copy()

        # Apply Min-Max Normalization to selected features
        for feature in feature_list:
            # print(feature)
            if feature in min_vals and feature in max_vals:
                min_val = min_vals[feature]
                max_val = max_vals[feature]

                # Avoid division by zero
                if max_val != min_val:
                    if mode == "normalize":
                        normalized_df[feature] = (
                            df[feature] - min_val) / (max_val - min_val)
                    elif mode == "denormalize":
                        normalized_df[feature] = (
                            df[feature] * (max_val - min_val)) + min_val
                else:
                    normalized_df[feature] = 0  # Assign 0 if no range

        return normalized_df[feature_list]
