
'''

    This is the main trainign script for the FDI localization model. 
    It defines the Trainer class which encapsulates the training and evaluation logic for the model. 
    The Trainer class has methods for training the model on a given dataset, 
    evaluating the model on a validation set, and fitting the model for a specified number of epochs.


    USAGE:
    run this code from the main directory of the project using the command:
    python training/trainer_main.py 

'''

from training.trainer import Trainer
from deep_models.models import TimeseriesDataset, FeatureAttention, FeatureAttentionOverTime, AttentionLSTMAutoencoder
from configurations.config import DataPreparer
from configurations.config import Config
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch
import os
import sys
# Ensure the project root is on sys.path when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# change the current working directory to the project root to ensure relative paths work correctly

# current working directory is C:\Users\kvasi\OneDrive - purdue.edu\projects\Autonomous Control System\codes\windowSHAP-Kalman\FDI_localization
cwd = "C:\\Users\\kvasi\\OneDrive - purdue.edu\\projects\\Autonomous Control System\\codes\\windowSHAP-Kalman\\FDI_localization"
os.chdir(cwd)


def main():
    # Load configuration
    config = Config()
    # Load the class with the data preparation methods
    data_preparer = DataPreparer(config)
    columns = config.columns
    path_for_training = config.path_for_training
    path_for_validation = config.path_for_validation

    ##################  TRAINING DATASET  ##########################
    all_trainX = []

    for root, dirs, files in os.walk(path_for_training):
        for file in files:
            if file.lower().endswith('.csv'):
                # path for training data
                filename_normal = os.path.join(root, file)

                df_data = data_preparer.load_data(filename_normal, columns, 1)

                # Normalize the data
                df_data.loc[:, columns] = data_preparer.min_max_normalizer(
                    df_data, columns, config.path_for_normalization_summary, mode="normalize")

                # Convert the data to sequences
                trainX = data_preparer.to_sequences(
                    df_data, seq_size=config.seq_len)

                if trainX is not None and len(trainX) > 0:
                    all_trainX.append(trainX)

        # Concatenate into a single array for training
    # Shape: [total_sequences, 10, num_features]
    all_trainX = np.concatenate(all_trainX, axis=0)
    print(f"Total training sequences: {len(all_trainX)}")


if __name__ == "__main__":
    main()
