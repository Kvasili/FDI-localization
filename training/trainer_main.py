
'''

    This is the main trainign script for the FDI localization model. 
    It defines the Trainer class which encapsulates the training and evaluation logic for the model. 
    The Trainer class has methods for training the model on a given dataset, 
    evaluating the model on a validation set, and fitting the model for a specified number of epochs.


    USAGE:
    run this code from the main directory of the project using the command:
    python -m training.trainer_main


    BEWARE DO NOT USE THE FOLLOWING COMMAND TO RUN THE CODE AS IT WILL CAUSE ISSUES WITH RELATIVE PATHS:
    THIS COMMAND WILL CAUSE PYTHON TO SET THE CURRENT WORKIGN DIRECTORY TO THE TRAINING FOLDER 
    python training/trainer_main.py 

'''


import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from training.trainer import Trainer
from deep_models.models import TimeseriesDataset, FeatureAttention, FeatureAttentionOverTime, AttentionLSTMAutoencoder
from configurations.config import DataPreparer, Config
import os


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
