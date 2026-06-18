
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
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from training.trainer import Trainer
from deep_models.models import TimeseriesDataset, AttentionLSTMAutoencoder
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
    print(f"Total training sequences: {all_trainX.shape}")

    ##################  VALIDATION DATASET  ##########################
    all_valX = []

    for root, dirs, files in os.walk(path_for_validation):
        for file in files:
            if file.lower().endswith('.csv'):

                filename = os.path.join(root, file)
                # print(f"Processing file: {filename}")

                df_data = data_preparer.load_data(filename, columns, 1)

                # Normalize the data
                df_data.loc[:, columns] = data_preparer.min_max_normalizer(
                    df_data, columns, config.path_for_normalization_summary, mode="normalize")

                valX = data_preparer.to_sequences(
                    df_data, seq_size=config.seq_len)

                if valX is not None and len(valX) > 0:
                    all_valX.append(valX)

        # Concatenate into a single array for training

    all_valX = np.concatenate(all_valX, axis=0)
    print("Total validation sequences:", all_valX.shape)

    # Prepare DataLoaders for training and validation
    train_loader = DataPreparer.prepare_tensors(
        all_trainX, batch_size=config.batch_size, shuffle=False)
    val_loader = DataPreparer.prepare_tensors(
        all_valX, batch_size=config.batch_size, shuffle=True)

    # Initialize the model, criterion, optimizer, and device
    model = AttentionLSTMAutoencoder(
        input_dim=all_trainX.shape[2], seq_len=all_trainX.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # initialize the trainer and fit the model
    trainer = Trainer(model, optimizer, criterion, device)
    train_losses, val_losses = trainer.fit(
        train_loader, val_loader, epochs=config.epochs)
    trainer.save_model(os.path.join(
        config.models_folder, config.model_name))

    # plot training and validation losses
    trainer.plot_train_val_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
