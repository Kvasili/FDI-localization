
'''

    This is the main trainign script for the FDI localization model. 
    It defines the Trainer class which encapsulates the training and evaluation logic for the model. 
    The Trainer class has methods for training the model on a given dataset, 
    evaluating the model on a validation set, and fitting the model for a specified number of epochs.

'''

import torch
from FDI_localization.config.config import Config

config = Config()
