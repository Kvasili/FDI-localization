'''
    This file contains the AutoencoderDetector class, which is responsible for computing the reconstruction error of an
    input sequence using the trained autoencoder model. The class includes a method to calculate the Mean Absolute Error (MAE)
    between the input and the reconstructed output, as well as to extract attention weights and matrices if the model includes 
    an attention mechanism. This is used for fault detection in the FDI localization framework.

'''


import torch
import numpy as np


class AutoencoderDetector:

    def __init__(self, model):
        self.model = model

    def reconstruct_error(self, input_data, selected_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        """
        Computes the MAE reconstruction error for the input sequence.
        Expects input_data of shape: (1, window_size, num_features)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        with torch.no_grad():
            # Convert numpy array to torch tensor if needed
            if isinstance(input_data, np.ndarray):
                input_data = torch.tensor(input_data, dtype=torch.float32)

                # Ensure batch dimension
            if input_data.dim() == 2:
                # (seq_len, num_features) -> (1, seq_len, num_features)
                input_data = input_data.unsqueeze(0)

            input_data = input_data.to(device)
            # print('input data:', input_data)
            outputs, attn_weights, attn_matrix = self.model(input_data)

            # shape: (seq_len, num_features)
            input_array = input_data[0].detach().cpu().numpy()
            output_array = outputs[0].detach().cpu().numpy()

            # Select only desired columns
            input_sel = input_array[:, selected_indices]
            output_sel = output_array[:, selected_indices]

            # Compute absolute error
            errors = np.abs(input_sel - output_sel)

            # Extract mean reconstruction error over the selected features and all time steps
            # mean over time for each feature
            mae_per_feature = errors.mean(axis=0)

            # Mean over all time steps and selected features -> single scalar
            mae = errors.mean()

            return (mae, outputs.detach().cpu().numpy(), attn_weights.detach().cpu().numpy(), attn_matrix.detach().cpu().numpy())
