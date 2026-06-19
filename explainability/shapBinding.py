

import torch
import numpy as np
from explainability.WindowSHAP.windowshap import StationaryWindowSHAP


class SHAPBinding:

    '''A binding class to compute SHAP values for LSTM sequences with Pytorch models
       and WindowSHAP package
    '''

    def __init__(self, model, sub_window_size):
        self.model = model
        self.sub_window_size = sub_window_size

    def prediction_function(self, input_data):
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

            input_data = input_data.to(device)
            # print('input data:', input_data)
            outputs, _, _ = self.model(input_data)

            reconstruction_errors = torch.mean(
                # mean over features, shape: [batch, seq_len]
                torch.abs(input_data - outputs), dim=2)
            # reconstruction_errors = torch.mean(
            #     torch.abs(input_data[:, selected_indices] - outputs[:, selected_indices]), dim=2)
            # mean over time, shape: [batch]
            mae = torch.mean(reconstruction_errors, dim=1)
            # print('MAE:', mae)
        return mae.cpu().numpy().reshape(-1, 1)

    def explain_anomaly(self, current_seq, background_seq):
        """
        Computes SHAP values for the current sequence if an anomaly is detected.
        Both current_seq and background_seq must be of shape (1, window_size, num_features)
        """
        explainer = StationaryWindowSHAP(
            model=self.prediction_function,
            window_len=self.sub_window_size,
            B_ts=background_seq.cpu().numpy(),
            test_ts=current_seq.cpu().numpy(),
            model_type='lstm'
        )
        return explainer.shap_values()

    def CreateBackground(self, data, scaler=None):

        # if scaler:
        # if data are scaled unscale them
        data = scaler.inverse_transform(data.squeeze())

        b_ts = data.copy()
        for i in [3, 5, 7]:
            # hardcode the active states to be 0 - no movement
            b_ts[:, i] = 0
        for i in [0, 1, 2, 4, 6, 8]:
            # set the rest of the signals with the first value of the sequence
            b_ts[:, i] = data[0, i]

        b_ts = scaler.transform(b_ts)

        b_ts = np.array(b_ts[None, :, :])

        # print('Background Shape:', np.shape(b_ts))

        return b_ts
