

'''

    @Author: Konstantinos Vasili
    @Date: 06/15/2026
    @Description:

    
    LSTM Autoencoder for sensor diagnostics for windowSHAP analysis 

    
    @USAGE
    python Attention_autoencoder_evaluate_Kalman_SHAP_DB.py

    
'''

from configurations.inference_config import Config
from deep_models.models import AttentionLSTMAutoencoder
from deep_models.localization import AutoencoderDetector
from explainability.shapBinding import SHAPBinding
from data.data_preparer import DataPreparer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import os
from scipy.linalg import expm
import csv

import time
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")


# Connect to PostgreSQL database
def connect_to_db():
    '''Connect to the PostgreSQL database using psycopg2.'''
    return psycopg2.connect(
        dbname="reactor_data",
        user="postgres",
        password="ellia94",
        host="localhost",
        port="5432"
    )


def load_data(filename, cols_to_be_read, percentage):
    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    df = df.loc[:, cols_to_be_read]
    length = len(df)
    if 0 < percentage <= 1.0:
        number_of_rows = int(percentage * length)
        return df[:number_of_rows]
    else:
        raise ValueError("values in percentage should be in the range (0, 1]")


def compute_reactivity(SS1, SS2, RR, No, baseline_reactiviy):
    '''Computes the reactivity (rho) based on the positions of the control rods. Neutron population is used
       as a proxy for temperature feedback. The coefficients in the polynomial are hypothetical and should be 
       replaced with actual values from the reactor physics data.'''

    SS1rho = -0.0401*(SS1**3) + 3.8511*(SS1**2) - 21.994*SS1 + 37.296
    SS2rho = -0.0203*(SS2**3) + 2.0107*(SS2**2) - 15.201*SS2 + 27.103
    RRrho = -0.0029*(RR**3) + 0.2693*(RR**2) - 0.5052*RR + 0.2674

    # Temperature reactivity feedback
    rho_T = 1.1e-6 * No

    # Convert from pcm → dk/k
    rho = (SS1rho + SS2rho + RRrho - rho_T -
           baseline_reactiviy) * 1e-5  # 5586.9
    return rho


class KalmanPKE:
    def __init__(self, x0, P0, Q, R, dt, L, beta_total, beta_i, lambda_i, C):
        # store current state
        self.x_hat = x0.copy()
        self.P = P0.copy()

        # store initial values for reset()
        self._x0_init = x0.copy()
        self._P0_init = P0.copy()

        self.Q = Q
        self.R = R
        self.dt = dt
        self.L = L
        self.beta_total = beta_total
        self.beta_i = beta_i
        self.lambda_i = lambda_i
        self.C = C

    def step_with_measurement(self, measurement, rho):
        """
        One time-step KF:
        Uses time-varying Ad (ρ changes each step)
        """

        # 1) Build continuous A_c matrix and C
        A_c = np.zeros((7, 7))
        A_c[0, 0] = (rho - self.beta_total) / self.L
        A_c[0, 1:] = self.lambda_i
        A_c[1:, 0] = self.beta_i / self.L
        A_c[1:, 1:] = -np.diag(self.lambda_i)

        # 2) Discretize using matrix exponential
        Ad = expm(A_c * self.dt)

        # 3) Predict
        x_pred = Ad @ self.x_hat
        P_pred = Ad @ self.P @ Ad.T + self.Q

        # 4) Update (Correction)

        y_pred = (self.C @ x_pred).reshape(-1)[0]
        nu = measurement - y_pred
        S = self.C @ P_pred @ self.C.T + self.R
        K = (P_pred @ self.C.T) / S  # Kalman gain

        self.x_hat = x_pred + K * nu
        self.P = (np.eye(7) - K @ self.C) @ P_pred

        return self.x_hat[0]  # return estimated power

    def step_no_measurement(self, rho):
        """
        One time-step KF:
        Uses time-varying Ad (rho changes each step)
        No measurement update
        """

        # 1) Build continuous A_c matrix
        A_c = np.zeros((7, 7))
        A_c[0, 0] = (rho - self.beta_total) / self.L
        A_c[0, 1:] = self.lambda_i
        A_c[1:, 0] = self.beta_i / self.L
        A_c[1:, 1:] = -np.diag(self.lambda_i)

        # 2) Discretize using matrix exponential
        Ad = expm(A_c * self.dt)

        # 3) Predict
        x_pred = (Ad @ self.x_hat)
        P_pred = Ad @ self.P @ Ad.T + self.Q

        self.x_hat = x_pred
        self.P = P_pred

        # return estimated power without measurment update
        return self.x_hat[0]


def build_continuous_pke():
    """
    1 prompt neutron population + 6 delayed groups
    """
    L = 1e-5  # neutron generation time [s] (example)

    # multiply betas with 0.7
    beta_i = 0.7 * np.array([0.00025, 0.00125, 0.00120,
                             0.00260, 0.00080, 0.00040])
    lambda_i = 0.7 * np.array([0.0124, 0.0305, 0.111,
                               0.301, 1.14, 3.01])

    beta_total = beta_i.sum()

    n_x = 7  # state space dimension
    # This creates the observation matrix C = [1 0 0 0 0 0 0], meaning we only measure the neutron population and not the precursor concentrations.
    C = np.zeros((1, n_x))
    C[0, 0] = 1.0

    return L, beta_total, beta_i, lambda_i, C


def linear_model_predict(est_Kalman):
    '''A simple linear model to predict the other abundant signals based on the estimated neutron population from the KF.'''

    nf2_log = 2.1989E-06 * est_Kalman + 1.072803448
    nf3_pwr = 2.3593E-06 * est_Kalman + 0.179082421
    nf4_flux = 2.19132E-06 * est_Kalman + 0.182602579

    return nf2_log, nf3_pwr, nf4_flux


def main():

    # current time at the beggining of the script execution
    start_time = time.time()

    config = Config()
    data_preparer = DataPreparer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset exclude the first column
    df = load_data(config.path_for_abnormal_data, config.columns, 1.0)
    print(f"Data shape: {df.shape}")

    ##################################################################
    # export data in real time to a csv file
    # Write header once (only if file doesn't exist)
    # shap fields names
    shap_names = [f'SHAP_{f}' for f in config.columns]
    headers = ['time'] + config.columns + ['kf-estimated-cps'] + ['kf-estimated-nf2-log'] + ['kf-estimated-nf3-pwr'] + ['kf-estimated-nf4-flux'] + \
        ['reconstruction_error'] + shap_names

    csv_filename = config.csv_file
    if not os.path.isfile(csv_filename):
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    ###################################################################

    # Load the models for the KF and the autoencoder
    model_ = AttentionLSTMAutoencoder(
        input_dim=len(config.columns), seq_len=config.seq_len)

    model_.load_state_dict(torch.load(os.path.join(
        config.models_folder, config.model_name), map_location=device))
    model_.to(device)
    model_.eval()

    # instantiate windowSHAP
    window_shap = SHAPBinding(model=model_, sub_window_size=5)

    # PKE parameters
    L, beta_total, beta_i, lambda_i, C = build_continuous_pke()
    # Sampling interval:
    dt = 1.0  # seconds

    # Need to change this in the future
    #######################################
    N0 = df["nfd-1-cps"].iloc[0]
    #######################################

    # --- Initial conditions ---
    C0_6g = (beta_i / lambda_i) * N0 / L
    print(f"Initial delayed neutron precursor concentrations: {C0_6g}")
    x0 = np.concatenate(([N0], C0_6g))
    P0 = np.eye(7) * 1.0
    Q = np.eye(7) * 1e-4  # encodes how much we trust the model
    # adjust to 1% of nominal neutron count
    Q[0, 0] = (0.02 * N0)**2   # 2% uncertainty in neutron dynamics

    # R = np.array([[1e-2]])
    # 0.5% measurement noise
    R = np.array([[(0.001 * N0)**2]])  # 1% measurement noise
    KF = KalmanPKE(x0, P0, Q, R, dt, L, beta_total, beta_i, lambda_i, C)

    # Connect to the database
    conn = connect_to_db()
    conn.set_session(autocommit=True)
    window_size = config.seq_len
    df_buffer = pd.DataFrame()
    last_id = 0
    last_processed_idx = 0

    true_vals = []
    est_vals = []
    errors = []
    est = 0

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(time) FROM testing_table;")
        max_id = cursor.fetchone()[0]
    except Exception as e:
        print(f"Error fetching max id: {e}")
        max_id = 0

    try:
        while True:
            # Read new data from DB
            query = f"""
                SELECT *
                FROM testing_table
                WHERE time > {last_id}
                ORDER BY time
                LIMIT 10;
            """

            chunk = pd.read_sql_query(query, conn)

            if chunk.empty:
                time.sleep(0.1)
                continue

            # update pointer
            last_id = chunk["time"].max()

            # append to buffer
            df_buffer = pd.concat([df_buffer, chunk], ignore_index=True)

            # Process NEW rows only
            for start_idx in range(last_processed_idx, len(df_buffer)):

                # print("\n--- NEW STEP ---")
                # print(
                #     f"\nProcessing row with time={df_buffer.loc[start_idx, 'time']} (index {start_idx})")

                if start_idx < window_size-1:
                    continue  # skip until we have enough data

                # initialize a row dictionary to store the data for the current step
                row = {}
                current_row = df_buffer.copy().iloc[start_idx-1, :].to_dict()

                meas = df_buffer.loc[start_idx, "nfd-1-cps"]
                SS1 = df_buffer.loc[start_idx, "ss1-position"]
                SS2 = df_buffer.loc[start_idx, "ss2-position"]
                RR = df_buffer.loc[start_idx, "rr-position"]
                rho = compute_reactivity(
                    SS1, SS2, RR, est, baseline_reactiviy=5710)

                if start_idx < 2500:
                    ''' In the early steps, we have a lot of uncertainty in the KF state and we want to use the measurements to correct it. 
                    After some time, we can start relying more on the model predictions and less on the measurements, 
                    especially if we suspect that the measurements might be noisy or unreliable during anomalies. 
                    This is a common strategy in Kalman filtering where you can adjust the measurement noise covariance R or even skip measurement updates based on certain conditions. 
                    Here, we simply choose to use measurements for the first 2500 steps and then switch to prediction-only mode.'''
                    # print(f"KF Update at step {start_idx}")
                    est = KF.step_with_measurement(meas, rho)[0]
                    nf2_log, nf3_pwr, nf4_flux = linear_model_predict(est)

                else:
                    # print(f"KF Prediction only at step {start_idx}")
                    est = KF.step_no_measurement(rho)[0]
                    nf2_log, nf3_pwr, nf4_flux = linear_model_predict(est)
                    time.sleep(0.05)

                # print(f"Step {i}: Measured={meas}, Estimated={est}")
                true_vals.append(meas)
                est_vals.append(est)

                df_buffer.at[start_idx, "kf-estimated-cps"] = est
                df_buffer.at[start_idx, "kf-estimated-nf2-log"] = nf2_log
                df_buffer.at[start_idx, "kf-estimated-nf3-pwr"] = nf3_pwr
                df_buffer.at[start_idx, "kf-estimated-nf4-flux"] = nf4_flux

                # Get the sequneces of the 10 prior seconds
                sequence_df = df_buffer.loc[start_idx -
                                            window_size + 1: start_idx].copy()

                # print(f"ID: {df_buffer.loc[start_idx, 'time']}")
                # # print(f"meas={meas}, SS1={SS1}, SS2={SS2}, RR={RR}")
                # print("Database Sequence (last 10 rows):")
                # print(sequence_df)
                # time.sleep(0.05)

                # Read the sensor data as received from the reactor database and normalize it
                sequence_df_ = sequence_df.loc[:, config.columns]
                # print("Wanted Sequence (last 10 rows):")
                # print(sequence_df_)
                # Normalize the data
                normalized_sequence = data_preparer.min_max_normalizer(
                    sequence_df_, config.columns, config.path_for_normalization_summary, mode="normalize").values

                # # Prepare the sequence for SHAP analysis
                # # Formulate bacgkground by replacing the 4 redundant signals with the KF estimates, while keeping the rest of the signals as they are in the original sequence. This way we can analyze the contribution of the KF estimated signals to the anomaly detection.
                # estimated nfd-4-flux
                last_col_values = sequence_df.iloc[:, -1].values
                # estimated nfd-3-pwr
                second_last_col_values = sequence_df.iloc[:, -2].values
                # estimated nfd-2-log
                third_last_col_values = sequence_df.iloc[:, -3].values
                # estimated nfd-1-cps
                fourth_last_col_values = sequence_df.iloc[:, -4].values

                background_df = sequence_df_.copy()
                # # Replace values of the fourth-third-second-last signals-column (keep name & order the same or the normalizer will be messed up) with the KF estimates
                background_df.iloc[:, 0] = fourth_last_col_values
                background_df.iloc[:, 1] = third_last_col_values
                background_df.iloc[:, 2] = second_last_col_values
                background_df.iloc[:, 3] = last_col_values
                # print("Sequence after dropping last column and replacing first column:")
                # print(background_df)

                normalized_background = data_preparer.min_max_normalizer(
                    background_df, config.columns, config.path_for_normalization_summary, mode="normalize").values
                # # # Reshape for LSTM input (1, seq_len, input_dim)
                current_seq = normalized_sequence.reshape(
                    1, config.seq_len, config.input_dim)

                detector = AutoencoderDetector(model_)

                error, output_seq, attn_weights, attn_matrix = detector.reconstruct_error(
                    current_seq)
                # attn_weights = reconstruct_error(current_seq, model_)[2]
                # attn_matrix = reconstruct_error(current_seq, model_)[3]

                # extract first value of last row of the reconstucted sequence
                nfd_1_seconstruced = output_seq[0, -1, 0]
                # print(
                #     f"Reconstructed nfd-1-cps at current step: {nfd_1_seconstruced}")
                errors.append(error)

                if error > 0.01:
                    # print(f"anomaly at index {start_idx} with error {error}")

                    # Reshape background for LSTM input (1, seq_len, input_dim)
                    current_background = normalized_background.reshape(
                        1, config.seq_len, config.input_dim)

                    shap_values = window_shap.explain_anomaly(
                        torch.tensor(
                            current_seq, dtype=torch.float32).to(device),
                        torch.tensor(current_background,
                                     dtype=torch.float32).to(device)
                    )

                    # print("SHAP values: ", shap_values)
                    mean_shap_values = np.mean(shap_values, axis=1)
                    mean_shap_values = mean_shap_values.flatten()
                    shap_values = {k: mean_shap_values[i]
                                   for i, k in enumerate(shap_names)}
                    # if mean_shap_values[i] > 0.0002
                else:
                    shap_values = {k: 0 for k in shap_names}

                    # Combine all info
                row.update(current_row)
                row['reconstruction_error'] = error
                row.update(shap_values)

                # Write row
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writerow(row)

            # update processed index
            last_processed_idx = len(df_buffer)

            if last_id >= max_id:
                print("Reached end of table. Exiting loop.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user. Closing database connection.")

    finally:
        conn.close()
        print("Database connection closed.")

        # close time
        end_time = time.time()
        print(
            f"Total execution time: {(end_time - start_time)/60:.2f} minutes")

        plt.figure(figsize=(14, 7))
        plt.plot(errors, label="Reconstruction Error (MAE)", color='red')
        plt.xlabel("Time step (s)")
        plt.ylabel("MAE Reconstruction Error")
        plt.title("Reconstruction Error over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

        # plot estiamated vs true neutron population
        plt.figure(figsize=(14, 7))
        plt.plot(true_vals, label="True Neutron Population (CPS)", color='blue')
        plt.plot(
            est_vals, label="KF Estimated Neutron Population (CPS)", color='orange')
        plt.xlabel("Time step (s)")
        plt.ylabel("Neutron Population (CPS)")
        plt.title("True vs KF Estimated Neutron Population")
        plt.legend()
        plt.show()


    # Run the function
if __name__ == "__main__":

    main()


'''

    LSTM Autoencoder for sensor diagnostics for windowSHAP analysis 

    USAGE
    python Attention_autoencoder_evaluate_Kalman_SHAP_DB.py

    TODO:

    - Export everything to a database table for later analysis and visualization (instead of csv)
    - Set a threshold for the SHAP values to triger an alert. For now seems like 0.005 is a good threshold for the mean SHAP values, 
       but this should be further analyzed and validated with more data and domain knowledge.

'''
