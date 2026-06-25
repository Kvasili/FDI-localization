

'''

    @Author: Konstantinos Vasili
    @Date: 06/15/2026
    @Description:

    
    LSTM Autoencoder for sensor diagnostics for windowSHAP analysis 

    
    @USAGE
    python main_averaging.py

    
'''

from configurations.inference_config import Config
from deep_models.models import AttentionLSTMAutoencoder
from deep_models.localization import AutoencoderDetector
from explainability.shapBinding import SHAPBinding
from data.data_preparer import DataPreparer
from physics.reactor_kinetics import KalmanPKE, build_continuous_pke, compute_reactivity, linear_model_predict
from database_scripts.csv_to_postgres import DatabaseHandler
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import csv
import warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")


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


def main():

    mode = ["Kalman", "averaging"]
    mode = mode[1]
    print(mode)

    # read steady average data for the control signals
    avg_df = pd.read_csv("./datasets/Power_Cycle_4794407_avg.csv")
    print(avg_df)

    # current time at the beggining of the script execution
    start_time = time.time()

    config = Config()
    data_preparer = DataPreparer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset exclude the first column
    df = load_data(config.path_for_abnormal_data, config.columns, 1.0)
    print(df.head())
    print(f"Data shape: {df.shape}")

    ##################################################################
    # export data in real time to a csv file
    # Write header once (only if file doesn't exist)
    # shap fields names
    shap_names = [f'SHAP_{f}' for f in config.columns]
    headers = ['time'] + config.columns + ['avg-estimated-cps'] + ['avg-estimated-nf2-log'] + ['avg-estimated-nf3-pwr'] + ['avg-estimated-nf4-flux'] + \
        ['reconstruction_error'] + shap_names

    # Run this once to create the CSV file with headers if it doesn't exist
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

    # Connect to the database the password and the rest credentials should be stored in environment variables or a config file for security, but hardcoded here for simplicity
    database_handler = DatabaseHandler(
        dbname="reactor_data", user="postgres", password="ellia94", host="localhost", port="5432")
    conn = database_handler.connect_to_db()

    if conn is None:
        print("Failed to connect to database.")
        return

    conn.set_session(autocommit=True)
    window_size = config.seq_len
    df_buffer = pd.DataFrame()
    last_id = 0
    last_processed_idx = 0

    errors = []

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

                # Get the sequneces of the 10 prior seconds
                sequence_df = df_buffer.loc[start_idx -
                                            window_size + 1: start_idx].copy()
                # This will get rid of the Index column
                sequence_df = sequence_df.loc[:, config.columns]

                # print('current sequence:')
                # print(sequence_df)
                background_df = sequence_df.copy()
                # background_df.iloc[:, 0:4] = avg_df.iloc[:, 1:5].values
                # This does not replaces the nfd-log
                # background_df.iloc[:, [0, 2, 3]
                #                    ] = avg_df.iloc[:, [1, 3, 4]].values
                background_df[["nfd-1-cps", "nfd-3-pwr", "nfd-4-flux"]] = avg_df[[
                    "nfd-1-cps-avg", "nfd-3-pwr-avg", "nfd-4-flux-avg"]].values

                # Normalize the data and convert them to numpy array
                normalized_sequence = data_preparer.min_max_normalizer(
                    sequence_df, config.columns, config.path_for_normalization_summary, mode="normalize").values

                normalized_background = data_preparer.min_max_normalizer(
                    background_df, config.columns, config.path_for_normalization_summary, mode="normalize").values

                # print('current sequence:')
                # print(normalized_sequence)
                # print(normalized_background)

                # # # Reshape for LSTM input (1, seq_len, input_dim)
                current_seq = normalized_sequence.reshape(
                    1, config.seq_len, config.input_dim)

                detector = AutoencoderDetector(model_)
                error, output_seq, attn_weights, attn_matrix = detector.reconstruct_error(
                    current_seq)

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

    # Run the function
if __name__ == "__main__":

    main()
