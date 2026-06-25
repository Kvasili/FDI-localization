"""
LSTM Autoencoder for sensor diagnostics with windowSHAP analysis.

Real-time FDI localization pipeline: streams reactor data from PostgreSQL,
runs Kalman-filter-based neutron estimation, detects anomalies via an
autoencoder, and exports SHAP attributions to CSV.

@Author: Konstantinos Vasili
@Date: 06/15/2026

@USAGE
    python main_organized.py
"""

from __future__ import annotations

import csv
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from configurations.inference_config import Config
from data.data_preparer import DataPreparer
from database_scripts.csv_to_postgres import DatabaseHandler
from deep_models.localization import AutoencoderDetector
from deep_models.models import AttentionLSTMAutoencoder
from explainability.shapBinding import SHAPBinding
from physics.reactor_kinetics import (
    KalmanPKE,
    build_continuous_pke,
    compute_reactivity,
    linear_model_predict,
)

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KF_MEASUREMENT_PHASE_STEPS = 2500
ANOMALY_ERROR_THRESHOLD = 0.01
SHAP_SUB_WINDOW_SIZE = 5
KALMAN_DT = 1.0
BASELINE_REACTIVITY = 5710
DB_POLL_LIMIT = 10
DB_POLL_SLEEP_S = 0.1

DB_CONFIG = {
    "dbname": "reactor_data",
    "user": "postgres",
    "password": "ellia94",
    "host": "localhost",
    "port": "5432",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filename: str, cols_to_be_read: list[str], percentage: float) -> pd.DataFrame:
    """Load a CSV subset, keeping only selected columns."""
    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    df = df.loc[:, cols_to_be_read]

    if not (0 < percentage <= 1.0):
        raise ValueError("values in percentage should be in the range (0, 1]")

    number_of_rows = int(percentage * len(df))
    return df[:number_of_rows]


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def build_csv_headers(config: Config) -> tuple[list[str], list[str]]:
    """Return (full headers, SHAP field names)."""
    shap_names = [f"SHAP_{f}" for f in config.columns]
    headers = (
        ["time"]
        + config.columns
        + [
            "kf-estimated-cps",
            "kf-estimated-nf2-log",
            "kf-estimated-nf3-pwr",
            "kf-estimated-nf4-flux",
            "reconstruction_error",
        ]
        + shap_names
    )
    return headers, shap_names


def init_csv_file(csv_filename: str, headers: list[str]) -> None:
    """Create the output CSV with headers if it does not exist."""
    if os.path.isfile(csv_filename):
        return
    with open(csv_filename, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=headers).writeheader()


def append_csv_row(csv_filename: str, headers: list[str], row: dict[str, Any]) -> None:
    with open(csv_filename, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=headers).writerow(row)


# ---------------------------------------------------------------------------
# Model & Kalman setup
# ---------------------------------------------------------------------------

def load_autoencoder(config: Config, device: torch.device) -> AttentionLSTMAutoencoder:
    model = AttentionLSTMAutoencoder(
        input_dim=len(config.columns),
        seq_len=config.seq_len,
    )
    model.load_state_dict(
        torch.load(
            os.path.join(config.models_folder, config.model_name),
            map_location=device,
        )
    )
    model.to(device)
    model.eval()
    return model


def create_kalman_filter(initial_cps: float) -> KalmanPKE:
    """Build and initialize the point-kinetics Kalman filter."""
    L, beta_total, beta_i, lambda_i, C = build_continuous_pke()

    C0_6g = (beta_i / lambda_i) * initial_cps / L
    print(f"Initial delayed neutron precursor concentrations: {C0_6g}")

    x0 = np.concatenate(([initial_cps], C0_6g))
    P0 = np.eye(7) * 1.0
    Q = np.eye(7) * 1e-4
    Q[0, 0] = (0.02 * initial_cps) ** 2
    R = np.array([[(0.001 * initial_cps) ** 2]])

    return KalmanPKE(x0, P0, Q, R, KALMAN_DT, L, beta_total, beta_i, lambda_i, C)


def connect_database() -> Any:
    """Return an open PostgreSQL connection, or None on failure."""
    handler = DatabaseHandler(**DB_CONFIG)
    conn = handler.connect_to_db()
    if conn is None:
        print("Failed to connect to database.")
        return None
    conn.set_session(autocommit=True)
    return conn


def fetch_max_table_time(conn) -> int:
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(time) FROM testing_table;")
        return cursor.fetchone()[0] or 0
    except Exception as e:
        print(f"Error fetching max id: {e}")
        return 0


# ---------------------------------------------------------------------------
# Per-step processing
# ---------------------------------------------------------------------------

@dataclass
class KalmanEstimates:
    cps: float
    nf2_log: float
    nf3_pwr: float
    nf4_flux: float


def run_kalman_step(
    kf: KalmanPKE,
    step_idx: int,
    measurement: float,
    reactivity: float,
) -> KalmanEstimates:
    """
    Update or predict the Kalman filter depending on the current phase.

    During the first KF_MEASUREMENT_PHASE_STEPS, measurements correct the
    state; afterward the filter runs in prediction-only mode.
    """
    if step_idx < KF_MEASUREMENT_PHASE_STEPS:
        cps = kf.step_with_measurement(measurement, reactivity)[0]
    else:
        cps = kf.step_no_measurement(reactivity)[0]

    nf2_log, nf3_pwr, nf4_flux = linear_model_predict(cps)
    return KalmanEstimates(cps=cps, nf2_log=nf2_log, nf3_pwr=nf3_pwr, nf4_flux=nf4_flux)


def normalize_sequence(
    data_preparer: DataPreparer,
    sequence_df: pd.DataFrame,
    config: Config,
) -> np.ndarray:
    sensor_df = sequence_df.loc[:, config.columns]
    return data_preparer.min_max_normalizer(
        sensor_df,
        config.columns,
        config.path_for_normalization_summary,
        mode="normalize",
    ).values


def build_kf_background(
    sequence_df: pd.DataFrame,
    sequence_df_sensors: pd.DataFrame,
    data_preparer: DataPreparer,
    config: Config,
) -> np.ndarray:
    """
    Replace redundant sensor channels with KF estimates for SHAP background.

    Keeps column names and order unchanged so the normalizer remains valid.
    """
    last_col_values = sequence_df.iloc[:, -1].values
    second_last_col_values = sequence_df.iloc[:, -2].values
    fourth_last_col_values = sequence_df.iloc[:, -4].values

    background_df = sequence_df_sensors.copy()
    background_df.iloc[:, 0] = fourth_last_col_values
    background_df.iloc[:, 2] = second_last_col_values
    background_df.iloc[:, 3] = last_col_values

    return data_preparer.min_max_normalizer(
        background_df,
        config.columns,
        config.path_for_normalization_summary,
        mode="normalize",
    ).values


def explain_shap_values(
    window_shap: SHAPBinding,
    normalized_sequence: np.ndarray,
    normalized_background: np.ndarray,
    config: Config,
    device: torch.device,
    shap_names: list[str],
) -> dict[str, float]:
    current_seq = normalized_sequence.reshape(1, config.seq_len, config.input_dim)
    current_background = normalized_background.reshape(1, config.seq_len, config.input_dim)
    shap_values = window_shap.explain_anomaly(
        torch.tensor(current_seq, dtype=torch.float32).to(device),
        torch.tensor(current_background, dtype=torch.float32).to(device),
    )
    mean_shap = np.mean(shap_values, axis=1).flatten()
    return {shap_names[i]: mean_shap[i] for i in range(len(shap_names))}


def write_kf_estimates_to_buffer(
    df_buffer: pd.DataFrame,
    step_idx: int,
    estimates: KalmanEstimates,
) -> None:
    df_buffer.at[step_idx, "kf-estimated-cps"] = estimates.cps
    df_buffer.at[step_idx, "kf-estimated-nf2-log"] = estimates.nf2_log
    df_buffer.at[step_idx, "kf-estimated-nf3-pwr"] = estimates.nf3_pwr
    df_buffer.at[step_idx, "kf-estimated-nf4-flux"] = estimates.nf4_flux


# ---------------------------------------------------------------------------
# Streaming loop state
# ---------------------------------------------------------------------------

@dataclass
class StreamingState:
    df_buffer: pd.DataFrame = field(default_factory=pd.DataFrame)
    last_id: int = 0
    last_processed_idx: int = 0
    nfd_1_cps_est: float = 0.0
    true_vals: list = field(default_factory=list)
    est_vals: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def fetch_new_chunk(conn, last_id: int) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM testing_table
        WHERE time > {last_id}
        ORDER BY time
        LIMIT {DB_POLL_LIMIT};
    """
    return pd.read_sql_query(query, conn)


def process_buffer_rows(
    state: StreamingState,
    config: Config,
    data_preparer: DataPreparer,
    kf: KalmanPKE,
    detector: AutoencoderDetector,
    window_shap: SHAPBinding,
    device: torch.device,
    csv_filename: str,
    headers: list[str],
    shap_names: list[str],
    window_size: int,
) -> None:
    """Process all unprocessed rows currently held in the buffer."""
    for start_idx in range(state.last_processed_idx, len(state.df_buffer)):
        if start_idx < window_size - 1:
            continue

        current_row = state.df_buffer.copy().iloc[start_idx - 1, :].to_dict()
        buffer_row = state.df_buffer.loc[start_idx]

        meas = buffer_row["nfd-1-cps"]
        rho = compute_reactivity(
            buffer_row["ss1-position"],
            buffer_row["ss2-position"],
            buffer_row["rr-position"],
            state.nfd_1_cps_est,
            baseline_reactiviy=BASELINE_REACTIVITY,
        )

        estimates = run_kalman_step(kf, start_idx, meas, rho)
        state.nfd_1_cps_est = estimates.cps
        state.true_vals.append(meas)
        state.est_vals.append(estimates.cps)
        write_kf_estimates_to_buffer(state.df_buffer, start_idx, estimates)

        sequence_df = state.df_buffer.loc[start_idx - window_size + 1 : start_idx].copy()
        sequence_df_sensors = sequence_df.loc[:, config.columns]

        normalized_sequence = normalize_sequence(data_preparer, sequence_df, config)
        normalized_background = build_kf_background(
            sequence_df, sequence_df_sensors, data_preparer, config
        )

        current_seq = normalized_sequence.reshape(1, config.seq_len, config.input_dim)
        error, _, _, _ = detector.reconstruct_error(current_seq)
        state.errors.append(error)

        if error > ANOMALY_ERROR_THRESHOLD:
            shap_dict = explain_shap_values(
                window_shap,
                normalized_sequence,
                normalized_background,
                config,
                device,
                shap_names,
            )
        else:
            shap_dict = {name: 0 for name in shap_names}

        output_row = dict(current_row)
        output_row["reconstruction_error"] = error
        output_row.update(shap_dict)
        append_csv_row(csv_filename, headers, output_row)

    state.last_processed_idx = len(state.df_buffer)


def run_streaming_pipeline(
    conn,
    max_id: int,
    config: Config,
    data_preparer: DataPreparer,
    kf: KalmanPKE,
    detector: AutoencoderDetector,
    window_shap: SHAPBinding,
    device: torch.device,
    csv_filename: str,
    headers: list[str],
    shap_names: list[str],
) -> StreamingState:
    """Poll the database and process rows until the table end is reached."""
    state = StreamingState()
    window_size = config.seq_len

    while True:
        chunk = fetch_new_chunk(conn, state.last_id)
        if chunk.empty:
            time.sleep(DB_POLL_SLEEP_S)
            continue

        state.last_id = chunk["time"].max()
        state.df_buffer = pd.concat([state.df_buffer, chunk], ignore_index=True)

        process_buffer_rows(
            state,
            config,
            data_preparer,
            kf,
            detector,
            window_shap,
            device,
            csv_filename,
            headers,
            shap_names,
            window_size,
        )

        if state.last_id >= max_id:
            print("Reached end of table. Exiting loop.")
            break

    return state


# ---------------------------------------------------------------------------
# Visualization & orchestration
# ---------------------------------------------------------------------------

def plot_diagnostics(errors: list, true_vals: list, est_vals: list) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(errors, label="Reconstruction Error (MAE)", color="red")
    plt.xlabel("Time step (s)")
    plt.ylabel("MAE Reconstruction Error")
    plt.title("Reconstruction Error over Time")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(true_vals, label="True Neutron Population (CPS)", color="blue")
    plt.plot(est_vals, label="KF Estimated Neutron Population (CPS)", color="orange")
    plt.xlabel("Time step (s)")
    plt.ylabel("Neutron Population (CPS)")
    plt.title("True vs KF Estimated Neutron Population")
    plt.legend()
    plt.show()


def main() -> None:
    start_time = time.time()

    config = Config()
    data_preparer = DataPreparer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Used only for Kalman initial conditions
    df = load_data(config.path_for_abnormal_data, config.columns, 1.0)
    print(f"Data shape: {df.shape}")

    headers, shap_names = build_csv_headers(config)
    csv_filename = config.csv_file
    init_csv_file(csv_filename, headers)

    model = load_autoencoder(config, device)
    window_shap = SHAPBinding(model=model, sub_window_size=SHAP_SUB_WINDOW_SIZE)
    detector = AutoencoderDetector(model)

    initial_cps = df["nfd-1-cps"].iloc[0]
    kf = create_kalman_filter(initial_cps)

    conn = connect_database()
    if conn is None:
        return

    max_id = fetch_max_table_time(conn)
    state = StreamingState()

    try:
        state = run_streaming_pipeline(
            conn,
            max_id,
            config,
            data_preparer,
            kf,
            detector,
            window_shap,
            device,
            csv_filename,
            headers,
            shap_names,
        )
    except KeyboardInterrupt:
        print("Interrupted by user. Closing database connection.")
    finally:
        conn.close()
        print("Database connection closed.")
        elapsed_min = (time.time() - start_time) / 60
        print(f"Total execution time: {elapsed_min:.2f} minutes")
        plot_diagnostics(state.errors, state.true_vals, state.est_vals)


if __name__ == "__main__":
    main()
