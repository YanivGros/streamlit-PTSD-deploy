import os
import re
import boto3
import io
import streamlit as st
from scipy.signal import butter, filtfilt, detrend, find_peaks
from scipy.stats import zscore
import pandas as pd
import numpy as np
import scipy.io as sio
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def process_ppg_signal(
    data, fs=2000, low_cutoff=0.5, high_cutoff=1.5, window_size=10, z_threshold=2
):
    # Low-pass filter to remove background signal
    t = np.linspace(0, len(data) / fs, len(data))
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    b, a = butter(4, low, btype="low")
    background_signal = filtfilt(b, a, data)
    data_detrend = data - background_signal

    # Define a low-pass filter function
    def low_pass_filter(signal, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    # Apply a low-pass filter to the detrended signal
    filtered_signal = low_pass_filter(data_detrend, high_cutoff, fs)

    # Detrend the filtered signal
    detrended_signal = detrend(filtered_signal)

    # Invert the signal to find valleys
    inverted_signal = -detrended_signal

    # Find valleys in the inverted signal
    valleys, _ = find_peaks(
        inverted_signal, distance=fs / 2
    )  # Assuming a minimum distance between valleys

    # Calculate time differences between consecutive valleys
    valley_times = t[valleys]
    valley_index = valleys
    time_diffs = np.diff(valley_times)

    # Estimate the frequency
    if len(time_diffs) > 0:
        avg_time_diff = np.mean(time_diffs)
        estimated_frequency = 1 / avg_time_diff
    else:
        estimated_frequency = 0

    # print(f"Estimated Frequency of the PPG Signal: {estimated_frequency:.2f} Hz")

    # Calculate moving window standard deviation and average
    z_scores = zscore(time_diffs)

    # Filter out outliers based on Z-score threshold
    time_diffs_without_outliers = time_diffs[np.abs(z_scores) < z_threshold]
    time_diffs_without_outliers = time_diffs
    # Calculate moving window standard deviation from the center of the window
    half_window = window_size // 2
    # time_diffs_sd = np.array([np.std(time_diffs_without_outliers[i-half_window:i+half_window+1]) for i in range(half_window, len(time_diffs_without_outliers) - half_window)])
    time_diffs_sd = np.array(
        [
            np.std(
                time_diffs_without_outliers[
                    max(0, i - half_window) : min(
                        len(time_diffs_without_outliers), i + half_window + 1
                    )
                ]
            )
            for i in range(len(time_diffs_without_outliers))
        ]
    )

    # Calculate moving window average from the center of the window
    time_diffs_avg = np.array(
        [
            np.mean(
                time_diffs_without_outliers[
                    max(0, i - half_window) : min(
                        len(time_diffs_without_outliers), i + half_window + 1
                    )
                ]
            )
            for i in range(len(time_diffs_without_outliers))
        ]
    )

    return {
        "estimated_frequency": estimated_frequency,
        "time_diffs_sd": time_diffs_sd,
        "time_diffs_avg": time_diffs_avg,
        "valley_times": valley_times,
        "valley_index": valley_index,
        "time_diffs": time_diffs,
    }


def plot_helper(HR_df, df_avrage, intervals, is_subplots=False, title=""):
    if is_subplots:
        fig = make_subplots(
            rows=6,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
        )
        fig.add_trace(
            go.Scattergl(
                x=df_avrage.index,
                y=df_avrage["GSR - EDA100C-MRI"],
                mode="lines",
                name="GSR",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(x=HR_df.index, y=HR_df["HRV"], mode="lines", name="HRV"),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scattergl(
                x=df_avrage.index,
                y=df_avrage["Pulse - PPG100C"],
                mode="lines",
                name="PPG",
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(x=HR_df.index, y=HR_df["HR"], mode="lines", name="HR"),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=df_avrage.index,
                y=df_avrage["ECG - ECG100C"],
                mode="lines",
                name="ECG",
            ),
            row=5,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=df_avrage.index,
                y=df_avrage["Trig MRI - Custom, AMI / HLT - A 6"],
                mode="lines",
                name="Trig",
            ),
            row=6,
            col=1,
        )
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=df_avrage.index,
                y=df_avrage["GSR - EDA100C-MRI"],
                mode="lines",
                name="GSR",
            ),
        )
        fig.add_trace(
            go.Scattergl(x=HR_df.index, y=HR_df["HRV"], mode="lines", name="HRV"),
        )

        fig.add_trace(
            go.Scattergl(
                x=df_avrage.index,
                y=df_avrage["Pulse - PPG100C"],
                mode="lines",
                name="PPG",
            ),
        )
        fig.add_trace(
            go.Scattergl(x=HR_df.index, y=HR_df["HR"], mode="lines", name="HR"),
        )
        fig.add_trace(
            go.Scattergl(
                x=df_avrage.index,
                y=df_avrage["ECG - ECG100C"],
                mode="lines",
                name="ECG",
            ),
        )
        fig.add_trace(
            go.Scattergl(
                x=df_avrage.index,
                y=df_avrage["Trig MRI - Custom, AMI / HLT - A 6"],
                mode="lines",
                name="Trig",
            ),
        )
    for interval in intervals:
        fig.add_vrect(
            x0=interval[0],
            x1=interval[1],
            fillcolor="green",
            opacity=0.25,
            line_width=0,
        )
    fig.update_layout(
        autosize=True,
        title_text=title,
    )
    return fig


def interval_selector(df, threshold=0.01, fs=2000, averaging_factor=100_00):

    df_avrage = df.groupby(np.arange(len(df)) // averaging_factor).mean()
    df_avrage.index = np.linspace(0, len(df) / (fs * 60), len(df_avrage))
    data = df_avrage["Trig MRI - Custom, AMI / HLT - A 6"]
    # Find indices where data is larger than 0.1
    threshold = data.mean()
    indices = np.where(data > threshold)[0]

    # Find intervals of consecutive indices
    intervals = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)

    # Convert intervals to a list of tuples (start, end)

    intervals = [
        (df_avrage.index[interval[0]], df_avrage.index[interval[-1]])
        for interval in intervals
        if len(interval) > 0
    ]

    return intervals


def extract_session_suffix(input_string):
    match = re.search(r"session_(\w+)", input_string)
    if match:
        return match.group(1)
    return None


def extract_ids_from_dir(directory):
    ids = []
    for entry in os.listdir(directory):
        match = re.match(r"ptsd_(\d+)_complete", entry)
        if match:
            ids.append(match.group(1))
    return ids


def list_files_for_id(directory, id):
    files = []
    id_dir = f"ptsd_{id}_complete"
    biopac_dir = os.path.join(directory, id_dir, "Biopac_data")
    if os.path.exists(biopac_dir):
        for file in os.listdir(biopac_dir):
            files.append(os.path.join(biopac_dir, file))
    return files


def get_files_from_aws(id_aws, key_aws, bucket_name="assaf-harofeh"):
    import boto3

    s3 = boto3.client(
        "s3",
        region_name="us-east-1",  # US East (N. Virginia)
        aws_access_key_id=id_aws,
        aws_secret_access_key=key_aws,
    )
    response = s3.list_objects_v2(Bucket=bucket_name)
    files = []
    for content in response.get("Contents", []):
        files.append(content["Key"])
    return files


def extract_id_from_path(file_path):
    # Assuming the ID is the part after 'ptsd_' and before the next '_'
    parts = file_path.split("/")
    for part in parts:
        if part.startswith("ptsd_"):
            return part.split("_")[1]
    return None


def create_id_file_dict(file_list):
    id_file_dict = {}
    for file in file_list:
        id = extract_id_from_path(file)
        if id:
            if id not in id_file_dict:
                id_file_dict[id] = []
            id_file_dict[id].append(file)
    return id_file_dict


# Example usage
def get_data_from_aws(id_aws, key_aws, file_path, bucket_name="assaf-harofeh"):

    s3 = boto3.client(
        "s3",
        region_name="us-east-1",  # US East (N. Virginia)
        aws_access_key_id=id_aws,
        aws_secret_access_key=key_aws,
    )
    obj = s3.get_object(Bucket=bucket_name, Key=file_path)
    return sio.loadmat(io.BytesIO(obj["Body"].read()))


df_scores = pd.read_csv("./CAPS.csv")


def main():
    id_aws = st.secrets["ID_AWS"]
    key_aws = st.secrets["KEY_AWS"]
    files = get_files_from_aws(id_aws, key_aws)
    id_file_dict = create_id_file_dict(files)
    chosen_id = None
    with st.sidebar:
        chosen_id = st.selectbox("Select Participant", id_file_dict.keys())
        dir_file_name_path = {}
        for file in id_file_dict[chosen_id]:
            file_name = file.split("\\")[-1][:-4]
            file_name = extract_session_suffix(file_name)
            dir_file_name_path[file_name] = file
        file_name = st.selectbox("Select File", dir_file_name_path.keys())
        file_path = dir_file_name_path[file_name]
        is_subplots = st.checkbox("Subplots", value=False)
        Normalize = st.checkbox("Normalize", value=False)
    cap = df_scores[df_scores["ID"] == int(chosen_id)]
    # print(cap)
    ptsd_score_to_text = {0: "No PTSD", 1: "PTSD", 2: "Borderline"}
    title = f"CAPS Score: {cap['CAPS'].values[0]}, {ptsd_score_to_text[cap['PTSD'].values[0]]}"
    st.title(title)
    mat_contents = get_data_from_aws(id_aws, key_aws, file_path)
    df = pd.DataFrame(mat_contents["data"], columns=mat_contents["labels"])
    res = process_ppg_signal(df["Pulse - PPG100C"], window_size=60)
    HR_df = pd.DataFrame(
        {"HRV": res["time_diffs_sd"], "HR": res["time_diffs_avg"]},
        index=res["valley_times"][:-1] / 60,
    )
    averaging_factor = 100
    fs = 2000
    df_avrage = df.groupby(np.arange(len(df)) // averaging_factor).mean()
    df_avrage.set_index(
        np.linspace(0, df.shape[0] / (fs * 60), df_avrage.shape[0]), inplace=True
    )
    intervals = interval_selector(df)

    if Normalize:
        df_avrage = df_avrage.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        HR_df = HR_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    fig = plot_helper(
        HR_df,
        df_avrage,
        intervals,
        is_subplots=is_subplots,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
