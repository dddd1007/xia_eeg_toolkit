import mne
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from statsmodels.formula.api import rlm
from statsmodels.stats.multitest import multipletests


def load_epochs(eeg_filename):
    print(f"读取脑电数据文件:\n {eeg_filename} ... \n")
    epochs = mne.read_epochs(eeg_filename)
    print(epochs.info)
    return epochs


def load_reject_log(reject_log_filename):
    print(f"读取剔除坏段文件:\n {reject_log_filename} ... \n")
    reject_log = np.load(reject_log_filename, allow_pickle=True)
    return reject_log["bad_epochs"]

def preprocess_data(epochs, bad_epochs, beh_data):
    # 检验 epoch 数量和剔除坏段数量一致性
    print("=== Verify the integrity of data === \n")
    if len(epochs) == len(bad_epochs):
        print("The epochs and reject log is equal! \n")
    else:
        raise Exception("The epoch data mismatch the reject_log!")

    # 检验 epoch 数量和行为数据的数量一致性
    if len(epochs) == len(beh_data):
        print("The epochs and behavioral data is equal! \n")
    else:
        print("The epoch data is mismatch the behavioral data, try to fix it...")
        beh_data = beh_data.tail(len(epochs))
    print("=== The Validation is Finished === \n")
    return epochs, beh_data


def perform_regression(ch_data, X, window_size, n_windows, analysis_variable):
    beta_values = np.zeros(n_windows)
    p_values = np.zeros(n_windows)
    t_values = np.zeros(n_windows)
    for win_idx in range(n_windows):
        win_data = np.mean(ch_data[:, win_idx : win_idx + window_size], axis=1)
        data = pd.concat([pd.Series(win_data, name="win_data"), X], axis=1)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        formula = "win_data ~ " + " + ".join(
            [col for col in data.columns if col != "win_data"]
        )
        model = rlm(formula, data=data).fit()
        beta_values[win_idx] = model.params[analysis_variable]
        p_values[win_idx] = model.pvalues[analysis_variable]
        t_values[win_idx] = model.tvalues[analysis_variable]
    return beta_values, p_values, t_values


def analyze_data(
        epochs, behavior_data, window_size, n_windows, analysis_variables
):
    X = behavior_data[
        [analysis_variables, "congruency_num", "resp_num", "run_num"]
    ].copy()
    X = pd.get_dummies(X, columns=["run_num"], drop_first=False)

    # Remove rows with NaNs and infs
    X = X.replace([np.inf, -np.inf], np.nan)
    before_rows = len(X)
    X = X.dropna()
    after_rows = len(X)

    n_removed_rows = before_rows - after_rows
    print(
        f"Removed {n_removed_rows} rows ({100 * n_removed_rows / before_rows:.2f}% of total)"
    )

    # Remove corresponding epochs
    good_indices = np.where(~np.isnan(X).any(axis=1))[0]
    epochs = epochs[good_indices]

    n_channels = len(epochs.ch_names)

    # 检查model是否正确
    print("=== The head of data is: ====\n ")
    print(X.head())
    print("\n")
    print("=== The model summary is: ===\n ")
    foo = sm.OLS(epochs.get_data()[:, 0, 1], X).fit()
    print(foo.summary())
    print("=============================\n")
    results = Parallel(n_jobs=-1)(
        delayed(perform_regression)(
            epochs.get_data()[:, ch_idx, :],
            X,
            window_size,
            n_windows,
            analysis_variables,
        )
        for ch_idx in range(n_channels)
    )
    beta_values, p_values, t_values = zip(*results)
    return np.array(beta_values), np.array(p_values), np.array(t_values)


def correct_p_values(p_values):
    _, p_values_fdr, _, _ = multipletests(p_values.flatten(), method="fdr_bh")
    return p_values_fdr.reshape(p_values.shape)

def single_trial_analysis(
    eeg_filename,
    beh_data,
    reject_log_filename,
    window_size_ms,
    analysis_variables,
):
    epochs = load_epochs(eeg_filename)
    reject_log = load_reject_log(reject_log_filename)

    epochs, fixed_beh_data = preprocess_data(
        epochs, reject_log, beh_data
    )
    window_size = int(window_size_ms / 1000 * epochs.info["sfreq"])
    n_windows = len(epochs.times) - window_size
    beta_values_stim, p_values_stim, t_values_stim = analyze_data(
        epochs,
        fixed_beh_data,
        window_size,
        n_windows,
        analysis_variables,
    )
    p_values_stim_corrected = correct_p_values(p_values_stim)

    return (
        beta_values_stim,
        p_values_stim,
        p_values_stim_corrected,
        t_values_stim
    )

