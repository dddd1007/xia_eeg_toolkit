import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.formula.api import rlm
from tqdm import tqdm


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
        print("The epoch data is mismatch the behavioral data, \n"
              f"The count of epochs is {len(epochs)} and the len of ben_data is {len(beh_data)} \n"
              "Now fix it...")
        beh_data = beh_data.tail(len(epochs))
    print("=== The Validation is Finished === \n")
    return epochs, beh_data


def perform_regression(ch_data, indep_var, analysis_var, reject_log, window_size, n_windows):
    beta_values = np.zeros(n_windows)
    p_values = np.zeros(n_windows)
    t_values = np.zeros(n_windows)
    for win_idx in range(n_windows):
        win_data = np.mean(ch_data[:, win_idx: win_idx + window_size], axis=1)
        data = pd.concat([pd.Series(win_data, name="win_data"), indep_var], axis=1)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        #data = data[~data.index.isin(reject_log)]
        rows_before_filtering = len(data)
        data = data[~np.array(reject_log)]
        rows_after_filtering = len(data)
        print(f"Removed {rows_before_filtering - rows_after_filtering} rows, "
              f"{100 * (rows_before_filtering - rows_after_filtering) / rows_before_filtering:.2f}% of total")
        formula = "win_data ~ " + " + ".join(
            [col for col in data.columns if col != "win_data"]
        )
        model = rlm(formula, data=data).fit()
        beta_values[win_idx] = model.params[analysis_var]
        p_values[win_idx] = model.pvalues[analysis_var]
        t_values[win_idx] = model.tvalues[analysis_var]
    return beta_values, p_values, t_values


def generate_indep_var(behavior_data, analysis_var):
    indep_var = behavior_data[
        [analysis_var, "congruency_num", "resp_num", "run_num"]
    ].copy()
    indep_var = pd.get_dummies(indep_var, columns=["run_num"], drop_first=False)

    # Remove rows with NaNs and infs
    indep_var = indep_var.replace([np.inf, -np.inf], np.nan)
    before_rows = len(indep_var)
    indep_var = indep_var.dropna()
    after_rows = len(indep_var)

    n_removed_rows = before_rows - after_rows
    print(
        f"Removed {n_removed_rows} rows ({100 * n_removed_rows / before_rows:.2f}% of total)"
    )
    return indep_var


def analyze_data(
        epochs, indep_var, reject_log, analysis_var, window_size, n_windows
):
    n_channels = len(epochs.ch_names)
    # 创建一个tqdm对象，用于在循环中显示进度条
    with tqdm(total=n_channels, desc="Processing Channels", position=0) as pbar:
        def update(*a):
            pbar.update()

        # 使用Parallel和delayed，并在每次调用perform_regression后更新进度条
        results = Parallel(n_jobs=-1)(
            delayed(perform_regression)(
                epochs.get_data()[:, ch_idx, :],
                indep_var,
                analysis_var,
                reject_log,
                window_size,
                n_windows,
            )(callback=update)  # 将回调添加到每个延迟的调用中
            for ch_idx in range(n_channels)
        )

    beta_values, p_values, t_values = zip(*results)
    return np.array(beta_values), np.array(p_values), np.array(t_values)


def single_trial_analysis(
    eeg_filename,
    beh_data,
    reject_log_filename,
    analysis_var,
    window_size_ms,
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
        reject_log,
        analysis_var,
        window_size,
        n_windows,
    )

    return (
        beta_values_stim,
        p_values_stim,
        t_values_stim
    )
