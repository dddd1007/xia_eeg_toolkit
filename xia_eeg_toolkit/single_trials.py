import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.formula.api import rlm
from termcolor import colored
from tqdm import tqdm


def load_epochs(eeg_filename):
    print(f"读取脑电数据文件:\n {eeg_filename} ... \n")
    data = mne.read_epochs(eeg_filename)
    ch_names = data.ch_names
    erp_table = data.to_data_frame()
    del data
    return {
        "erp_table": erp_table,
        "ch_names": ch_names,
    }


def load_reject_log(reject_log_filename):
    print(f"读取剔除坏段文件:\n {reject_log_filename} ... \n")
    reject_log = np.load(reject_log_filename, allow_pickle=True)
    return reject_log["bad_epochs"]


def load_sub_beh_data(all_beh_data, sub_num, remove_miss_trials=False):
    sub_beh_data = all_beh_data[all_beh_data["sub_num"] == sub_num].copy()
    sub_beh_data = sub_beh_data.reset_index(drop=True)
    if remove_miss_trials:
        sub_beh_data = sub_beh_data[pd.notna(sub_beh_data["rt"])]
        sub_beh_data = sub_beh_data.reset_index(drop=True)
    return sub_beh_data


def single_trial_robust_regression(
    filtered_erp_table,
    ch_names,
    indep_value,
    other_dep_value=["stim_loc_num", "resp_num", "congruency_num"],
    dummy_col="run_num",
    debug=False,
):
    t_values = []
    p_values = []
    beta_values = []

    from statsmodels.formula.api import rlm
    from pandas import get_dummies
    import pandas as pd

    # 预处理列名
    regression_df = filtered_erp_table.copy()
    renamed_column_names = {
        name: name.replace(" ", "_") for name in regression_df.columns
    }
    regression_df.rename(columns=renamed_column_names, inplace=True)
    ch_names = [name.replace(" ", "_") for name in ch_names]

    # Dummy coding the 'dummy_col'
    dummy_cols = get_dummies(regression_df[dummy_col], prefix=dummy_col)
    regression_df = pd.concat([regression_df, dummy_cols], axis=1)

    # Define the independent variables
    indep_vars = (
        [str(indep_value)]
        + [str(val) for val in other_dep_value]
        + [str(col) for col in dummy_cols.columns.tolist()]
    )

    # Perform robust regression for each column in 'ch_names'
    for col in ch_names:
        if debug:
            print(f"Performing robust regression for {col}")
        # Define the model
        model = rlm(f"{col} ~ {' + '.join(indep_vars)} - 1", data=regression_df)
        if debug:
            print(f"Model defined: {model}")
        # Fit the model
        results = model.fit()
        if debug:
            print(results.summary())

        # Save the t-values, p-values and beta values
        t_values.append(results.tvalues[indep_value])
        p_values.append(results.pvalues[indep_value])
        beta_values.append(results.params[indep_value])

        # Explicitly delete the model and results objects
        del model
        del results

    return pd.DataFrame(
        {
            "ch_names": ch_names,
            "t_value": t_values,
            "p_value": p_values,
            "beta_value": beta_values,
        }
    )


def split_condition_column(
    single_time_point_table,
    target_col="condition",
    into_col=["type", "hand", "congruency", "prop", "volatility"],
):
    # Check if 'condition' column exists in the DataFrame
    if target_col in single_time_point_table.columns:
        # Split 'condition' column and store the result in new columns
        single_time_point_table[into_col] = single_time_point_table[
            target_col
        ].str.split("/", expand=True)
        # Drop the original 'condition' column
        single_time_point_table.drop(columns=[target_col], inplace=True)
    else:
        print("Column 'condition' does not exist in the DataFrame.")

    return single_time_point_table


def select_erp_by_annotation(
    single_time_point_table, filtered_by={"type": ["corResp", "wrongResp"]}
):
    for key, values in filtered_by.items():
        filtered_erp_table = single_time_point_table[
            single_time_point_table[key].isin(values)
        ]
    filtered_erp_table = filtered_erp_table.reset_index(drop=True)
    return filtered_erp_table


def validate_and_generate_regression_data(
    filtered_erp_table,
    sub_beh_data,
    reject_log,
    rt_col="rt",
    valid_col="congruency",
    other_dep_value=["stim_loc_num", "resp_num", "congruency_num"],
    indep_value="bl_sr_pe",
    dummy_col="run_num",
):
    erp_num = len(filtered_erp_table)

    # ====== Valid and combine beh and erp data =====
    matched_sub_beh_data = (
        sub_beh_data.dropna(subset=[rt_col])
        .tail(erp_num)
        .copy()
        .reset_index(drop=True)
    )
    if not filtered_erp_table[valid_col].equals(
        matched_sub_beh_data[valid_col]
    ):
        raise Exception(
            "The Beh data mismatch with ERP data, Please check your data"
        )

    extend_col = [indep_value] + other_dep_value + [dummy_col]
    filtered_erp_table[extend_col] = matched_sub_beh_data[extend_col]
    filtered_erp_table["reject"] = reject_log[
        filtered_erp_table["epoch"].values
    ].copy()

    return filtered_erp_table


def perform_regression_and_add_metadata(
    data,
    ch_names,
    indep_value,
    other_dep_value,
    dummy_col,
    time_point,
    sub_num,
    if_reject,
    debug=False,
):
    result = single_trial_robust_regression(
        data,
        ch_names,
        indep_value=indep_value,
        other_dep_value=other_dep_value,
        dummy_col=dummy_col,
        debug=debug,
    )
    result["time_point"] = time_point
    result["sub_num"] = sub_num
    result["if_reject"] = if_reject
    return result


# def generate_labels(beh_data_subset):
#     # Convert columns to the corresponding labels
#     beh_data_labels = beh_data_subset[
#         ["corr_resp_num", "congruency_num", "prop", "volatile"]
#     ].copy()
#     beh_data_labels["corr_resp_num"] = beh_data_labels["corr_resp_num"].replace(
#         {0: "leftHand", 1: "rightHand"}
#     )
#     beh_data_labels["congruency_num"] = beh_data_labels[
#         "congruency_num"
#     ].replace({0: "con", 1: "inc"})
#     return beh_data_labels.apply(
#         lambda row: f"{row['corr_resp_num']}/{row['congruency_num']}/{row['prop']}/{row['volatile']}",
#         axis=1,
#     ).tolist()


# def valid_data(epochs, reject_log, beh_data):
#     print("=== Verify the integrity of data ===")
#     if len(epochs) != len(reject_log):
#         raise Exception("The epoch data mismatch the reject_log!")
#     else:
#         print("|| The epoch data matches the reject_log!")

#     if len(epochs) != len(beh_data):
#         print(
#             "⚠️ The epoch data mismatch the behavior data! Trying to fix it..."
#         )

#         # Extract epoch labels
#         events, event_ids = epochs.events, epochs.event_id
#         epoch_labels = [
#             event_id
#             for event in events
#             for event_id, value in event_ids.items()
#             if value == event[-1]
#         ]
#         epoch_labels = [label.replace("stim/", "") for label in epoch_labels]

#         # Try different ways to fix the mismatch
#         for fix_method in ["head", "tail"]:
#             beh_data_subset = getattr(beh_data, fix_method)(len(epochs)).copy()
#             beh_data_labels_corrected = generate_labels(beh_data_subset)

#             if epoch_labels == beh_data_labels_corrected:
#                 print("✅ Data fixation is successful!")
#                 return epochs, beh_data_subset
#         # If the code reaches here, both methods have failed
#         sub_num = beh_data["sub_num"].unique()[0]
#         print(
#             f"The data of sub {colored(sub_num, 'red')} is mismatch between epoch labels and behavior data."
#         )
#         raise Exception("❌ Data fixation failed!")

#     print("|| The epoch data matches the behavior data!")
#     print("=== Verification of data is successful! === \n")
#     return epochs, beh_data


# def perform_regression(
#     ch_voltage_data, indep_var, analysis_var, reject_log, debug=False
# ):
#     n_windows = ch_voltage_data.shape[1]
#     beta_values = np.zeros(n_windows)
#     p_values = np.zeros(n_windows)
#     t_values = np.zeros(n_windows)
#     for win_idx in range(n_windows):
#         dep_var = ch_voltage_data[:, win_idx]
#         data = pd.concat(
#             [pd.Series(dep_var, name="dep_var"), indep_var], axis=1
#         )
#         data = data.replace([np.inf, -np.inf], np.nan).dropna()
#         data = data[~np.array(reject_log)]
#         formula = "dep_var ~ -1 + " + " + ".join(
#             [col for col in data.columns if col != "dep_var"]
#         )

#         if win_idx == 0 and debug:  # for debugging
#             print(f"Regression formula: {formula}")

#         model = rlm(formula, data=data).fit()
#         beta_values[win_idx] = model.params[analysis_var]
#         p_values[win_idx] = model.pvalues[analysis_var]
#         t_values[win_idx] = model.tvalues[analysis_var]
#     return beta_values, p_values, t_values


# def generate_indep_var(beh_data, analysis_var, debug=False):
#     indep_var = beh_data[
#         [analysis_var, "congruency_num", "resp_num", "run_num"]
#     ].copy()
#     indep_var = pd.get_dummies(indep_var, columns=["run_num"], drop_first=False)

#     # Remove rows with NaNs and infs
#     indep_var = indep_var.replace([np.inf, -np.inf], np.nan)
#     before_rows = len(indep_var)
#     indep_var = indep_var.dropna().reset_index(drop=True)
#     after_rows = len(indep_var)
#     n_removed_rows = before_rows - after_rows
#     print(
#         f"Removed {n_removed_rows} rows with NA ({100 * n_removed_rows / before_rows:.2f}% of total) \n"
#     )

#     if debug:
#         print("Independent variable dataframe: \n")
#         print(indep_var.head())
#     return indep_var


# def analyze_data(
#     epochs, indep_var, reject_log, analysis_var, do_parallel=True, debug=False
# ):
#     print("=== Begin analyzing data... ===")

#     if do_parallel:
#         # 创建一个 tqdm 对象，用于在循环中显示进度条
#         results = Parallel(n_jobs=-1, verbose=10)(
#             delayed(perform_regression)(
#                 ch_voltage_data=epochs.copy()
#                 .pick_channels([ch_name])
#                 .get_data()
#                 .squeeze(axis=1),
#                 indep_var=indep_var,
#                 analysis_var=analysis_var,
#                 reject_log=reject_log,
#                 debug=debug,
#             )
#             for ch_name in epochs.ch_names
#         )
#     else:
#         results = []
#         for ch_name in epochs.ch_names:
#             result = perform_regression(
#                 ch_voltage_data=epochs.copy()
#                 .pick_channels([ch_name])
#                 .get_data()
#                 .squeeze(axis=1),
#                 indep_var=indep_var,
#                 analysis_var=analysis_var,
#                 reject_log=reject_log,
#                 debug=debug,
#             )
#             results.append(result)

#     beta_values, p_values, t_values = zip(*results)

#     # 创建 DataFrame，并将 epochs.ch_names 用作行索引
#     beta_df = pd.DataFrame(np.array(beta_values), index=epochs.ch_names)
#     p_values_df = pd.DataFrame(np.array(p_values), index=epochs.ch_names)
#     t_values_df = pd.DataFrame(np.array(t_values), index=epochs.ch_names)

#     return beta_df, p_values_df, t_values_df


# def single_trial_analysis(
#     epochs,
#     beh_data,
#     reject_log,
#     analysis_var,
#     do_parallel=True,
#     debug=False,
# ):
#     epochs, valid_beh_data = valid_data(epochs, reject_log, beh_data)

#     indep_var = generate_indep_var(valid_beh_data, analysis_var)
#     beta_values_stim, p_values_stim, t_values_stim = analyze_data(
#         epochs,
#         indep_var,
#         reject_log,
#         analysis_var,
#         do_parallel=do_parallel,
#         debug=debug,
#     )

#     return (beta_values_stim, p_values_stim, t_values_stim)
