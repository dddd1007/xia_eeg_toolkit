import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from tqdm import tqdm

mne.set_log_level('WARNING')

def perform_time_frequency_analysis(
    input_data, conditions, frequency_band, channel, time_window
):
    """
    Perform time-frequency analysis for the given conditions, frequency band, and channel.

    Parameters
    ----------
    input_data : str
        Path to the .fif file that contains the preprocessed EEG data.
    conditions : list of str
        The experimental conditions for which to perform the analysis.
    frequency_band : tuple of float
        The frequency band (in Hz) for the analysis (e.g., (4, 7) for theta band).
    channel : str
        The EEG channel for which to perform the analysis.
    time_window : tuple of float
        The time window (in seconds) for the analysis (e.g., (-0.3, 1.2)).

    Returns
    -------
    dict
        A dictionary where the keys are the conditions and the values are the AverageTFR objects
        that contain the results of the time-frequency analysis for the given condition.
    """
    # Load the epochs
    epochs = input_data

    # Select the epochs for the given conditions
    epochs_cond = {cond: epochs[cond] for cond in conditions}

    # Select the data for the given channel
    epochs_channel = {
        cond: ep.copy().pick([channel]) for cond, ep in epochs_cond.items()
    }

    # Define the frequencies and number of cycles for the Morlet wavelets
    freqs = np.arange(frequency_band[0], frequency_band[1] + 1.0, 1.0)
    n_cycles = freqs / 2.0  # different number of cycle per frequency

    # Perform time-frequency analysis (using Morlet wavelets) for each condition
    power = {
        cond: mne.time_frequency.tfr_morlet(
            ep, freqs=freqs, n_cycles=n_cycles, return_itc=False, decim=3, n_jobs=1
        )
        for cond, ep in epochs_channel.items()
    }

    # Crop the time-frequency representations to the given time window
    power = {
        cond: pow.crop(tmin=time_window[0], tmax=time_window[1])
        for cond, pow in power.items()
    }

    # Apply baseline correction using the time points before 0
    power = {
        cond: pow.apply_baseline(baseline=(None, 0), mode="logratio")
        for cond, pow in power.items()
    }

    return power


def average_time_frequency_analysis(
    epochs_dict, conditions, frequency_band, channel, time_window
):
    """
    Perform time-frequency analysis for the given conditions, frequency band, and channel for each subject,
    and compute the average result for each condition across all subjects.

    Parameters
    ----------
    epochs_dict : dict
        A dictionary where the keys are the subject IDs and the values are the Epochs objects that
        contain the preprocessed EEG data for the corresponding subject.
    conditions : list of str
        The experimental conditions for which to perform the analysis.
    frequency_band : tuple of float
        The frequency band (in Hz) for the analysis (e.g., (4, 7) for theta band).
    channel : str
        The EEG channel for which to perform the analysis.
    time_window : tuple of float
        The time window (in seconds) for the analysis (e.g., (-0.3, 1.2)).

    Returns
    -------
    dict
        A dictionary where the keys are the conditions and the values are the AverageTFR objects
        that contain the average results of the time-frequency analysis for the given condition
        across all subjects.
    """
    # Define the frequencies for the Morlet wavelets
    freqs = np.arange(frequency_band[0], frequency_band[1] + 1.0, 1.0)

    # Prepare a dictionary to store the power for each condition and subject
    power_all_subjects = {cond: [] for cond in conditions}

    # Perform the time-frequency analysis for each subject separately
    for subject, epochs in epochs_dict.items():
        # Perform the time-frequency analysis for this subject
        power = perform_time_frequency_analysis(
            epochs, conditions, frequency_band, channel, time_window
        )

        # Store the power in the dictionary
        for cond in conditions:
            power_all_subjects[cond].append(power[cond])

    # Compute the average power data for each condition across all subjects
    average_power_data = {
        cond: np.mean([pow.data for pow in power_list], axis=0)
        for cond, power_list in power_all_subjects.items()
    }

    # Create AverageTFR objects for the average power data
    average_power = {}
    for cond, data in average_power_data.items():
        # Use the info and times from the first subject's power
        info = power_all_subjects[cond][0].info
        times = power_all_subjects[cond][0].times
        average_power[cond] = mne.time_frequency.AverageTFR(
            info, data, times, freqs, len(epochs_dict)
        )

    return average_power


def plot_avg_power(avg_power, theta_band=(4.0, 8.0), time_window=(-0.1, 1.0)):
    # Compute the average power in the theta band for each condition
    avg_power_theta = {}
    for cond, pow in avg_power.items():
        freq_mask = (pow.freqs >= theta_band[0]) & (
            pow.freqs <= theta_band[1]
        )  # create a mask for the theta band
        time_mask = (pow.times >= time_window[0]) & (
            pow.times <= time_window[1]
        )  # create a mask for the time window
        avg_power_theta[cond] = pow.data[:, freq_mask, :][:, :, time_mask].mean(
            axis=1
        )  # compute the average power in the theta band

    # Create a figure with two subplots (one for MC and one for MI)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    # Define the conditions for MC and MI
    conds_MC = ["stim/con/MC", "stim/inc/MC"]
    conds_MI = ["stim/con/MI", "stim/inc/MI"]

    # Define the colors for each condition
    colors = {
        "stim/con/MC": "darkorange",
        "stim/con/MI": "orange",
        "stim/inc/MC": "darkblue",
        "stim/inc/MI": "blue",
    }

    # Plot the average power in the theta band for each condition
    for ax, conds in zip(axs, [conds_MC, conds_MI]):
        for cond in conds:
            power_data = avg_power_theta[cond]
            times = avg_power[cond].times  # get the time points
            times_masked = times[time_mask]  # apply the time window mask
            ax.plot(times_masked, power_data[0, :], label=cond, color=colors[cond])
            ax.axhline(0, color="gray", linestyle="--")  # add a horizontal line at y=0
            ax.set_xlabel("Time (s)")
            ax.legend()

    # Add a label for the y-axis
    axs[0].set_ylabel("Power (dB)")

    # Show the figure
    plt.show()


def calculate_time_frequency_for_all(
    data_cache, conditions, frequency_band, channel, time_window
):
    """
    Perform time-frequency analysis for the given conditions, frequency band, and channel for each subject.

    Parameters
    ----------
    epochs_dict : dict
        A dictionary where the keys are the subject IDs and the values are the Epochs objects that
        contain the preprocessed EEG data for the corresponding subject.
    conditions : list of str
        The experimental conditions for which to perform the analysis.
    frequency_band : tuple of float
        The frequency band (in Hz) for the analysis (e.g., (4, 7) for theta band).
    channel : str
        The EEG channel for which to perform the analysis.
    time_window : tuple of float
        The time window (in seconds) for the analysis (e.g., (-0.3, 1.2)).

    Returns
    -------
    dict
        A dictionary where the keys are the subject IDs and the values are dictionaries that contain
        the AverageTFR objects for the time-frequency analysis results of each condition.
    """
    # Perform the time-frequency analysis for each subject
    power_all_subjects = {}
    for subject in tqdm(data_cache.keys()):
        print(f"Processing subject: {subject}\r", end="")
        power_all_subjects[subject] = perform_time_frequency_analysis(
            data_cache[subject], conditions, frequency_band, channel, time_window
        )
    return power_all_subjects

def mask_anova_data(single_tfr, time_window):
    times = single_tfr.times
    time_mask = (times >= time_window[0]) & (times <= time_window[1])
    masked_data = single_tfr.data[:, :, time_mask]
    return masked_data

def calculate_single_subject_power(
    power_all_subjects, sub_i, conditions, anova_time_window
):
    sub_dict = {}
    sub_dict["sub"] = sub_i
    for cond_i in conditions:
        single_tfr = power_all_subjects[sub_i][cond_i]
        single_sub_power = mask_anova_data(single_tfr, anova_time_window)
        sub_dict[cond_i] = np.mean(single_sub_power)
    return sub_dict


def perform_anova(long_format_df):
    model = AnovaRM(
        data=long_format_df, depvar="power", subject="sub", within=["congruent", "prop"]
    )
    anova_table = model.fit()
    return anova_table


def perform_post_hoc_test(long_format_df, main_effect):
    print(f"Performing post-hoc test for '{main_effect}'...")
    levels = long_format_df[main_effect].unique()
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            data1 = long_format_df.loc[
                long_format_df[main_effect] == levels[i], "power"
            ]
            data2 = long_format_df.loc[
                long_format_df[main_effect] == levels[j], "power"
            ]
            t_stat, p_val = ttest_rel(data1, data2)
            p_val *= len(levels)  # Bonferroni correction
            print(
                f"t-test between {levels[i]} and {levels[j]}: t = {t_stat}, p = {p_val}"
            )
            if t_stat > 0:
                print(f"{levels[i]} > {levels[j]}")
            elif t_stat < 0:
                print(f"{levels[j]} > {levels[i]}")


def perform_simple_effect_analysis(long_format_df, factor, level):
    print(
        f"Simple effect analysis for '{factor}' at '{level}' = {long_format_df[level].unique()[0]}"
    )
    subset = long_format_df[long_format_df[level] == long_format_df[level].unique()[0]]
    model = AnovaRM(data=subset, depvar="power", subject="sub", within=[factor])
    res = model.fit()
    print(res)


def anova_power_over_all(power_all_subjects, conditions, anova_time_window):
    power_sub_list = list(power_all_subjects.keys())
    avg_power_data = [
        calculate_single_subject_power(
            power_all_subjects, sub_key, conditions, anova_time_window
        )
        for sub_key in tqdm(power_sub_list)  # Add progress bar here
    ]

    # Create the DataFrame
    avg_power_table = pd.DataFrame(avg_power_data)

    # Melt the DataFrame to long format
    long_format_df = pd.melt(
        avg_power_table, id_vars="sub", var_name="condition", value_name="power"
    )

    # Extract 'congruent' and 'prop' from 'condition'
    long_format_df["congruent"] = long_format_df["condition"].str.split("/").str[1]
    long_format_df["prop"] = long_format_df["condition"].str.split("/").str[2]

    # Perform ANOVA
    anova_table = perform_anova(long_format_df)

    print(anova_table.anova_table)

    # If the p-value for the main effect of 'congruent' is less than 0.05
    if anova_table.anova_table.loc["congruent", "Pr > F"] < 0.05:
        perform_post_hoc_test(long_format_df, "congruent")

    # If the p-value for the main effect of 'prop' is less than 0.05
    if anova_table.anova_table.loc["prop", "Pr > F"] < 0.05:
        perform_post_hoc_test(long_format_df, "prop")

    # If the p-value for the interaction effect is less than 0.05
    if anova_table.anova_table.loc["congruent:prop", "Pr > F"] < 0.05:
        print("Performing post-hoc test for the interaction effect...")
        perform_simple_effect_analysis(long_format_df, "prop", "congruent")
        perform_simple_effect_analysis(long_format_df, "congruent", "prop")

    return anova_table
