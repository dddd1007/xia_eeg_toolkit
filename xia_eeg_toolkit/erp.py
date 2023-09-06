import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import t, sem
from tqdm import tqdm
import mne

mne.set_log_level("WARNING")


# preprocess data
def load_data(processed_data_list: list):
    """
    This function loads a list of file paths and reads each file into an mne.Epochs
    object.

    Args:
        processed_data_list: A list containing the paths of the files to be loaded.

    Returns:
        A dictionary where the keys are the file paths and the values are the
        corresponding mne.Epochs objects.
    """
    import mne

    data_cache = {}
    for d in tqdm(processed_data_list):
        data_cache[d] = mne.read_epochs(d)
    return data_cache


def generate_evokes(data_cache: dict, condition_list: list):
    """
    This function generates ERPs from the loaded data according to the given condition
    list.

    Args:
        data_cache: A dictionary where the keys are the file paths and the values are
        the corresponding mne.Epochs objects.
        condition_list: A list containing the conditions to be processed.

    Returns:
        If the condition list has only one element, it returns a list where each
        element is the ERP corresponding to the given condition.
        If the condition list has more than one element, it returns a dictionary where
        the keys are the conditions and the values are a list where each element is the
        ERP corresponding to that condition.
    """
    evokes = {}
    if len(condition_list) == 1:
        cond = condition_list[0]
        evokes = []
        for d, epochs in data_cache.items():
            evokes.append(epochs[cond].average())
    else:
        for cond in condition_list:
            evokes[cond] = []
            for d, epochs in data_cache.items():
                evokes[cond].append(epochs[cond].average())
    return evokes


def generate_diff_evokes(evokes):
    """
    This function generates the difference between the evoked responses of two
    conditions.

    Args:
        evokes: A dictionary where the keys are the conditions and the values are the
        evoked data.

    Returns:
        A list of mne.Evoked objects, each representing the difference between the
        evoked responses of the two conditions.
    """
    diff_evokes = []
    for i in range(len(evokes[list(evokes.keys())[0]])):
        diff_evokes.append(
            mne.combine_evoked(
                [
                    evokes[list(evokes.keys())[0]][i],
                    evokes[list(evokes.keys())[1]][i],
                ],
                weights=[1, -1],
            )
        )
    return diff_evokes


def generate_mean_evokes(evokes):
    """
    This function generates the mean of the evoked responses of two conditions.

    Args:
        evokes: A dictionary where the keys are the conditions and the values are the
        evoked data.

    Returns:
        A list of mne.Evoked objects, each representing the mean of the evoked responses
        of the two conditions.
    """
    mean_evokes = []
    for i in range(len(evokes[list(evokes.keys())[0]])):
        mean_evokes.append(
            mne.combine_evoked(
                [
                    evokes[list(evokes.keys())[0]][i],
                    evokes[list(evokes.keys())[1]][i],
                ],
                weights="nave",
            )
        )
    return mean_evokes


def compare_evoke_wave(evokes, chan_name, vlines="auto", show=False, axes=None):
    """
    This function compares the evoked waveforms of two conditions.

    Args:
        evokes: A dictionary where the keys are the conditions and the values are the
                evoked data.
        chan_name: The name of the channel to be plotted.
        vlines: The vertical lines to be plotted. Default is "auto".
        show: Whether to show the plot. Default is False.
        axes: The axes on which to plot. Default is None.

    Returns:
        The plot of the compared evoked waveforms.
    """
    cond1 = list(evokes.keys())[0]
    cond2 = list(evokes.keys())[1]
    roi = [chan_name]
    # Get evokes
    # Define parameters
    color_dict = {cond1: "blue", cond2: "red"}
    linestyle_dict = {cond1: "solid", cond2: "dashed"}
    evoke_plot = mne.viz.plot_compare_evokeds(
        evokes,
        vlines=vlines,
        legend="lower right",
        picks=roi,
        show_sensors="upper right",
        show=show,
        ci=False,
        axes=axes,
        colors=color_dict,
        linestyles=linestyle_dict,
        title=chan_name + " :  " + cond1 + " vs. " + cond2,
    )
    return evoke_plot


def show_difference_wave(evokes_diff, chan_name, axes=None):
    """
    This function shows the difference wave of the evoked data.

    Args:
        evokes_diff: The difference of the evoked data.
        chan_name: The name of the channel to be plotted.
        axes: The axes on which to plot. Default is None.

    Returns:
        The plot of the difference wave.
    """
    roi = [chan_name]
    difference_wave_plot = mne.viz.plot_compare_evokeds(
        {"diff_evoke": evokes_diff},
        picks=roi,
        axes=axes,
        show_sensors="upper right",
        combine="mean",
        title=chan_name + "Difference Wave",
    )  # noqa: E501
    return difference_wave_plot


def plot_multi_erp(
    data_cache, condition_lists, ch_names, plot_save_path=None, figsize=(30, 10)
):
    """
    This function plots the ERP of multiple conditions.

    Args:
        data_cache: A dictionary where the keys are the file paths and the values are
                    the corresponding mne.Epochs objects.
        condition_lists: A list containing the conditions to be processed.
        ch_names: The names of the channels to be plotted.
        plot_save_path: The path to save the image. If None, the image will not
                        be saved. Default is None.
        figsize: The size of the image, in the format (width, height).
                 Default is (30, 10).

    Returns:
        None. This function will directly display the image, and if plot_save_path is
        provided, it will also save the image.
    """
    f, axarr = plt.subplots(
        len(condition_lists), len(ch_names), figsize=figsize
    )

    for i, sub_cond in enumerate(condition_lists):
        for j, ch_name in enumerate(ch_names):
            foo = generate_evokes(data_cache, sub_cond)
            compare_evoke_wave(foo, ch_name, axes=axarr[i, j])

    # plt.subplots_adjust(wspace=0.5, hspace=0.5)

    if plot_save_path is not None:
        f.savefig(plot_save_path)


def plot_multi_topo(
    data_cache,
    condition_lists,
    plot_save_path=None,
    tmin=0.29,
    tmax=0.35,
    figsize=(30, 10),
):
    """
    This function is used to plot the topographic map of EEG under multiple conditions.

    Parameters:
        data_cache (dict): A dictionary containing EEG data, where the key is the file
                           path and the value is the corresponding mne.Epochs object.
        condition_lists (list): A list containing the conditions to be processed.
        plot_save_path (str, optional): The path to save the image. If None, the image
                                        will not be saved. Default is None.
        tmin (float, optional): The minimum time used for data cropping.
        tmax (float, optional): The maximum time used for data cropping.
        figsize (tuple, optional): The size of the image, in the format (width, height).

    Returns:
        None. This function will directly display the image,
        and if plot_save_path is provided, it will also save the image.
    """
    f, axarr = plt.subplots(1, len(condition_lists), figsize=figsize)

    for i, sub_cond in enumerate(condition_lists):
        evokes = generate_evokes(data_cache, [sub_cond])
        evoked = mne.combine_evoked(evokes, weights="nave")
        evoked.crop(tmin=tmin, tmax=tmax)  # Crop the data
        avg_data = evoked.data.mean(axis=1)  # Calculate the average

        # Create a new Evoked object with avg_data
        evoked_avg = mne.EvokedArray(avg_data[:, np.newaxis], evoked.info)
        evoked_avg.plot_topomap(ch_type="eeg", axes=axarr[i], colorbar=False)
        axarr[i].set_title(
            sub_cond, fontsize=20
        )  # Set title with larger font size
    if plot_save_path is not None:
        f.savefig(plot_save_path)
    plt.show()


def calc_erp_ttest(
    data_cache: dict,
    condition_list: list,
    time_window: list,
    direction: str,
    ch_name="eeg",
):
    """
    This function calculates the t-test for ERP data.

    Args:
        data_cache: A dictionary where the keys are the file paths and the values are
                    the corresponding mne.Epochs objects.
        condition_list: A list containing the conditions to be processed.
        time_window: A list containing the start and end times for the ERP analysis.
        direction: The alternative hypothesis for the t-test. It can be 'two-sided',
                   'less' or 'greater'.
        ch_name: The name of the channel to be analyzed. Default is 'eeg'.

    Returns:
        A DataFrame containing the t-statistic, p-value, 95% confidence interval, and
        a boolean indicating whether the result is significant for each channel.
    """
    if ch_name == "eeg":
        ch_nums = 64
    else:
        ch_nums = 1

    # condition 1
    condition = condition_list[0]
    cond1_array = np.zeros((ch_nums, len(data_cache)))
    for i, processed_data in enumerate(data_cache.values()):
        evoke_data = processed_data[condition].average()
        if ch_name != "eeg":
            evoke_data = evoke_data.pick_channels([ch_name])
        evoke_data = evoke_data.get_data(
            tmin=time_window[0], tmax=time_window[1]
        )
        evoke_mean = (
            evoke_data.mean() if ch_name != "eeg" else evoke_data.mean(axis=1)
        )
        cond1_array[:, i] = evoke_mean

    # condition 2
    condition = condition_list[1]
    cond2_array = np.zeros((ch_nums, len(data_cache)))
    for i, processed_data in enumerate(data_cache.values()):
        evoke_data = processed_data[condition].average()
        if ch_name != "eeg":
            evoke_data = evoke_data.pick_channels([ch_name])
        evoke_data = evoke_data.get_data(
            tmin=time_window[0], tmax=time_window[1]
        )
        evoke_mean = (
            evoke_data.mean() if ch_name != "eeg" else evoke_data.mean(axis=1)
        )
        cond2_array[:, i] = evoke_mean

    # ttest
    tt_results = []
    for chan_num in range(cond1_array.shape[0]):
        cond1_data = cond1_array[chan_num, :]
        cond2_data = cond2_array[chan_num, :]
        tt_result = ttest_rel(cond1_data, cond2_data, alternative=direction)

        # 计算置信区间
        se_diff = sem(cond1_data - cond2_data)
        df = len(cond1_data) - 1
        ci_95 = se_diff * np.array(t.interval(0.95, df))

        tt_results.append(
            [
                tt_result.statistic,
                tt_result.pvalue,
                ci_95[0],
                ci_95[1],
                tt_result.pvalue < 0.05,
            ]
        )

    if ch_name == "eeg":
        # Just take the channel names from the last processed data
        chan_name = list(processed_data[condition].info["ch_names"][:64])
    else:
        chan_name = [ch_name]

    df = pd.DataFrame(
        tt_results,
        columns=[
            "T-Statistic",
            "P-Value",
            "95% CI Lower",
            "95% CI Upper",
            "Significant",
        ],
    )
    df.index = pd.Index(chan_name)

    return df
