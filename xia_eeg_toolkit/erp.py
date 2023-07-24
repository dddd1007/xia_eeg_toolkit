
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import t, sem
import tqdm
import mne

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
    diff_evokes = []
    for i in range(len(evokes[list(evokes.keys())[0]])):
        diff_evokes.append(mne.combine_evoked(
            [evokes[list(evokes.keys())[0]][i], evokes[list(evokes.keys())[1]][i]], 
            weights=[1, -1]))
    return diff_evokes


def generate_mean_evokes(evokes):
    mean_evokes = []
    for i in range(len(evokes[list(evokes.keys())[0]])):
        mean_evokes.append(mne.combine_evoked(
            [evokes[list(evokes.keys())[0]][i], evokes[list(evokes.keys())[1]][i]], 
            weights='nave'))
    return mean_evokes


def compare_evoke_wave(evokes, chan_name, vlines="auto", show=False, axes=None):
    cond1 = list(evokes.keys())[0]
    cond2 = list(evokes.keys())[1]
    roi = [chan_name]
    # Get evokes
    # Define parameters
    color_dict = {cond1: 'blue', cond2: 'red'}
    linestyle_dict = {cond1: 'solid', cond2: 'dashed'}
    evoke_plot = mne.viz.plot_compare_evokeds(evokes,
                                              vlines=vlines,
                                              legend='lower right',
                                              picks=roi, show_sensors='upper right',
                                              show=show,
                                              ci=False,
                                              axes=axes,
                                              colors=color_dict,
                                              linestyles=linestyle_dict,
                                              title=chan_name + " :  " + cond1 +
                                              ' vs. ' + cond2)
    return evoke_plot


def show_difference_wave(evokes_diff, chan_name, axes=None):
    roi = [chan_name]
    difference_wave_plot = mne.viz.plot_compare_evokeds({'diff_evoke': evokes_diff},
                                                        picks=roi,
                                                        axes=axes,
                                                        show_sensors='upper right',
                                                        combine='mean',
                                                        title=chan_name + "Difference Wave")  # noqa: E501
    return difference_wave_plot


def calc_erp_ttest(data_cache: dict, condition_list: list, time_window: list,
                   direction: str, ch_name='eeg'):
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
        evoke_data = evoke_data.get_data(tmin=time_window[0], tmax=time_window[1])
        evoke_mean = evoke_data.mean() if ch_name != "eeg" else evoke_data.mean(axis=1)
        cond1_array[:, i] = evoke_mean

    # condition 2
    condition = condition_list[1]
    cond2_array = np.zeros((ch_nums, len(data_cache)))
    for i, processed_data in enumerate(data_cache.values()):
        evoke_data = processed_data[condition].average()
        if ch_name != "eeg":
            evoke_data = evoke_data.pick_channels([ch_name])
        evoke_data = evoke_data.get_data(tmin=time_window[0], tmax=time_window[1])
        evoke_mean = evoke_data.mean() if ch_name != "eeg" else evoke_data.mean(axis=1)
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

        tt_results.append([tt_result.statistic, tt_result.pvalue,
                           ci_95[0], ci_95[1], tt_result.pvalue < 0.05])

    if ch_name == "eeg":
        # Just take the channel names from the last processed data
        chan_name = list(processed_data[condition].info['ch_names'][:64])
    else:
        chan_name = [ch_name]

    df = pd.DataFrame(tt_results, columns=['T-Statistic', 'P-Value',
                                           '95% CI Lower', '95% CI Upper',
                                           'Significant'])
    df.index = chan_name

    return df
