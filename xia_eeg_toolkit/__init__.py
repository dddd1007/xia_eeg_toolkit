import os
from autoreject import AutoReject
import mne
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import t, sem
import multiprocessing

#
# Preprocess EEG data
#

def preprocess_epoch_data(raw_data_path, montage_file_path,
                          event_file_path, savefile_path,
                          input_event_dict: dict, rm_chans_list: list,
                          eog_chans: list, sub_num: int,
                          tmin=-0.3, tmax=1.2, l_freq=1, h_freq=30,
                          ica_z_thresh=1.96, ref_channels='average',
                          export_to_mne=True, export_to_eeglab=True,
                          do_autoreject=True):

    # Message to begin
    text = "Beginning preprocessing subject " + str(sub_num) + " ..."
    text = "* " + text + " *"
    border = "*" * len(text)

    print("\033[91m")  # 开始红色文本
    print(border)
    print(text)
    print(border)
    print("\033[0m")  # 结束红色文本

    # Import data
    raw = mne.io.read_raw_curry(raw_data_path, preload=True)

    # Set the channel type
    if rm_chans_list:
        raw.info['bads'].extend(rm_chans_list)
        raw.set_channel_types(dict(zip(eog_chans, ['eog']*len(eog_chans))))
    else:
        print("No channels to remove")

    # Set the montage
    montage_file = mne.channels.read_custom_montage(montage_file_path)
    raw.set_montage(montage_file)

    # Set the preprocessing setting
    filter_data = raw.copy()

    # Remove channels which are not needed
    # filter_data.drop_channels(rm_chans_list)

    # Filter the data
    filter_data.filter(l_freq, h_freq, n_jobs=multiprocessing.cpu_count())

    # prepare for ICA
    # reject_para = get_rejection_threshold(epochs_forica)

    # compute ICA
    ica = mne.preprocessing.ICA(n_components=.999, method='picard')
    ica.fit(filter_data)

    # Exclude blink artifact components
    eog_indices, eog_scores = ica.find_bads_eog(
        raw, ch_name=['FP1', 'FP2', 'F8'], threshold=ica_z_thresh)
    muscle_indices, muscle_scores = ica.find_bads_muscle(
        raw, threshold=ica_z_thresh)
    ica.exclude = muscle_indices + eog_indices
    ica_file_path = os.path.join(savefile_path, 'mne_fif', 'ICA', 'sub'+str(sub_num) + 
                                 '-ica.fif')
    os.makedirs(os.path.dirname(ica_file_path), exist_ok=True)
    ica.save(ica_file_path, overwrite=True)
    ica_data = ica.apply(filter_data)

    # Rereference
    ica_data.set_eeg_reference(ref_channels)

    # Import Event
    new_event = mne.read_events(event_file_path)
    annot_from_events = mne.annotations_from_events(events=new_event,
                                                    event_desc=input_event_dict,
                                                    sfreq=raw.info['sfreq'],
                                                    orig_time=raw.info['meas_date'])
    ica_data.set_annotations(annot_from_events)

    # Check the event in data
    events, event_dict = mne.events_from_annotations(ica_data)
    ica_data.del_proj()
    epochs = mne.Epochs(ica_data, events, event_dict, tmin, tmax,
                        baseline=(None, 0),  # reject=reject_para,
                        verbose=False, detrend=0, preload=True)

    # Save epoch data without autoreject
    if export_to_mne:
        epochs_filename = os.path.join(
            savefile_path, "mne_fif", "before_reject", "sub" + str(sub_num).zfill(2) + 
            '-before-reject-epo.fif')
        os.makedirs(os.path.dirname(epochs_filename), exist_ok=True)
        epochs.save(epochs_filename, overwrite=True)
        print("Saving epochs to %s" % epochs_filename)
    
    # Save epoch data with autoreject
    if do_autoreject:
        # Autoreject
        ar = AutoReject(n_jobs=multiprocessing.cpu_count())
        ar.fit(epochs)  # fit on a few epochs to save time
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)

        # Save rejected data
        if export_to_mne:
            epochs_ar_filename = os.path.join(
                savefile_path, "mne_fif", "rejected", "sub" + str(sub_num).zfill(2) +
                '-epo.fif')
            os.makedirs(os.path.dirname(epochs_ar_filename), exist_ok=True)
            epochs_ar.save(epochs_ar_filename, overwrite=True)
            print("Saving epochs to %s" % epochs_ar_filename)
            reject_filename = os.path.join(
                savefile_path, "mne_fif", "rejected", "sub" + str(sub_num).zfill(2) +
                '-reject-log.npz')
            os.makedirs(os.path.dirname(reject_filename), exist_ok=True)
            reject_log.save(reject_filename, overwrite=True)
            print("Saving reject log to %s" % reject_filename)

    text = "Ending Preprocessing subject " + str(sub_num) + " ..."
    text = "* " + text + " *"
    border = "*" * len(text)

    print("\033[92m")  # 开始红色文本
    print(border)
    print(text)
    print(border)
    print("\033[0m")  # 结束红色文本

#
# Show the result
#

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
    for d in processed_data_list:
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
        If the condition list has only one element, it returns a list where each element 
        is the ERP corresponding to the given condition.
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
                                                        title=chan_name + "Difference Wave")
    return difference_wave_plot


def calc_erp_ttest(processed_data_list: list, condition_list: list, time_window: list, 
                   direction: str, ch_name='eeg'):
    if ch_name == "eeg":
        ch_nums = 64
    else:
        ch_nums = 1
    # condition 1
    condition = condition_list[0]
    cond1_array = np.zeros((ch_nums, len(processed_data_list)))
    for i in range(len(processed_data_list)):
        evoke_data = mne.read_epochs(processed_data_list[i])[condition].average(
        ).get_data(tmin=time_window[0], tmax=time_window[1])
        evoke_mean = evoke_data.mean(axis=1)
        cond1_array[:, i] = evoke_mean

    # condition 2
    condition = condition_list[1]
    cond2_array = np.zeros((ch_nums, len(processed_data_list)))
    for i in range(len(processed_data_list)):
        evoke_data = mne.read_epochs(processed_data_list[i])[condition].average(
        ).get_data(tmin=time_window[0], tmax=time_window[1])
        evoke_mean = evoke_data.mean(axis=1)
        cond2_array[:, i] = evoke_mean

    # ttest
    tt_results = []
    for chan_num in range(cond1_array.shape[0]):
        cond1_data = cond1_array[chan_num - 1, :]
        cond2_data = cond2_array[chan_num - 1, :]
        tt_result = ttest_rel(cond1_data, cond2_data, alternative=direction)

        # 计算置信区间
        se_diff = sem(cond1_data - cond2_data)
        df = len(cond1_data) - 1
        ci_95 = se_diff * np.array(t.interval(0.95, df))

        tt_results.append([tt_result.statistic, tt_result.pvalue,
                           ci_95, tt_result.pvalue < 0.05])

    if ch_name == "eeg":
        chan_name = mne.read_epochs(processed_data_list[i])[
            condition].info['ch_names'][0:64]
    else:
        chan_name = ch_name

    df = pd.DataFrame(tt_results, columns=['T-Statistic', 'P-Value',
                                           '95% CI', 'Significant'])
    df.index = chan_name

    return df

    # # ttest
    # tt_results = np.zeros(cond1_array.shape[0])
    # for chan_num in range(cond1_array.shape[0]):
    #     cond1_data = cond1_array[chan_num - 1, :]
    #     cond2_data = cond2_array[chan_num - 1, :]
    #     foo = ttest_rel(cond1_data, cond2_data, alternative=direction)
    #     tt_results[chan_num - 1] = foo[1]

    # if ch_name == "eeg":
    #     chan_name = mne.read_epochs(processed_data_list[i])[
    #         condition].info['ch_names'][0:63]
    # else:
    #     chan_name = ch_name

    # tt_results_dict = dict(zip(chan_name, tt_results))

    # return tt_results_dict
