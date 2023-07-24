import os
from autoreject import AutoReject
import mne
import multiprocessing

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
    try:
        ica = mne.preprocessing.ICA(n_components=.999, method='picard')
        ica.fit(filter_data)
    except RuntimeError:
        print("Error encountered with n_components=0.999, trying with n_components=10")
        ica = mne.preprocessing.ICA(n_components=10, method='picard')
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