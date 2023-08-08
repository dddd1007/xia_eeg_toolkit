import multiprocessing
import os

import mne
import numpy as np
from autoreject import AutoReject


def print_message(message, color_code):
    """Helper function to print colored messages with borders."""
    text = f"* {message} *"
    border = "*" * len(text)
    print(f"\033[{color_code}m{border}\n{text}\n{border}\033[0m")


def save_to_path(save_path, subfolder, filename, sub_num):
    """Helper function to create file paths."""
    file_path = os.path.join(save_path, "mne_fif", subfolder, f"sub{sub_num:02}-{filename}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path


def preprocess_epoch_data(raw_data_path, montage_file_path, event_file_path, savefile_path,
                          input_event_dict, rm_chans_list, eog_chans, sub_num, tmin=-0.3, tmax=1.2,
                          l_freq=1, h_freq=30, ref_channels='average', ica=True, ica_z_thresh=1.96,
                          auto_reject=True, export_to_mne=True):

    print_message(f"Beginning preprocessing subject {sub_num} ...", "91")

    # Import data
    raw = mne.io.read_raw_curry(raw_data_path, preload=True)
    channels_to_drop = ['HL 1', 'HL 2', 'Trigger']
    for ch in channels_to_drop:
        if ch in raw.ch_names:
            raw.drop_channels([ch])

    # Set channel types and montage
    raw.info['bads'].extend(rm_chans_list or [])
    raw.set_channel_types({chan: 'eog' for chan in eog_chans})
    montage = mne.channels.read_custom_montage(montage_file_path,
                                               ch_names=[ch for ch in raw.ch_names if ch not in ['HEO', 'VEO']])
    raw.set_montage(montage)

    # Filter the data
    filter_data = raw.filter(l_freq, h_freq, n_jobs=multiprocessing.cpu_count())

    # ICA processing
    if ica:
        print(" == Doing the ICA ==")
        try:
            ica_method = mne.preprocessing.ICA(n_components=.999, method='picard')
            ica_method.fit(filter_data)
        except RuntimeError:
            print("Error encountered with n_components=0.999, trying with n_components=10")
            ica_method = mne.preprocessing.ICA(n_components=10, method='picard')
            ica_method.fit(filter_data)

        eog_indices, _ = ica_method.find_bads_eog(raw, ch_name=['FP1', 'FP2', 'F8', 'HEO', 'VEO'], threshold=ica_z_thresh)
        muscle_indices, _ = ica_method.find_bads_muscle(raw, threshold=ica_z_thresh)
        ica_method.exclude = muscle_indices + eog_indices

        if export_to_mne:
            ica_file_path = save_to_path(savefile_path, "ICA", "ica.fif", sub_num)
            ica_method.save(ica_file_path, overwrite=True)

        ica_data = ica_method.apply(filter_data)
        ica_data.set_eeg_reference(ref_channels)
    else:
        ica_data = filter_data

    # Annotate with events
    events = mne.read_events(event_file_path)
    annotations = mne.annotations_from_events(events, event_desc=input_event_dict, sfreq=raw.info['sfreq'], orig_time=raw.info['meas_date'])
    ica_data.set_annotations(annotations)
    ica_data.drop_channels(['HEO', 'VEO'])

    # Save the data before epoching
    if export_to_mne:
        save_path = save_to_path(savefile_path, "before_epoch", "after-filter.fif", sub_num)
        ica_data.save(save_path, overwrite=True)

    # Create epochs
    events, event_dict = mne.events_from_annotations(ica_data)
    ica_data.del_proj()
    epochs = mne.Epochs(ica_data, events, event_dict, tmin, tmax, baseline=(None, 0), verbose=False, detrend=0, preload=True)

    # Save epochs before autoreject
    if export_to_mne:
        save_path = save_to_path(savefile_path, "before_reject", "before-reject-epo.fif", sub_num)
        epochs.save(save_path, overwrite=True)

    # Autoreject processing
    if auto_reject:
        print(" == Doing the Autoreject ==")

        # Define the parameter range
        n_interpolates = np.array([1, 4, 32])
        consensus_percs = np.linspace(0, 1.0, 11)

        # Initialize AutoReject object with parameter ranges
        ar = AutoReject(n_interpolate=n_interpolates,
                        n_jobs=multiprocessing.cpu_count(),
                        verbose=True)

        # Fit the data (AutoReject will automatically choose the best parameters)
        ar.fit(epochs)

        # Apply the automatic rejection
        epochs_ar, reject_log = ar.transform(epochs, return_log=True)

        print(f"Best n_interpolate: {ar.n_interpolate_}")
        print(f"Best consensus_perc: {ar.consensus_perc_}")

        if export_to_mne:
                save_path = save_to_path(savefile_path, "rejected", "epo.fif", sub_num)
                epochs_ar.save(save_path, overwrite=True)

                reject_log_path = save_to_path(savefile_path, "rejected", "reject-log.npz", sub_num)
                reject_log.save(reject_log_path, overwrite=True)

    print_message(f"Ending Preprocessing subject {sub_num} ...", "92")
