# Parameters example
# raw_data_path = "/Users/dddd1007/research/Project4_EEG_Volatility_to_Control/data/input/eeg/sub01.cdt"
# savefile_path = "/Users/dddd1007/research/Project4_EEG_Volatility_to_Control/data/output"
# montage_file_path ="/Users/dddd1007/research/Project4_EEG_Volatility_to_Control/data/resource/ch_loc_export_from_eeglab.loc"
# event_file_path = "/Users/dddd1007/research/Project4_EEG_Volatility_to_Control/data/output/exp_event_file/sub01_new_event_file_mne_eve.txt"
# rm_chans_list = ['F1', 'F5', 'CP3']
# input_event_dict = {31: 'con/MC/s', 32: 'inc/MC/s', 33: 'con/MI/s', 34: 'inc/MI/s',
#                     41: 'con/MC/v', 42: 'inc/MC/v', 43: 'con/MI/v', 44: 'inc/MI/v',
#                     98: 'corr_resp', 99: 'wrong_resp'}
# eog_chans = ['HEO', 'VEO']
# sub_num = 1
# tmin = -0.3
# tmax=1.2
# l_freq=1
# h_freq=30
# ica_z_thresh = 1.96

def preprocess_epoch_data(raw_data_path,  montage_file_path, event_file_path, input_event_dict:dict,
                          savefile_path, rm_chans_list:list, eog_chans:list, sub_num,
                          tmin = -0.3, tmax=1.2, l_freq=1, h_freq=30, ica_z_thresh = 1.96,
                          export_to_mne = True,export_to_eeglab=False):
    import os
    from autoreject import (get_rejection_threshold,AutoReject)
    import mne
    import numpy as np
    os.environ['OPENBLAS_NUM_THREADS'] = '6'

    # Import data
    raw = mne.io.read_raw_curry(raw_data_path, preload=True)

    # Set the channel type
    raw.info['bads'].extend(rm_chans_list)
    raw.set_channel_types(dict(zip(eog_chans, ['eog']*len(eog_chans))))

    # Set the montage
    montage_file = mne.channels.read_custom_montage(montage_file_path)
    raw.set_montage(montage_file)

    # Set the preprocessing setting
    filter_data = raw.copy()

    # Remove channels which are not needed
    filter_data.drop_channels(rm_chans_list)

    # Filter the data
    filter_data.filter(l_freq, h_freq, n_jobs=6)

    # prepare for ICA
    tstep = 1.0
    events_forica = mne.make_fixed_length_events(filter_data, duration=tstep)
    epochs_forica = mne.Epochs(filter_data, events_forica,
                            tmin=0.0, tmax=tstep,
                            baseline=None,
                            preload=True)
    reject_para = get_rejection_threshold(epochs_forica)

    # compute ICA
    ica = mne.preprocessing.ICA(n_components=.99, method='picard');
    ica.fit(epochs_forica);

    # Exclude blink artifact components
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['FP1', 'FP2', 'F8'], threshold=ica_z_thresh)
    muscle_index, muscle_scores = ica.find_bads_muscle(raw, threshold=ica_z_thresh)
    ica.exclude = muscle_index + eog_indices
    ica.save(os.path.join(savefile_path, 'mne_fif', 'ICA', 'sub'+str(sub_num) + '-ica.fif'), overwrite=True)
    ica_data = ica.apply(filter_data.copy())

    # Rereference
    ica_data.set_eeg_reference(ref_channels='average')

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
                        baseline=(None, 0), reject=reject_para,
                        verbose=False, detrend=0, preload=True)

    # Autoreject
    ar = AutoReject(n_jobs=6, verbose=True)
    ar.fit(epochs[:60])  # fit on a few epochs to save time
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)

    # Create evoked data
    conditions = [i for i in input_event_dict.values()]
    evokes = [epochs_ar[cond].average() for cond in conditions]
    # TODO 将这里的文件生成改好

    if export_to_mne:
        epochs_filename =  os.path.join(savefile_path, "mne_fif" ,"sub" + str(sub_num) + '-epo.fif')
        evokes_filename = os.path.join(savefile_path, "mne_fif", "sub" + str(sub_num) + '-ave.fif')
        epochs_ar.save(epochs_filename, overwrite=True)
        mne.write_evokeds(evokes_filename, evokes)

    # # Export file to EEGLAB format
    # if export_to_eeglab:
    #     fileout = os.path.splitext(raw_data_path)[0];
    #     fileout_cond1 = fileout + '_cond1_mne.set'
    #     fileout_cond2 = fileout + '_cond2_mne.set'
    #     epochs_ar[cond1].export(fileout_cond1, overwrite=True)
    #     epochs_ar[cond2].export(fileout_cond2, overwrite=True)
