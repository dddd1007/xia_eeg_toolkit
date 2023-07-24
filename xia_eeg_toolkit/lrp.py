import numpy as np
import mne

def compute_lrp(epochs, ch1='C3', ch2='C4'):
    """Compute LRP by subtracting the average signal at one electrode from the other."""
    data1 = epochs.copy().pick_channels([ch1]).get_data()
    data2 = epochs.copy().pick_channels([ch2]).get_data()
    lrp = data1 - data2
    return lrp

def compute_and_average_lrp(epochs, ch1='C3', ch2='C4'):
    """Compute and average the LRP for left-hand and right-hand responses."""
    # Select correct responses only
    epochs_corResp = epochs['corResp']

    # Further divide the epochs into left-hand and right-hand responses
    epochs_left = epochs_corResp['leftHand'].load_data()
    epochs_right = epochs_corResp['rightHand'].load_data()

    # Compute LRP for left-hand and right-hand responses
    lrp_left = compute_lrp(epochs_left, ch1, ch2)
    lrp_right = compute_lrp(epochs_right, ch1, ch2)

    # Average LRP across epochs
    lrp_left_mean = lrp_left.mean(axis=0)
    lrp_right_mean = lrp_right.mean(axis=0)

    # Package the results into a dictionary
    results = {
        'lrp_left': lrp_left,
        'lrp_right': lrp_right,
        'lrp_left_mean': lrp_left_mean,
        'lrp_right_mean': lrp_right_mean
    }

    return results

def compute_group_average_lrp(data_cache, plot=True):
    """Compute the group-average LRP and return as Evoked objects."""
    lrps_left, lrps_right = [], []
    lrps_left_mean, lrps_right_mean = [], []

    # Loop over each participant
    for epochs in data_cache.values():
        # Compute LRP and average LRP for each participant
        results = compute_and_average_lrp(epochs)

        # Append to the lists
        lrps_left.append(results['lrp_left'])
        lrps_right.append(results['lrp_right'])
        lrps_left_mean.append(results['lrp_left_mean'])
        lrps_right_mean.append(results['lrp_right_mean'])

    # Convert to numpy arrays
    lrps_left = np.array(lrps_left)
    lrps_right = np.array(lrps_right)
    lrps_left_mean = np.array(lrps_left_mean)
    lrps_right_mean = np.array(lrps_right_mean)

    # Compute group averages
    lrps_left_mean_group = lrps_left.mean(axis=0)
    lrps_right_mean_group = lrps_right.mean(axis=0)

    # Create mne.Info object
    info = mne.create_info(ch_names=['LRP'], sfreq=epochs.info['sfreq'])

    # Create mne.Evoked objects
    evoked_left = mne.EvokedArray(lrps_left_mean_group, info)
    evoked_right = mne.EvokedArray(lrps_right_mean_group, info)

    # Package the results into a dictionary
    results_group = {
        'lrp_left': lrps_left,
        'lrp_right': lrps_right,
        'lrp_left_mean': lrps_left_mean,
        'lrp_right_mean': lrps_right_mean,
        'evoked_left': evoked_left,
        'evoked_right': evoked_right
    }

    if plot:
        mne.viz.plot_compare_evokeds(dict(Left=results_group['evoked_left'],
                                          Right=results_group['evoked_right']))

    return results_group