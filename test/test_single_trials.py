import sys
import pandas as pd
sys.path.append("/Users/dddd1007/project2git/xia_eeg_toolkit")
import xia_eeg_toolkit

all_sub_data = pd.read_csv("/Users/dddd1007/Research/Project4_EEG_Volatility_to_Control/data/output/01_behavior/all_sub_data_with_param.csv")
sub_data = all_sub_data[all_sub_data['sub_num'] == 2]
epochs = xia_eeg_toolkit.single_trials.load_epochs("/Users/dddd1007/Downloads/sub02-before-reject-epo.fif")
reject_log = xia_eeg_toolkit.single_trials.load_reject_log("/Users/dddd1007/Downloads/sub02-reject-log.npz")

xia_eeg_toolkit.single_trials.single_trial_analysis(epochs, sub_data, reject_log,
                                                    analysis_var='bl_sr_pe', do_parallel=True)
