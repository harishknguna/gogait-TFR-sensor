#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:34:24 2024
 https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html
 https://mne.tools/stable/generated/mne.time_frequency.tfr_multitaper.html#mne.time_frequency.tfr_multitaper
@author: harish.gunasekaran
"""

"""
=============================================
import single sub evoked and make Evoked plots and topomaps 

==============================================


"""  



import os.path as op

import mne
from mne.parallel import parallel_func
from mne.channels.montage import get_builtin_montages
from warnings import warn
from pymatreader import read_mat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from scipy import signal
from scipy import stats 
from scipy.linalg import norm
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse 
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid, inset_locator
from mne.stats import spatio_temporal_cluster_test, summarize_clusters_stc
from mne.time_frequency import tfr_multitaper

import config_for_gogait

def find_closest(arr, val):
    idx = np.abs(arr - val).argmin()
    return arr[idx]

n_subs = len(config_for_gogait.subjects_list)
fs_dir = fetch_fsaverage(verbose=True)
template_subjects_dir = op.dirname(fs_dir)

# ampliNormalization = ['AmpliNorm', 'AmpliActual']
# ampliNormalization = 'AmpliActual'

event_type = ['target']
baseline = 'bslnNoCorr' # 'bslnCorr' | 'bslnNoCorr' 
corr = 'CorrApply'  # 'CorrApply' | 'CorrNotApply'
avgType = 'perSUB' # 'perSUB' | 'perGROUP'


## note: always take baseline corr version, it allows to do common scaling before plotting
version_list = ['CHANremove']
ep_extension = 'TF'
waveType = 'morlet' # 'morlet' | 'multitaper'
numType = 'real' # 'real' | 'complex'

if version_list[0] == 'GOODremove':
    n_chs = 128
elif version_list[0] == 'CHANremove':
    n_chs = 103

for ei, evnt in enumerate(event_type):
    sfreq = 500
    decim = 5 # when doing tfr computation
    sfreq = sfreq/decim
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['GOu'] 
    condi_name = ['GOc', 'GOu', 'NoGo']
    ncondi = len(condi_name)
   
    # sampling_freq = 500 # in hz
    tfr_freqs = np.linspace(3,40,num = 40, endpoint= True)
    n_TF_freqs = len(tfr_freqs)
    
    for veri, version in enumerate(version_list):
        
        
        for ci, condi in enumerate(condi_name): 
            
            print("condition: %s" % condi)        
       
            ## n_samples_esti = int(sampling_freq * (isi[ind_isi] + t_end - t_start + 0.002))  
            # estimate the num of time samples per condi/ISI to allocate numpy array
            ## added on 15/01/2024
            
            if evnt == 'cue' and ep_extension == 'TF':
                tsec_start = 0.8 # pre-stimulus(t2/S2/S4) duration in sec
                tsec_end = 1.3 # post-stimulus (t3/S8/S16) duration in sec
            elif evnt == 'target' and ep_extension == 'TF':
                tsec_start = 1.0 # pre-stimulus(t2/S2/S4) duration in sec
                tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
            else:
                tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
                tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec      
      
            
            n_samples_esti  = int(sfreq*(tsec_start + tsec_end + 1/sfreq)) # one sample added for zeroth loc
            
            # if numType == 'real': 
            tf_pwr_array_all_sub = np.empty([n_subs, n_chs, n_TF_freqs, n_samples_esti])
            # elif numType == 'complex':
            #     tf_pwr_array_all_sub = np.empty([n_subs, n_chs, n_TF_freqs, n_samples_esti]).astype(complex)
                
            # tf_itc_array_all_sub = np.ones([n_subs, n_chs, n_TF_freqs, n_samples_esti])*np.nan
           
            for sub_num, subject in enumerate(config_for_gogait.subjects_list): 
                print("Processing subject: %s" % subject)
                eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
                      
                #% reading the epochs from disk
                print('Reading the epochs from disk')
                
                if ep_extension == 'TF':
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_'+ ep_extension +'_epo'
                else:
                    extension = condi_name[ci] +'_' + event_type[ei] +'_' + version + '_epo'
                
                epochs_fname = op.join(eeg_subject_dir_GOODremove,
                                          config_for_gogait.base_fname.format(**locals()))
               
                # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                print("Input: ", epochs_fname)
                epochs = mne.read_epochs(epochs_fname, proj=True, preload=True).pick('eeg', exclude='bads')  
                info = epochs.info   # taking info,events,id from the last sub's data (same for all subs)
                
                #% importing TF files of each sub from disk
                if numType == 'real':
                    if baseline == 'bslnCorr':
                        print('Importing the TFR_power_bslnCorr from disk')
                        extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR_bslnCorr-tfr.h5'
                    elif baseline == 'bslnNoCorr':
                        print('Importing the TFR_power from disk')
                        extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR-tfr.h5'
               
                elif numType == 'complex':
                    if baseline == 'bslnCorr':
                        print('Importing the TFR_power_bslnCorr from disk')
                        extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR_CMPLX_bslnCorr-tfr.h5'
                    elif baseline == 'bslnNoCorr':
                        print('Importing the TFR_power from disk')
                        extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR_CMPLX-tfr.h5'
                        
                tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                        config_for_gogait.base_fname_no_fif.format(**locals()))
               
                print("Output: ", tfr_fname)
                pwr_read = mne.time_frequency.read_tfrs(tfr_fname)
                
                if numType == 'real':
                    pwr_per_sub = pwr_read[0].data
                elif numType == 'complex':
                    pwr_per_sub = pwr_read[0].data.mean(axis = 0) # avg across epochs
                
                ##"NB: Now doing baseline correction PER SUB using MNE containers"
                if corr == 'CorrApply' and avgType == 'perSUB': 
                    #% put them in the respective MNE containers
                    print('Importing the TFR_times from disk')
                    extension =  'TFR_times_'+ event_type[0] +'.npy'
                    tfr_fname = op.join(config_for_gogait.eeg_dir_GOODremove,
                                            config_for_gogait.base_fname_generic.format(**locals()))
                 
                    print("Input: ", tfr_fname)
                    tfr_times = np.load(tfr_fname)  
                    power_perSub = mne.time_frequency.AverageTFR(info = info, 
                                                          data = pwr_per_sub ,
                                                          times = tfr_times, 
                                                          freqs = tfr_freqs, 
                                                          nave = 1, 
                                                          comment = 'avgPower' ,
                                                          method = 'morlet')
                    
                    power_perSub_bslnCorr = power_perSub.copy().apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
                    
                    ## get back to numpy array format
                    pwr_data_per_sub = power_perSub_bslnCorr.data
                
                elif corr == 'corrNotApply' or avgType == 'perGROUP':
                    pwr_data_per_sub = pwr_per_sub.copy()

                # store sub in dim1, chs in dim2, freq in dim 3, time in dim4 
                pwr_per_sub_exp_dim = np.expand_dims(pwr_data_per_sub, axis = 0) 
                # itc_per_sub_exp_dim = np.expand_dims(itc_per_sub, axis = 0) 
                
                if sub_num == 0:
                    tf_pwr_array_all_sub = pwr_per_sub_exp_dim
                    # tf_itc_array_all_sub = itc_per_sub_exp_dim
                else:
                    tf_pwr_array_all_sub = np.vstack((tf_pwr_array_all_sub, pwr_per_sub_exp_dim))                     
                    # tf_itc_array_all_sub = np.vstack((tf_itc_array_all_sub, itc_per_sub_exp_dim))                     
                    
            # averaging TF arrays across subjects
            tf_pwr_array_avg_sub = np.mean(tf_pwr_array_all_sub, axis = 0)
            # tf_itc_array_avg_sub = np.mean(tf_itc_array_all_sub, axis = 0)
        
            # store condi in dim1, chs in dim2, freq in dim 3, time in dim4 
            tf_pwr_array_avg_sub_exp_dim =  np.expand_dims(tf_pwr_array_avg_sub, axis = 0) 
            # tf_itc_array_avg_sub_exp_dim =  np.expand_dims(tf_itc_array_avg_sub, axis = 0) 
            
            if ci == 0:
                tf_pwr_array_avg_sub_all_condi = tf_pwr_array_avg_sub_exp_dim 
                # tf_itc_array_avg_sub_all_condi = tf_itc_array_avg_sub_exp_dim
            else:
                tf_pwr_array_avg_sub_all_condi = np.vstack((tf_pwr_array_avg_sub_all_condi, tf_pwr_array_avg_sub_exp_dim))                     
                # tf_itc_array_avg_sub_all_condi = np.vstack((tf_itc_array_avg_sub_all_condi, tf_itc_array_avg_sub_exp_dim)) 
            
            
            
            
        #%% put them in the respective MNE containers
        tfr_all = tf_pwr_array_avg_sub_all_condi.copy()
        
        print('Importing the TFR_times from disk')
        extension =  'TFR_times_'+ event_type[0] +'.npy'
        tfr_fname = op.join(config_for_gogait.eeg_dir_GOODremove,
                                config_for_gogait.base_fname_generic.format(**locals()))
     
        print("Input: ", tfr_fname)
        tfr_times = np.load(tfr_fname)  
    
        for ci, condi in enumerate(condi_name): 
            pwr_array = tfr_all[ci,:,:,:]
            power_group = mne.time_frequency.AverageTFR(info = info, 
                                                  data = pwr_array,
                                                  times = tfr_times, 
                                                  freqs = tfr_freqs, 
                                                  nave = 1, 
                                                  comment = 'avgPower' ,
                                                  method = 'morlet')
            
            if corr == 'CorrApply' and avgType == 'perGROUP':
                power_group_bslnCorr = power_group.copy().apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
                tfr_group_data =  power_group_bslnCorr.copy() 
            
            else: 
                tfr_group_data =  power_group.copy() 
                
    
            ##  saving the complex epochsTFR: no baseline correction applied 
            print('\n Writing the grand averaged complex epochsTFR to disk')
            extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_'+ numType +'_'+ baseline+ '_' + corr + '_' + avgType + '_grand_avg-tfr.h5'
            tfr_fname = op.join(config_for_gogait.eeg_dir_GOODremove,
                                    config_for_gogait.base_fname_generic.format(**locals()))
           
            print("Output: ", tfr_fname)
            mne.time_frequency.write_tfrs(tfr_fname,  tfr_group_data, overwrite=True)  
   
 
      


    