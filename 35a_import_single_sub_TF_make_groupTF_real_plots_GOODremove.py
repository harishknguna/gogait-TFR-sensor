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
n_subs = len(config_for_gogait.subjects_list)
fs_dir = fetch_fsaverage(verbose=True)
template_subjects_dir = op.dirname(fs_dir)

# ampliNormalization = ['AmpliNorm', 'AmpliActual']
# ampliNormalization = 'AmpliActual'

event_type = ['target']
baseline = 'bslnNoCorr' # 'bslnCorr' | 'bslnNoCorr'
event_type = ['cue']
# version_list = ['GOODremove','CHANremove']
version_list = ['CHANremove']
ep_extension = 'TF'
waveType = 'morlet'
numType = 'complex' # 'real' | 'complex'

if version_list[0] == 'GOODremove':
    n_chs = 128
elif version_list[0] == 'CHANremove':
    n_chs = 103

for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['GOu'] 
    condi_name = ['GOc', 'GOu', 'NoGo']
   
    # sampling_freq = 500 # in hz
    tfr_freqs = np.linspace(3,40,num = 40, endpoint= True)
    n_TF_freqs = len(tfr_freqs)
    
    for veri, version in enumerate(version_list):
        report = mne.Report()
        
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
      
            
            n_samples_esti  = int(500*(tsec_start + tsec_end + 1/500)) # one sample added for zeroth loc
            tf_pwr_array_all_sub = np.ones([n_subs, n_chs, n_TF_freqs, n_samples_esti])*np.nan
            tf_itc_array_all_sub = np.ones([n_subs, n_chs, n_TF_freqs, n_samples_esti])*np.nan
           
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
                
               
                #%% importing TF files of each sub from disk
                
                if numType == 'real':
                    if baseline == 'bslnCorr':
                        print('Importing the TFR_power_bslnCorr from disk')
                        extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR_bslnCorr-tfr.h5'
                    elif baseline == 'bslnNoCorr':
                        print('Importing the TFR_power from disk')
                        extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR-tfr.h5'
                        
                        
                    tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                            config_for_gogait.base_fname_no_fif.format(**locals()))
                   
                    print("Output: ", tfr_fname)
                    pwr_read = mne.time_frequency.read_tfrs(tfr_fname)
                    
                    pwr_per_sub = pwr_read[0].data
                    
                    "NB. Loading is long for first time alone"
                   
                    if baseline == 'bslnCorr':
                        print('Importing the TFR_itc_bslnCorr from disk')
                        extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_ITC_bslnCorr-tfr.h5'
                    elif baseline == 'bslnNoCorr':
                        print('Importing the TFR_itc from disk')
                        extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_ITC-tfr.h5'
                
                    tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                            config_for_gogait.base_fname_no_fif.format(**locals()))
               
                    print("Output: ", tfr_fname)
                    itc_read = mne.time_frequency.read_tfrs(tfr_fname, condition=None)
                    itc_per_sub = itc_read[0].data 
                
                    # store sub in dim1, chs in dim2, freq in dim 3, time in dim4 
                    pwr_per_sub_exp_dim = np.expand_dims(pwr_per_sub, axis = 0) 
                    itc_per_sub_exp_dim = np.expand_dims(itc_per_sub, axis = 0) 
                    
                    if sub_num == 0:
                        tf_pwr_array_all_sub = pwr_per_sub_exp_dim
                        tf_itc_array_all_sub = itc_per_sub_exp_dim
                    else:
                        tf_pwr_array_all_sub = np.vstack((tf_pwr_array_all_sub, pwr_per_sub_exp_dim))                     
                        tf_itc_array_all_sub = np.vstack((tf_itc_array_all_sub, itc_per_sub_exp_dim))                     
                        
                # averaging TF arrays across subjects
                tf_pwr_array_avg_sub = np.mean(tf_pwr_array_all_sub, axis = 0)
                tf_itc_array_avg_sub = np.mean(tf_itc_array_all_sub, axis = 0)
            
            #%% put them in the respective MNE containers
            print('Importing the TFR_times to disk')
            extension =  'TFR_times_'+ event_type[0] +'.npy'
            tfr_fname = op.join(config_for_gogait.eeg_dir_GOODremove,
                                    config_for_gogait.base_fname_generic.format(**locals()))
         
            print("Input: ", tfr_fname)
            tfr_times = np.load(tfr_fname)  
            
            power = mne.time_frequency.AverageTFR(info = info, 
                                                  data = tf_pwr_array_avg_sub, 
                                                  times = tfr_times, 
                                                  freqs = tfr_freqs, 
                                                  nave = 1, 
                                                  comment = 'avgPower' ,
                                                  method = 'multitaper')
            
            itc = mne.time_frequency.AverageTFR(info = info, 
                                                  data = tf_itc_array_avg_sub, 
                                                  times = tfr_times, 
                                                  freqs = tfr_freqs, 
                                                  nave = 1,  
                                                  comment = 'avgITC', 
                                                  method = 'multitaper')
            
            # plot the avg TFRs 
            # don't forget %matplotlib qt
            plt.rcParams.update({'font.size': 14})
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
            fig.suptitle('TFR power and ITC')
            power.plot(baseline=(-0.5, -0.1), mode="logratio", title = 'auto',
                       exclude = 'bads', combine = 'mean', axes = ax1)
            ax1.set_title('Power')
            plt.rcParams.update({'font.size': 14})
            itcplot = itc.plot(baseline=(-0.5, -0.1), mode="logratio", title = 'auto',
                       exclude = 'bads', combine = 'mean', axes = ax2)
            ax2.set_title('ITC')
            
            # if event_type[0] == 'cue':
            #     ax1.set_xlim([-0.2, 1.0])
            #     ax2.set_xlim([-0.2, 1.0])
            # elif event_type[0] == 'target':
            #     ax1.set_xlim([-0.2, 0.7])
            #     ax2.set_xlim([-0.2, 0.7])
                
            fig.tight_layout()
            report.add_figure(fig, title = condi + '_tfrplt_bslnCorr', replace = True)
            
            # no baseline corrected
            
            plt.rcParams.update({'font.size': 14})
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
            fig.suptitle('TFR power and ITC')
            power.plot(baseline = None, title = 'auto',
                       exclude = 'bads', combine = 'mean', axes = ax1)
            ax1.set_title('Power')
            plt.rcParams.update({'font.size': 14})
            itcplot = itc.plot(baseline = None, title = 'auto',
                       exclude = 'bads', combine = 'mean', axes = ax2)
            ax2.set_title('ITC')
            
            # if event_type[0] == 'cue':
            #     ax1.set_xlim([-0.2, 1.0])
            #     ax2.set_xlim([-0.2, 1.0])
            # elif event_type[0] == 'target':
            #     ax1.set_xlim([-0.2, 0.7])
            #     ax2.set_xlim([-0.2, 0.7])
            
            fig.tight_layout()
            report.add_figure(fig, title = condi + '_tfrplt', replace = True)
            
                 
                
            plt.close('all')
            
            
           
           
    # finally saving the report after the for subject loop ends.     
    print('Saving the reports to disk')  
    report.title = 'Group sub TF : ' + evnt + '_' + baseline + '_' +  version + '_' + waveType 
    #report.title = 'Group sub STC contrast at ' + evnt
    extension = 'group_sub_TF'
    report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
    report.save(report_fname+'_' + evnt + '_' + baseline + '_' + version+ '_' + waveType + '.html', overwrite=True)            
              
   
   



    