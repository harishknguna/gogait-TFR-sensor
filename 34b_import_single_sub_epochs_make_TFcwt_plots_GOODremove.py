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
# event_type = ['cue', 'target']
# version_list = ['GOODremove','CHANremove']
version_list = ['CHANremove']
ep_extension = 'TF'
waveType = 'complex' #'real' | 'complex'

for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['GOu'] 
    condi_name = ['GOc', 'GOu', 'NoGo']
    num_condi_with_eve = 6
    n_chs = 132
   
    sampling_freq = 500 # in hz
    
    for veri, version in enumerate(version_list):
        
        for ci, condi in enumerate(condi_name): 
            report_bslnCorr = mne.Report()
            report_no_bslnCorr = mne.Report()
            print("condition: %s" % condi)        
      
            for sub_num, subject in enumerate(config_for_gogait.subjects_list): 
                print("Processing subject: %s" % subject)
                eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
                      
                #%% reading the epochs from disk
                print('Reading the epochs from disk')
                
                if ep_extension == 'TF':
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_'+ ep_extension +'_epo'
                else:
                    extension = condi_name[ci] +'_' + event_type[ei] +'_' + version + '_epo'
                
                epochs_fname = op.join(eeg_subject_dir_GOODremove,
                                          config_for_gogait.base_fname.format(**locals()))
               
                # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                print("Input: ", epochs_fname)
                epochs = mne.read_epochs(epochs_fname, proj=True, preload=True).pick('eeg')  
                info = epochs.info   # taking info,events,id from the last sub's data (same for all subs)
                
               
                #%% compute and plot the TF maps
               
                # define frequencies of interest (log-spaced)
                # freqs = np.logspace(*np.log10([1, 40]), num=40)
                freqs = np.linspace(3,40,num = 40, endpoint= True)
                n_cycles = 3 # variable time window T = n_cycles/freqs (in sec)
                                # = should be equal or smaller than signal  
                
                ## (0.5 s window for 1 Hz, 0.0125 s for 40 Hz or 10 ms for 50 Hz )
                # n_cycles = 0.5* freqs / 2.0  # fixed time window or different number of cycle per frequency
                power, itc = tfr_multitaper(
                    epochs,
                    freqs = freqs,
                    n_cycles = n_cycles,
                    time_bandwidth = 2.0, # freq_bandwidth = time_bandwidth/T
                    use_fft = True,
                    return_itc = True,
                    decim = 5, ## time res = fs/decim = 500/5 = 100 or 10 ms
                    average = True, # average all the epochs
                    n_jobs = None,
                ) 
                
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
                fig.tight_layout()
                report_bslnCorr.add_figure(fig, title = subject + '_tfrplt', replace = True)
                
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
                fig.tight_layout()
                report_no_bslnCorr.add_figure(fig, title = subject + '_tfrplt', replace = True)

                plt.close('all')
                
            #     #%%   saving the tfr data in numpy array baseline NOT corrected
            #     print('\n Writing the TFR_power to disk: no bsnlneCorr')
            #     extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_PWR-tfr.h5'
            #     tfr_fname = op.join(eeg_subject_dir_GOODremove,
            #                             config_for_gogait.base_fname_no_fif.format(**locals()))
               
            #     # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
            #     print("Output: ", tfr_fname)
            #     mne.time_frequency.write_tfrs(tfr_fname, power, overwrite=True)    
                
            #     print('\n Writing the TFR_itc to disk: no bsnlneCorr')
            #     extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_ITC-tfr.h5'
            #     tfr_fname = op.join(eeg_subject_dir_GOODremove,
            #                             config_for_gogait.base_fname_no_fif.format(**locals()))
               
            #     # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
            #     print("Output: ", tfr_fname)
            #     mne.time_frequency.write_tfrs(tfr_fname, itc, overwrite=True)  
                
            #     # itc_read = mne.time_frequency.read_tfrs(tfr_fname, condition=None)
            #     # itc_read[0].plot(baseline=(-0.2, -0.1), mode="logratio", title = 'auto',
            #     #            exclude = 'bads', combine = 'mean')
                
            #     #%%   saving the tfr data in numpy array baseline corrected
            #     print('\n Writing the TFR_power to disk: bslnCorr')
            #     extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_PWR_bslnCorr-tfr.h5'
            #     tfr_fname = op.join(eeg_subject_dir_GOODremove,
            #                             config_for_gogait.base_fname_no_fif.format(**locals()))
               
            #     # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
            #     print("Output: ", tfr_fname)
            #     power_bslnCorr = power.apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
            #     mne.time_frequency.write_tfrs(tfr_fname, power, overwrite=True)    
                
            #     print('\n Writing the TFR_itc to disk: bslnCorr')
            #     extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_ITC_bslnCorr-tfr.h5'
            #     tfr_fname = op.join(eeg_subject_dir_GOODremove,
            #                             config_for_gogait.base_fname_no_fif.format(**locals()))
               
            #     # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
            #     print("Output: ", tfr_fname)
            #     itc_bslnCorr = itc.apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
            #     mne.time_frequency.write_tfrs(tfr_fname, itc, overwrite=True)  
                
            #     #%%   saving the tfr times in numpy array
            #     print('\n Writing the TFR_times to disk')
            #     extension =  'TFR_times_'+ event_type[0] +'.npy'
            #     tfr_fname = op.join(config_for_gogait.eeg_dir_GOODremove,
            #                             config_for_gogait.base_fname_generic.format(**locals()))
            #     tfr_times = itc._raw_times
            #     print("Output: ", tfr_fname)
            #     np.save(tfr_fname, tfr_times)  
           
           
            # # finally saving the report after the for subject loop ends.     
            # print('\n Saving the reports to disk')  
            # report_bslnCorr.title = 'Single sub TF baseline corr: ' + condi + '_' + evnt+ '_' + version 
            # #report.title = 'Group sub STC contrast at ' + evnt
            # extension = 'single_sub_TF_bslnCorr'
            # report_bslnCorr_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
            # report_bslnCorr.save(report_bslnCorr_fname+'_' + condi +'_'+ evnt + '_' + version+ '.html', overwrite=True)            
            
            # # finally saving the report after the for subject loop ends.     
            # print('\n Saving the reports to disk')  
            # report_no_bslnCorr.title = 'Single sub TF No baseline corr: ' + condi + '_' + evnt+ '_' + version 
            # #report.title = 'Group sub STC contrast at ' + evnt
            # extension = 'single_sub_TF_NO_bslnCorr'
            # report_no_bslnCorr_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
            # report_no_bslnCorr.save(report_no_bslnCorr_fname+'_' + condi +'_'+ evnt + '_' + version+ '.html', overwrite=True)            
                   
   
   



#%%
 # itc.plot(title='auto', cmap = 'Reds', vmin = 0, vmax = 1, 
 #          exclude = 'bads', combine = 'mean')
 
 

 # power.plot_joint(topomap_args = dict(sphere = 0.45),
 #                baseline=(-0.2, 0), mode="mean", tmin=-0.2, tmax=2, 
 #                timefreqs=[(0.1, 10), (0.2, 10)]
 #                 )
 # power.plot_topomap(tmin= -0.2, tmax=0,fmin= 7.0, fmax= 12.0, sphere = 0.45)
 
 # AttributeError: 'EpochsTFR' object has no attribute 'plot'
 # same as doing in the above with default settings
 # avgTFR = power.average()
 # avgTFR.plot(baseline=(-0.5, -0.1), mode="logratio", title = 'auto',
 #            exclude = 'bads', combine = 'mean')


 # for n_cycles upto 1, error occurs:  
 # ValueError: At least one of the wavelets is longer than the signal. Use a longer signal or shorter wavelets.   
               

    