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
from mne.time_frequency import tfr_multitaper, tfr_morlet

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
waveType = 'morlet' # 'morlet' | 'multitaper'
numType = 'complex' # 'real' | 'complex'

for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    condi_name = ['NoGo'] 
    # condi_name = ['GOc', 'GOu', 'NoGo']
    num_condi_with_eve = 6
    n_chs = 132
   
    sampling_freq = 500 # in hz
    
    for veri, version in enumerate(version_list):
        
        for ci, condi in enumerate(condi_name): 
            if numType == 'real':
                report_bslnCorr = mne.Report()
                report_no_bslnCorr = mne.Report()
            elif numType == 'complex':
                report_cmplx = mne.Report()
            
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
                # epochs = mne.read_epochs(epochs_fname, proj=True, preload=True).pick('eeg')
                epochs = mne.read_epochs(epochs_fname, proj=True, preload=True).pick('eeg', exclude='bads') 
                epochs = epochs.set_eeg_reference(projection=True)
                info = epochs.info   # taking info,events,id from the last sub's data (same for all subs)
                
               
                #%% compute and plot the TF maps
               
                # define frequencies of interest (log-spaced)
                # freqs = np.logspace(*np.log10([1, 40]), num=40)
                freqs = np.linspace(3,40,num = 40, endpoint= True)
                n_cycles = 3 # variable time window T = n_cycles/freqs (in sec)
                                # = should be equal or smaller than signal  
                
                ## (0.5 s window for 1 Hz, 0.0125 s for 40 Hz or 10 ms for 50 Hz )
                # n_cycles = 0.5* freqs / 2.0  # fixed time window or different number of cycle per frequency
                
               ##  if waveType == 'multitaper' and numType == 'real':  # use multitaper and set average = False
                power_multi, itc = tfr_multitaper(
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
               ## elif waveType == 'morlet' and numType == 'real': 
                pwrReal, itc = tfr_morlet(
                    epochs,
                    freqs = freqs,
                    n_cycles = n_cycles,
                    # time_bandwidth = 2.0, # freq_bandwidth = time_bandwidth/T
                    use_fft = True,
                    return_itc = True,
                    decim = 5, ## time res = fs/decim = 500/5 = 100 or 10 ms
                    average = True, # average all the epochs
                    output = 'power', # default: returns pwr and itc
                    n_jobs = None,
                ) 
                # elif waveType == 'morlet' and numType == 'complex': 
                pwrComplex  = tfr_morlet(
                    epochs,
                    freqs = freqs,
                    n_cycles = n_cycles,
                    # time_bandwidth = 2.0, # freq_bandwidth = time_bandwidth/T
                    use_fft = True,
                    return_itc = False,
                    decim = 5, ## time res = fs/decim = 500/5 = 100 or 10 ms
                    average = False, # average all the epochs
                    output = 'complex',
                    n_jobs = None,
                ) 
     #%%         PLOT MULTITAPER  
             
                plt.rcParams.update({'font.size': 14})
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
               
                avgTFR = power_multi.copy()
                
                ## baseline corrected version
                bslncorr = avgTFR.plot(baseline = (-0.5, -0.1), mode="logratio",
                            exclude = 'bads', combine = 'mean', axes = ax1)
                ax1.set_title('bsln corr')
                
                ## baseline NOT corrected version
                bslnNocorr = avgTFR.plot(baseline=None, title = 'multi-taper (mean 103 chs)',
                            exclude = 'bads', combine = 'mean', axes = ax2)
                ax2.set_title('no bsln corr')
                # ax1.set_title('Power')
                fig.tight_layout()
                
                #%% PLOT REAL WT  
                      
                plt.rcParams.update({'font.size': 14})
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
               
                avgTFR = pwrReal.copy()
                
                ## baseline corrected version
                bslncorr = avgTFR.plot(baseline = (-0.5, -0.1), mode="logratio", 
                            exclude = 'bads', combine = 'mean', axes = ax1)
                ax1.set_title('bsln corr')
                
                ## baseline NOT corrected version
                bslnNocorr = avgTFR.plot(baseline=None, title = 'real morlet (mean 103 chs)',
                            exclude = 'bads', combine = 'mean', axes = ax2)
                ax2.set_title('no bsln corr')
                # ax1.set_title('Power')
                fig.tight_layout()
                    
    #%%        PLOT COMPLEX WT 
            
                plt.rcParams.update({'font.size': 14})
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
                                
                # take average of trials (epochs) to plot the evoked spectrum
                avgTFR = pwrComplex.average(method='mean', dim='epochs')
                
                ## baseline corrected version
                bslncorr = avgTFR.plot(baseline = (-0.5, -0.1), mode="logratio", 
                            exclude = 'bads', combine = 'mean', axes = ax1)
                ax1.set_title('bsln corr')
                
                ## baseline NOT corrected version
                bslnNocorr = avgTFR.plot(baseline=None, title = 'complex morlet (mean 103 chs)',
                            exclude = 'bads', combine = 'mean', axes = ax2)
                ax2.set_title('no bsln corr')
                # ax1.set_title('Power')
                fig.tight_layout()
                
    #%%        PLOT ABS COMPLEX WT 
            
                plt.rcParams.update({'font.size': 14})
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
                                
                # take mag (real) here before average
                avgTFR = abs(pwrComplex).average(method='mean', dim='epochs')
                
                ## baseline corrected version
                bslncorr = avgTFR.plot(baseline = (-0.5, -0.1), mode="logratio", 
                            exclude = 'bads', combine = 'mean', axes = ax1)
                ax1.set_title('bsln corr')
                
                ## baseline NOT corrected version
                bslnNocorr = avgTFR.plot(baseline=None, title = 'abs (.) cmplx morlet (mean 103 chs)',
                            exclude = 'bads', combine = 'mean', axes = ax2)
                ax2.set_title('no bsln corr')
                # ax1.set_title('Power')
                fig.tight_layout()
#%%            PLOT SQRT(Re² + Im²) COMPLEX WT 
        
                plt.rcParams.update({'font.size': 14})
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
                                
                # take real and imag before average and taking abs in numpy array 
                pwrComplex_real = np.real(pwrComplex.data)
                pwrComplex_imag = np.imag(pwrComplex.data)
                pwrComplexABS = np.sqrt(np.square(pwrComplex_real)+ np.square(pwrComplex_imag))
               
                # putting the manually calculated ABS in TFR container
                tfr_times = pwrComplex.times
                tfr_freqs = pwrComplex.freqs
                epochsTFR = mne.time_frequency.EpochsTFR(info = info, 
                                                      data = pwrComplexABS,
                                                      times = tfr_times, 
                                                      freqs = tfr_freqs)
                
                                                     
                avgTFR =  epochsTFR.average(method='mean', dim='epochs')                                         
                
                ## baseline corrected version
                bslncorr = avgTFR.plot(baseline = (-0.5, -0.1), mode="logratio", 
                            exclude = 'bads', combine = 'mean', axes = ax1)
                ax1.set_title('bsln corr')
                
                ## baseline NOT corrected version
                bslnNocorr = avgTFR.plot(baseline=None, title = 'sqrt(re^2 + im^2) (mean 103 chs)',
                            exclude = 'bads', combine = 'mean', axes = ax2)
                ax2.set_title('no bsln corr')
                # ax1.set_title('Power')
                fig.tight_layout()
            
                
                
               
    