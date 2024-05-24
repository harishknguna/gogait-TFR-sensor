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
    # condi_name = ['GOu'] 
    condi_name = ['GOc', 'GOu', 'NoGo']
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
                report_cmplx_abs = mne.Report()
            
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
                
                if waveType == 'multitaper' and numType == 'real':  # use multitaper and set average = False
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
                elif waveType == 'morlet' and numType == 'real': 
                    power, itc = tfr_morlet(
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
                    ## use morlet for complex pwr but set average, itc = False, 
                    """ MNE: If "complex", then average must be False."""
                elif waveType == 'morlet' and numType == 'complex': 
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
                    # added on 29/04/2024: taking abs(.) value to match multitaper
                    pwrComplex_abs = abs(pwrComplex.copy())
    #%%        
                
                
                # don't forget %matplotlib qt
                if numType == 'real':
                                      
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
                
                elif numType == 'complex':
                    for icmplx in range(2): # doing twice for saving abs(.) and original separately
                        plt.rcParams.update({'font.size': 14})
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
                        fig.suptitle('TFR complex power')
                        
                        # added 29/04/2024: taking abs of complex at single trial level
                        if icmplx == 0:
                            avgTFR = pwrComplex.copy().average(method='mean', dim='epochs')
                        else:
                            avgTFR = pwrComplex_abs.copy().average(method='mean', dim='epochs')
                        
                        bslncorr = avgTFR.plot(baseline = (-0.5, -0.1), mode="logratio", title = 'auto',
                                    exclude = 'bads', combine = 'mean', axes = ax1)
                        ax1.set_title('bsln corr')
                        bslnNocorr = avgTFR.plot(baseline=None, title = 'auto',
                                    exclude = 'bads', combine = 'mean', axes = ax2)
                        ax2.set_title('no bsln corr')
                        # ax1.set_title('Power')
                        fig.tight_layout()
                        if icmplx == 0:
                            report_cmplx.add_figure(fig, title = subject + '_tfrplt', replace = True)
                        else:
                            report_cmplx_abs.add_figure(fig, title = subject + '_tfrplt', replace = True)
                            
                    

                plt.close('all')
                
                #%%   saving the tfr data in numpy array baseline NOT corrected
                
                if numType == 'real':
                    print('\n Writing the TFR_power to disk: no bsnlneCorr')
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR-tfr.h5'
                    tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                            config_for_gogait.base_fname_no_fif.format(**locals()))
                   
                    # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                    print("Output: ", tfr_fname)
                    mne.time_frequency.write_tfrs(tfr_fname, power, overwrite=True)    
                    
                    print('\n Writing the TFR_itc to disk: no bsnlneCorr')
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_ITC-tfr.h5'
                    tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                            config_for_gogait.base_fname_no_fif.format(**locals()))
                   
                    # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                    print("Output: ", tfr_fname)
                    mne.time_frequency.write_tfrs(tfr_fname, itc, overwrite=True)  
                    
                    # itc_read = mne.time_frequency.read_tfrs(tfr_fname, condition=None)
                    # itc_read[0].plot(baseline=(-0.2, -0.1), mode="logratio", title = 'auto',
                    #            exclude = 'bads', combine = 'mean')
                    
                    #%   saving the tfr data in numpy array baseline corrected
                    print('\n Writing the TFR_power to disk: bslnCorr')
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR_bslnCorr-tfr.h5'
                    tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                            config_for_gogait.base_fname_no_fif.format(**locals()))
                   
                    # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                    print("Output: ", tfr_fname)
                    power_bslnCorr = power.apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
                    mne.time_frequency.write_tfrs(tfr_fname, power_bslnCorr, overwrite=True)    
                    
                    print('\n Writing the TFR_itc to disk: bslnCorr')
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_ITC_bslnCorr-tfr.h5'
                    tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                            config_for_gogait.base_fname_no_fif.format(**locals()))
                   
                    # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                    print("Output: ", tfr_fname)
                    itc_bslnCorr = itc.apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
                    mne.time_frequency.write_tfrs(tfr_fname, itc_bslnCorr, overwrite=True)  
                
                elif numType == 'complex':
                
                    #%   saving the complex epochsTFR: no baseline correction applied 
                    print('\n Writing the complex epochsTFR to disk')
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR_CMPLX-tfr.h5'
                    tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                            config_for_gogait.base_fname_no_fif.format(**locals()))
                   
                    print("Output: ", tfr_fname)
                    mne.time_frequency.write_tfrs(tfr_fname, pwrComplex, overwrite=True) 
                    
                    ## saving abs: added on 29/04/2024
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR_CMPLX_ABS-tfr.h5'
                    tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                            config_for_gogait.base_fname_no_fif.format(**locals()))
                   
                    print("Output: ", tfr_fname)
                    mne.time_frequency.write_tfrs(tfr_fname, pwrComplex_abs, overwrite=True)  
                    
                    #%  saving the complex epochsTFR: baseline correction applied 
                    print('\n Writing the complex epochsTFR to disk baseline corrected')
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR_CMPLX_bslnCorr-tfr.h5'
                    tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                            config_for_gogait.base_fname_no_fif.format(**locals()))
                   
                    
                    print("Output: ", tfr_fname)
                    pwrComplex_bslnCorr = pwrComplex.copy().apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
                    mne.time_frequency.write_tfrs(tfr_fname, pwrComplex_bslnCorr, overwrite=True)
                    
                    ## saving abs: added on 29/04/2024
        
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR_CMPLX_ABS_bslnCorr-tfr.h5'
                    tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                            config_for_gogait.base_fname_no_fif.format(**locals()))
                   
                    
                    print("Output: ", tfr_fname)
                    pwrComplex_abs_bslnCorr = pwrComplex_abs.copy().apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
                    mne.time_frequency.write_tfrs(tfr_fname, pwrComplex_abs_bslnCorr, overwrite=True) 
                    
                #%%   saving the tfr times in numpy array
                # print('\n Writing the TFR_times to disk')
                # extension =  'TFR_times_'+ event_type[0] +'.npy'
                # tfr_fname = op.join(config_for_gogait.eeg_dir_GOODremove,
                #                         config_for_gogait.base_fname_generic.format(**locals()))
                # tfr_times = itc._raw_times
                # print("Output: ", tfr_fname)
                # np.save(tfr_fname, tfr_times) 
                
                
                
            if numType == 'real':
                # finally saving the report after the for subject loop ends.     
                print('\n Saving the reports to disk')  
                report_bslnCorr.title = 'Single sub TF baseline corr: ' + condi + '_' + evnt+ '_' + version + '_' + waveType + '_' + numType
                #report.title = 'Group sub STC contrast at ' + evnt
                extension = 'single_sub_TF_bslnCorr'
                report_bslnCorr_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
                report_bslnCorr.save(report_bslnCorr_fname+'_' + condi +'_'+ evnt + '_' + version+ '_' + waveType + '_' + numType + '.html', overwrite=True)            
                
                # finally saving the report after the for subject loop ends.     
                print('\n Saving the reports to disk')  
                report_no_bslnCorr.title = 'Single sub TF No baseline corr: ' + condi + '_' + evnt+ '_' + version + '_' + waveType + '_' + numType
                #report.title = 'Group sub STC contrast at ' + evnt
                extension = 'single_sub_TF_NO_bslnCorr'
                report_no_bslnCorr_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
                report_no_bslnCorr.save(report_no_bslnCorr_fname+'_' + condi +'_'+ evnt + '_' + version+ '_' + waveType + '_' + numType + '.html', overwrite=True)            
            
            elif numType == 'complex':
                # finally saving the report after the for subject loop ends.     
                print('\n Saving the reports to disk')  
                report_cmplx.title = 'Single sub TF: ' + condi + '_' + evnt+ '_' + version + '_' + waveType + '_' + numType
                #report.title = 'Group sub STC contrast at ' + evnt
                extension = 'single_sub_TF'
                report_cmplx_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
                report_cmplx.save(report_cmplx_fname+'_' + condi +'_'+ evnt + '_' + version+ '_' + waveType + '_' + numType + '.html', overwrite=True)            
                
                # finally saving the report after the for subject loop ends.     
                print('\n Saving the reports to disk')  
                report_cmplx_abs.title = 'Single sub TF: ' + condi + '_' + evnt+ '_' + version + '_' + waveType + '_' + numType + '_abs(.)' 
                #report.title = 'Group sub STC contrast at ' + evnt
                extension = 'single_sub_TF'
                report_cmplx_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
                report_cmplx_abs.save(report_cmplx_fname+'_' + condi +'_'+ evnt + '_' + version+ '_' + waveType + '_' + numType + '_abs.html', overwrite=True)            
        

    