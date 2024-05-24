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
## note: always take baseline corr version, it allows to do common scaling before plotting
version_list = ['CHANremove']
ep_extension = 'TF'
waveType = 'morlet' # 'morlet' | 'multitaper'
numType = 'complex' # 'real' | 'complex'

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
                
                print('Importing the TFR_itc from disk')
                "NB. Loading is long for first time alone"
                
                ### commenting the lines for ITC computation
                
                # if baseline == 'bslnCorr':
                #     print('Importing the TFR_itc_bslnCorr from disk')
                #     extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_ITC_bslnCorr-tfr.h5'
                # elif baseline == 'bslnNoCorr':
                #     print('Importing the TFR_itc from disk')
                #     extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_ITC-tfr.h5'
                
                
                # tfr_fname = op.join(eeg_subject_dir_GOODremove,
                #                         config_for_gogait.base_fname_no_fif.format(**locals()))
               
                # print("Output: ", tfr_fname)
                # itc_read = mne.time_frequency.read_tfrs(tfr_fname, condition=None)
                # itc_per_sub = itc_read[0].data 
                
                # store sub in dim1, chs in dim2, freq in dim 3, time in dim4 
                pwr_per_sub_exp_dim = np.expand_dims(pwr_per_sub, axis = 0) 
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
        report = mne.Report()
        scale = ['timeScale', 'globalScale']
        fiscl = 'timeScale' # | 'ziri' | 'timeScale'| 'globalScale'
        numTypeTopo = ['real','imag','complex', 'complexModulo']         
   
        # numTypeTopo = ['imag','complex']
        print('Importing the TFR_times to disk')
        extension =  'TFR_times_'+ event_type[0] +'.npy'
        tfr_fname = op.join(config_for_gogait.eeg_dir_GOODremove,
                                config_for_gogait.base_fname_generic.format(**locals()))
     
        print("Input: ", tfr_fname)
        tfr_times = np.load(tfr_fname)  
        
        for ntype in numTypeTopo: 
            if ntype == 'real': 
                pwr_data = np.real(tfr_all)
            elif ntype == 'imag':
                pwr_data = np.imag(tfr_all)
            elif ntype == 'complex':
                pwr_data = tfr_all.copy() ## be careful of bidirectional assignment
            elif ntype == 'complexModulo':
                pwr_data = np.abs(tfr_all)
                
        
            pwr_array_GOc = pwr_data[0,:,:,:]
            power_GOc = mne.time_frequency.AverageTFR(info = info, 
                                                  data = pwr_array_GOc,
                                                  times = tfr_times, 
                                                  freqs = tfr_freqs, 
                                                  nave = 1, 
                                                  comment = 'avgPower' ,
                                                  method = 'morlet')
            
            pwr_array_GOu = pwr_data[1,:,:,:]
            power_GOu = mne.time_frequency.AverageTFR(info = info, 
                                                  data = pwr_array_GOu, 
                                                  times = tfr_times, 
                                                  freqs = tfr_freqs, 
                                                  nave = 1, 
                                                  comment = 'avgPower' ,
                                                  method = 'morlet')
            
            pwr_array_NoGo = pwr_data[2,:,:,:]
            power_NoGo = mne.time_frequency.AverageTFR(info = info, 
                                                  data = pwr_array_NoGo, 
                                                  times = tfr_times, 
                                                  freqs = tfr_freqs, 
                                                  nave = 1, 
                                                  comment = 'avgPower' ,
                                                  method = 'morlet')
            freqMin = [3.0, 7.0, 13.0]
            freqMax = [7.0, 12.0, 21.0]
            fband = ['theta', 'alpha', 'beta']
            
            ### apply baseline corr only if imported data has no baseline corr
            # if baseline == 'bslnCorr':
            #     corr = 'bslnUncorr'
            # elif baseline == 'bslnNoCorr':
            # corr = 'bslnNoCorr'
            # power_GOc_bslcorr = power_GOc.copy()
            # power_GOu_bslcorr = power_GOu.copy()
            # power_NoGo_bslcorr = power_NoGo.copy()
            corr = 'bslnCorr'
            power_GOc_bslcorr = power_GOc.copy().apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
            power_GOu_bslcorr = power_GOu.copy().apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
            power_NoGo_bslcorr = power_NoGo.copy().apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
            #%
            for f, fn in enumerate(fband): 
                if fn == 'theta' or fn == 'alpha': 
                    tmin = [0.0, 0.050, 0.150, 0.350, 0.400] # in s
                    tmax = [0.000, 0.050, 0.150, 0.350, 0.400] # in s
                    tmin_ms = [0, 50, 150, 350, 400] # in ms
                    
                    # tmin = [0.0, 0.200, 0.200] # in s
                    # tmax = [0.200, 0.400, 0.450] # in s
                    # tmin_ms = [0, 200, 200] # in ms
                    # tmax_ms = [200, 400, 450] # in ms
                   
                    
                elif fn == 'beta':
                    tmin = [0.250, 0.350, 0.400, 0.450, 0.500] # in s
                    tmax = [0.250, 0.350, 0.400, 0.450, 0.500]
                    tmin_ms = [250, 350, 400, 450, 500] # in ms
                    # tmin = [0.0, 0.200, 0.200] # in s
                    # tmax = [0.200, 0.400, 0.450] # in s
                    # tmin_ms = [0, 200, 200] # in ms
                    # tmax_ms = [200, 400, 450] # in ms
                    
                    
                nT = len(tmin)
                tmin_ind = np.empty(np.shape(tmin), dtype=int) #np.ones(np.shape(tmin), dtype=int) * 99
                tmax_ind = np.empty(np.shape(tmin), dtype=int) #np.ones(np.shape(tmin), dtype=int) * 99
                
                for i,ind in enumerate(tmin): 
                    tmin_ind[i] = int(np.where(tfr_times == find_closest(tfr_times, ind))[0][0])
                
                for i,ind in enumerate(tmax): 
                    tmax_ind[i] = int(np.where(tfr_times == find_closest(tfr_times, ind))[0][0])
                    
                fmin = freqMin[f]
                fmax = freqMax[f]
                
                tfr_freqs_round = np.round(tfr_freqs)
                fmin_ind = np.where(tfr_freqs_round == find_closest(tfr_freqs_round, fmin))[0][0]
                fmax_ind = np.where(tfr_freqs_round == find_closest(tfr_freqs_round, fmax))[0][0]
                
                pwr_GOc_max_freq = pwr_array_GOc[:, fmin_ind : fmax_ind,:].mean(axis = 1)
                pwr_GOc_max_freq_t1 = pwr_GOc_max_freq[:,tmin_ind[0]] #:tmax_ind[0]]
                pwr_GOc_max_freq_t2 = pwr_GOc_max_freq[:,tmin_ind[1]] #:tmax_ind[1]]
                pwr_GOc_max_freq_t3 = pwr_GOc_max_freq[:,tmin_ind[2]] #:tmax_ind[2]]
                pwr_GOc_max_freq_t4 = pwr_GOc_max_freq[:,tmin_ind[3]] #:tmax_ind[3]]
                pwr_GOc_max_freq_t5 = pwr_GOc_max_freq[:,tmin_ind[4]] #:tmax_ind[4]]
                
                # pwr_GOc_max_freq_t1 = pwr_GOc_max_freq[:,tmin_ind[0]:tmax_ind[0]].mean(axis = 1)
                # pwr_GOc_max_freq_t2 = pwr_GOc_max_freq[:,tmin_ind[1]:tmax_ind[1]].mean(axis = 1)
                # pwr_GOc_max_freq_t3 = pwr_GOc_max_freq[:,tmin_ind[2]:tmax_ind[2]].mean(axis = 1)
                
                vabsmax_GOc = np.max([np.max(np.abs(pwr_GOc_max_freq_t1)),
                                  np.max(np.abs(pwr_GOc_max_freq_t2)), 
                                  np.max(np.abs(pwr_GOc_max_freq_t3)), 
                                   np.max(np.abs(pwr_GOc_max_freq_t4)), 
                                   np.max(np.abs(pwr_GOc_max_freq_t5))
                                  ])
                
                vmax_GOc = + vabsmax_GOc
                vmin_GOc = - vabsmax_GOc
                
                
                pwr_GOu_max_freq = pwr_array_GOu[:, fmin_ind : fmax_ind,:].mean(axis = 1)
                pwr_GOu_max_freq_t1 = pwr_GOu_max_freq[:,tmin_ind[0]]
                pwr_GOu_max_freq_t2 = pwr_GOu_max_freq[:,tmin_ind[1]]
                pwr_GOu_max_freq_t3 = pwr_GOu_max_freq[:,tmin_ind[2]]
                pwr_GOu_max_freq_t4 = pwr_GOu_max_freq[:,tmin_ind[3]]
                pwr_GOu_max_freq_t5 = pwr_GOu_max_freq[:,tmin_ind[4]]
                
                # pwr_GOu_max_freq_t1 = pwr_GOu_max_freq[:,tmin_ind[0]:tmax_ind[0]].mean(axis = 1)
                # pwr_GOu_max_freq_t2 = pwr_GOu_max_freq[:,tmin_ind[1]:tmax_ind[1]].mean(axis = 1)
                # pwr_GOu_max_freq_t3 = pwr_GOu_max_freq[:,tmin_ind[2]:tmax_ind[2]].mean(axis = 1)
                
                vabsmax_GOu = np.max([np.max(np.abs(pwr_GOu_max_freq_t1)),
                                  np.max(np.abs(pwr_GOu_max_freq_t2)), 
                                  np.max(np.abs(pwr_GOu_max_freq_t3)),
                                   np.max(np.abs(pwr_GOu_max_freq_t4)), 
                                   np.max(np.abs(pwr_GOu_max_freq_t5))
                                  ])
               
                vmax_GOu = + vabsmax_GOu
                vmin_GOu = - vabsmax_GOu
                
                pwr_NoGo_max_freq = pwr_array_NoGo[:, fmin_ind : fmax_ind,:].mean(axis = 1)
                pwr_NoGo_max_freq_t1 = pwr_NoGo_max_freq[:,tmin_ind[0]]
                pwr_NoGo_max_freq_t2 = pwr_NoGo_max_freq[:,tmin_ind[1]]
                pwr_NoGo_max_freq_t3 = pwr_NoGo_max_freq[:,tmin_ind[2]]
                pwr_NoGo_max_freq_t4 = pwr_NoGo_max_freq[:,tmin_ind[3]]
                pwr_NoGo_max_freq_t5 = pwr_NoGo_max_freq[:,tmin_ind[4]]
                
                # pwr_NoGo_max_freq_t1 = pwr_NoGo_max_freq[:,tmin_ind[0]:tmax_ind[0]].mean(axis = 1)
                # pwr_NoGo_max_freq_t2 = pwr_NoGo_max_freq[:,tmin_ind[1]:tmax_ind[1]].mean(axis = 1)
                # pwr_NoGo_max_freq_t3 = pwr_NoGo_max_freq[:,tmin_ind[2]:tmax_ind[2]].mean(axis = 1)
                
                vabsmax_NoGo = np.max([np.max(np.abs(pwr_NoGo_max_freq_t1)),
                                  np.max(np.abs(pwr_NoGo_max_freq_t2)), 
                                  np.max(np.abs(pwr_NoGo_max_freq_t3)), 
                                   np.max(np.abs(pwr_NoGo_max_freq_t4)), 
                                   np.max(np.abs(pwr_NoGo_max_freq_t5))
                                  ])
               
                vmax_NoGo = + vabsmax_NoGo
                vmin_NoGo = - vabsmax_NoGo
                
                vmax_all_condi = np.max([vabsmax_GOc, vabsmax_GOu , vabsmax_NoGo])
                
                vminGlobal = -vmax_all_condi 
                vmaxGlobal = vmax_all_condi
                
                
                ## uncomment these lines to replicate # Ziri et al submitted
                # if fn == 'theta' or fn == 'alpha': 
                #     vmin = -4e-1
                #     vmax = 4.0e-1
                # elif fn == 'beta':
                #     vmin = -2e-1
                #     vmax = 2e-1
                    
                ## don't forget to run, %matplotlib qt, else topos wont be plotted
                fig, axs = plt.subplots(ncondi,nT, figsize=(9,4), sharex=True, sharey=True)
                for rows in range(ncondi):                          
                    for cols in range(nT): 
                        ## for column wise scale
                        vabsmax = np.max([np.max(np.abs(pwr_GOc_max_freq[:,tmin_ind[cols]])),
                                          np.max(np.abs(pwr_GOu_max_freq[:,tmin_ind[cols]])), 
                                          np.max(np.abs(pwr_NoGo_max_freq[:,tmin_ind[cols]]))])
                        # vabsmax = np.max([np.max(np.abs(pwr_GOc_max_freq[:,tmin_ind[cols]:tmax_ind[cols]].mean(axis = 1))),
                        #                   np.max(np.abs(pwr_GOu_max_freq[:,tmin_ind[cols]:tmax_ind[cols]].mean(axis = 1))), 
                        #                   np.max(np.abs(pwr_NoGo_max_freq[:,tmin_ind[cols]:tmax_ind[cols]].mean(axis = 1)))])
                        
                        vmaxTime = + vabsmax
                        vminTime = - vabsmax
                        if fiscl == 'timeScale':
                            vmin = vminTime
                            vmax = vmaxTime
                        elif fiscl == 'globalScale':
                            vmin = vminGlobal
                            vmax = vmaxGlobal
                        elif fiscl == 'ziri':
                            if fn == 'theta' or fn == 'alpha': 
                                vmin = -5.0e-1
                                vmax = 5.0e-1
                            elif fn == 'beta':
                                vmin = -2.0e-1
                                vmax = 2.0e-1
                            
                            
                        if rows == 0:
                            # vmin = vmin_GOc
                            # vmax = vmax_GOc
                            im = power_GOc_bslcorr.plot_topomap(baseline = None,
                                                        tmin = tmin[cols], tmax = tmax[cols], 
                                                        fmin = fmin, fmax = fmax,
                                                        sphere = 0.45, axes = axs[rows, cols], 
                                                        vlim=(vmin, vmax), cmap = 'RdBu_r', colorbar= False)
                            if cols == 0:
                                axs[rows, cols].set_ylabel("GOc")
                            
                            # axs[rows, cols].set_title(str(tmin_ms[cols])+' - '+str(tmax_ms[cols]) +' ms')
                            axs[rows, cols].set_title(str(tmin_ms[cols]) +' ms')
                                
                        elif rows == 1:
                            # vmin = vmin_GOu
                            # vmax = vmax_GOu
                            im = power_GOu_bslcorr.plot_topomap(baseline = None,
                                                        tmin = tmin[cols], tmax = tmax[cols], 
                                                        fmin = fmin, fmax = fmax,
                                                        sphere = 0.45, axes = axs[rows, cols], 
                                                        vlim=(vmin, vmax),  cmap = 'RdBu_r', colorbar= False)
                            if cols == 0:
                                axs[rows, cols].set_ylabel("GOu")
                            
                            axs[rows, cols].set_title("")
                                
                        elif rows == 2:
                            # vmin = vmin_NoGo
                            # vmax = vmax_NoGo
                            im = power_NoGo_bslcorr.plot_topomap(baseline = None,
                                                        tmin = tmin[cols], tmax = tmax[cols], 
                                                        fmin = fmin, fmax = fmax,
                                                        sphere = 0.45, axes = axs[rows, cols],
                                                        vlim=(vmin, vmax), cmap = 'RdBu_r', colorbar= False)
                            if cols == 0:
                                axs[rows, cols].set_ylabel("NoGo")
                            
                            axs[rows, cols].set_title("")
                            
                            ## colorbar of topo
                            if fiscl == 'timeScale' or cols == nT-1:
                                ## create additional axes (for ERF and colorbar)
                                divider = make_axes_locatable(axs[rows, cols])
                                # add axes for colorbar
                                ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
                                image = axs[rows, cols].images[0]
                                plt.colorbar(image, cax=ax_colorbar) # label =  r"$\mu$" + 'V')
                                ax_colorbar.set_title(r"$dB$", fontsize = '8')
                        
                fig.tight_layout()
                report.add_figure(fig, title = fn + '_' + ntype + '_' + corr, replace = True)
            
            del pwr_data
            del pwr_array_GOc
            del pwr_array_GOu
            del pwr_array_NoGo
            del power_GOc
            del power_GOu
            del power_NoGo
            del power_GOc_bslcorr
            del power_GOu_bslcorr
            del power_NoGo_bslcorr
            
            # plt.savefig('evktopo_time_avg.png', dpi=300)
            plt.close('all')
#%           
           
           
    #% finally saving the report after the for subject loop ends.     
    print('Saving the reports to disk')  
    report.title = 'Group sub TF topomaps : ' + evnt + '_' + baseline + '_data_' + '_' + corr + '_topo_' + version + '_' + waveType + '_' + numType
    #report.title = 'Group sub STC contrast at ' + evnt
    extension = 'group_sub_TF_topomaps'
    report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
    report.save(report_fname+'_' + evnt + '_' + baseline +'_data_'+ '_' + corr + '_topo_'+ version + '_' + waveType + '_' + numType +'_all.html', overwrite=True)            
              
   
   
 
      


    