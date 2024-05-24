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

event_type = ['cue']
baseline = 'bslnNoCorr' # 'bslnCorr' | 'bslnNoCorr'
# event_type = ['cue', 'target']
# version_list = ['GOODremove','CHANremove']
version_list = ['CHANremove']
ep_extension = 'TF'

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
   
    sampling_freq = 500 # in hz
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
                
                #% importing TF files of each sub from disk
                
                if baseline == 'bslnCorr':
                    print('Importing the TFR_power_bslnCorr from disk')
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_PWR_bslnCorr-tfr.h5'
                elif baseline == 'bslnNoCorr':
                    print('Importing the TFR_power from disk')
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_PWR-tfr.h5'
                    
                tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                        config_for_gogait.base_fname_no_fif.format(**locals()))
               
                print("Output: ", tfr_fname)
                pwr_read = mne.time_frequency.read_tfrs(tfr_fname)
                
                pwr_per_sub = pwr_read[0].data
                
                print('Importing the TFR_itc from disk')
                "NB. Loading is long for first time alone"
                
                if baseline == 'bslnCorr':
                    print('Importing the TFR_itc_bslnCorr from disk')
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_ITC_bslnCorr-tfr.h5'
                elif baseline == 'bslnNoCorr':
                    print('Importing the TFR_itc from disk')
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_ITC-tfr.h5'
                
                
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
        
            # store condi in dim1, chs in dim2, freq in dim 3, time in dim4 
            tf_pwr_array_avg_sub_exp_dim =  np.expand_dims(tf_pwr_array_avg_sub, axis = 0) 
            tf_itc_array_avg_sub_exp_dim =  np.expand_dims(tf_itc_array_avg_sub, axis = 0) 
            
            if ci == 0:
                tf_pwr_array_avg_sub_all_condi = tf_pwr_array_avg_sub_exp_dim 
                tf_itc_array_avg_sub_all_condi = tf_itc_array_avg_sub_exp_dim
            else:
                tf_pwr_array_avg_sub_all_condi = np.vstack((tf_pwr_array_avg_sub_all_condi, tf_pwr_array_avg_sub_exp_dim))                     
                tf_itc_array_avg_sub_all_condi = np.vstack((tf_itc_array_avg_sub_all_condi, tf_itc_array_avg_sub_exp_dim)) 
            
            
            
            
        #%% put them in the respective MNE containers
        report = mne.Report()
        
        print('Importing the TFR_times to disk')
        extension =  'TFR_times_'+ event_type[0] +'.npy'
        tfr_fname = op.join(config_for_gogait.eeg_dir_GOODremove,
                                config_for_gogait.base_fname_generic.format(**locals()))
     
        print("Input: ", tfr_fname)
        tfr_times = np.load(tfr_fname)  
        
        pwr_array_GOc = tf_itc_array_avg_sub_all_condi[0,:,:,:]
        power_GOc = mne.time_frequency.AverageTFR(info = info, 
                                              data = pwr_array_GOc,
                                              times = tfr_times, 
                                              freqs = tfr_freqs, 
                                              nave = 1, 
                                              comment = 'avgPower' ,
                                              method = 'multitaper')
        
        pwr_array_GOu = tf_itc_array_avg_sub_all_condi[1,:,:,:]
        power_GOu = mne.time_frequency.AverageTFR(info = info, 
                                              data = pwr_array_GOu, 
                                              times = tfr_times, 
                                              freqs = tfr_freqs, 
                                              nave = 1, 
                                              comment = 'avgPower' ,
                                              method = 'multitaper')
        
        pwr_array_NoGo = tf_itc_array_avg_sub_all_condi[2,:,:,:]
        power_NoGo = mne.time_frequency.AverageTFR(info = info, 
                                              data = pwr_array_NoGo, 
                                              times = tfr_times, 
                                              freqs = tfr_freqs, 
                                              nave = 1, 
                                              comment = 'avgPower' ,
                                              method = 'multitaper')
        freqMin = [3.0, 7.0, 13.0]
        freqMax = [7.0, 12.0, 21.0]
        fband = ['theta', 'alpha', 'beta']
        
        for f, fn in enumerate(fband):   
            tmin = [-0.5, 0.0, 0.4] # in s
            tmax = [-0.3, 0.2, 0.6] # in s
            tmin_ind = np.ones(np.shape(tmin), dtype=int) * 99
            tmax_ind = np.ones(np.shape(tmin), dtype=int) * 99
            
            for i,ind in enumerate(tmin): 
                tmin_ind[i] = int(np.where(tfr_times == find_closest(tfr_times, ind))[0][0])
            
            for i,ind in enumerate(tmax): 
                tmax_ind[i] = int(np.where(tfr_times == find_closest(tfr_times, ind))[0][0])
                
            fmin = freqMin[f]
            fmax = freqMax[f]
            
            tfr_freqs_round = np.round(tfr_freqs)
            fmin_ind = np.where(tfr_freqs_round == find_closest(tfr_freqs_round, fmin))[0][0]
            fmax_ind = np.where(tfr_freqs_round == find_closest(tfr_freqs_round, fmax))[0][0]
            
            pwr_GOc_max_alpha = pwr_array_GOc[:, fmin_ind : fmax_ind,:].mean(axis = 1)
            pwr_GOc_max_alpha_t1 = pwr_GOc_max_alpha[:,tmin_ind[0]:tmax_ind[0]].mean(axis = 1)
            pwr_GOc_max_alpha_t2 = pwr_GOc_max_alpha[:,tmin_ind[1]:tmax_ind[1]].mean(axis = 1)
            pwr_GOc_max_alpha_t3 = pwr_GOc_max_alpha[:,tmin_ind[2]:tmax_ind[2]].mean(axis = 1)
            
            vabsmax_GOc = np.max([np.max(np.abs(pwr_GOc_max_alpha_t1)),
                              np.max(np.abs(pwr_GOc_max_alpha_t2)), 
                              np.max(np.abs(pwr_GOc_max_alpha_t3))])
            vmax_GOc = + vabsmax_GOc
            vmin_GOc = - vabsmax_GOc
            
            
            pwr_GOu_max_alpha = pwr_array_GOu[:, fmin_ind : fmax_ind,:].mean(axis = 1)
            pwr_GOu_max_alpha_t1 = pwr_GOu_max_alpha[:,tmin_ind[0]:tmax_ind[0]].mean(axis = 1)
            pwr_GOu_max_alpha_t2 = pwr_GOu_max_alpha[:,tmin_ind[1]:tmax_ind[1]].mean(axis = 1)
            pwr_GOu_max_alpha_t3 = pwr_GOu_max_alpha[:,tmin_ind[2]:tmax_ind[2]].mean(axis = 1)
            
            vabsmax_GOu = np.max([np.max(np.abs(pwr_GOu_max_alpha_t1)),
                              np.max(np.abs(pwr_GOu_max_alpha_t2)), 
                              np.max(np.abs(pwr_GOu_max_alpha_t3))])
            vmax_GOu = + vabsmax_GOu
            vmin_GOu = - vabsmax_GOu
            
            pwr_NoGo_max_alpha = pwr_array_NoGo[:, fmin_ind : fmax_ind,:].mean(axis = 1)
            pwr_NoGo_max_alpha_t1 = pwr_NoGo_max_alpha[:,tmin_ind[0]:tmax_ind[0]].mean(axis = 1)
            pwr_NoGo_max_alpha_t2 = pwr_NoGo_max_alpha[:,tmin_ind[1]:tmax_ind[1]].mean(axis = 1)
            pwr_NoGo_max_alpha_t3 = pwr_NoGo_max_alpha[:,tmin_ind[2]:tmax_ind[2]].mean(axis = 1)
            
            vabsmax_NoGo = np.max([np.max(np.abs(pwr_NoGo_max_alpha_t1)),
                              np.max(np.abs(pwr_NoGo_max_alpha_t2)), 
                              np.max(np.abs(pwr_NoGo_max_alpha_t3))])
            vmax_NoGo = + vabsmax_NoGo
            vmin_NoGo = - vabsmax_NoGo
                
            ## don't forget to run, %matplotlib qt, else topos wont be plotted
            fig, axs = plt.subplots(3,3, figsize=(9,6), sharex=True, sharey=True)
            for rows in range(3):
                          
                for cols in range(3): 
                    
                    if rows == 0:
                        vmin = vmin_GOc
                        vmax = vmax_GOc
                        im = power_GOc.plot_topomap(tmin = tmin[cols], tmax = tmax[cols], fmin = fmin, fmax = fmax,
                                           sphere = 0.45, axes = axs[rows, cols], vlim=(vmin, vmax), cmap = 'RdBu_r', colorbar= True)
                        if cols == 0:
                            axs[rows, cols].set_ylabel("GOc")
                        
                        axs[rows, cols].set_title(str(tmin[cols]) + ' to ' + str(tmax[cols]) + ' s')
                            
                    elif rows == 1:
                        vmin = vmin_GOu
                        vmax = vmax_GOu
                        im = power_GOu.plot_topomap(tmin = tmin[cols], tmax = tmax[cols], fmin = fmin, fmax = fmax,
                                           sphere = 0.45, axes = axs[rows, cols], vlim=(vmin, vmax),  cmap = 'RdBu_r', colorbar= True)
                        if cols == 0:
                            axs[rows, cols].set_ylabel("GOu")
                            
                    elif rows == 2:
                        vmin = vmin_NoGo
                        vmax = vmax_NoGo
                        im = power_NoGo.plot_topomap(tmin = tmin[cols], tmax = tmax[cols], fmin = fmin, fmax = fmax,
                                           sphere = 0.45, axes = axs[rows, cols],vlim=(vmin, vmax), cmap = 'RdBu_r', colorbar= True)
                        if cols == 0:
                            axs[rows, cols].set_ylabel("NoGo")
                    
            fig.tight_layout()
            report.add_figure(fig, title = fn, replace = True)
        
        # plt.savefig('evktopo_time_avg.png', dpi=300)
        # plt.close('all')
            
            
           
           
    #%% finally saving the report after the for subject loop ends.     
    print('Saving the reports to disk')  
    report.title = 'Group sub TF topomaps : ' + evnt + '_' + baseline + '_' + version
    #report.title = 'Group sub STC contrast at ' + evnt
    extension = 'group_sub_TF_topomaps'
    report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
    report.save(report_fname+'_' + evnt + '_' + baseline + '_' + version+ '.html', overwrite=True)            
              
   
   
 # # plot the avg TFRs 
 # # don't forget %matplotlib qt
 # plt.rcParams.update({'font.size': 14})
 # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
 # fig.suptitle('TFR power and ITC')
 # power.plot(baseline=(-0.5, -0.1), mode="logratio", title = 'auto',
 #            exclude = 'bads', combine = 'mean', axes = ax1)
 # ax1.set_title('Power')
 # plt.rcParams.update({'font.size': 14})
 # itcplot = itc.plot(baseline=(-0.5, -0.1), mode="logratio", title = 'auto',
 #            exclude = 'bads', combine = 'mean', axes = ax2)
 # ax2.set_title('ITC')
 
 # # if event_type[0] == 'cue':
 # #     ax1.set_xlim([-0.2, 1.0])
 # #     ax2.set_xlim([-0.2, 1.0])
 # # elif event_type[0] == 'target':
 # #     ax1.set_xlim([-0.2, 0.7])
 # #     ax2.set_xlim([-0.2, 0.7])
     
 # fig.tight_layout()
 # report.add_figure(fig, title = condi + '_tfrplt_bslnCorr', replace = True)
 
 # # no baseline corrected
 
 # plt.rcParams.update({'font.size': 14})
 # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
 # fig.suptitle('TFR power and ITC')
 # power.plot(baseline = None, title = 'auto',
 #            exclude = 'bads', combine = 'mean', axes = ax1)
 # ax1.set_title('Power')
 # plt.rcParams.update({'font.size': 14})
 # itcplot = itc.plot(baseline = None, title = 'auto',
 #            exclude = 'bads', combine = 'mean', axes = ax2)
 # ax2.set_title('ITC')
 
 # # if event_type[0] == 'cue':
 # #     ax1.set_xlim([-0.2, 1.0])
 # #     ax2.set_xlim([-0.2, 1.0])
 # # elif event_type[0] == 'target':
 # #     ax1.set_xlim([-0.2, 0.7])
 # #     ax2.set_xlim([-0.2, 0.7])
 
 # fig.tight_layout()
 # report.add_figure(fig, title = condi + '_tfrplt', replace = True)
 
      


    