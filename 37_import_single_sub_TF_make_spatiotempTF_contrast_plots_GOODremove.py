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
import scipy.stats
from scipy.linalg import norm
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse 
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid, inset_locator
from mne.stats import spatio_temporal_cluster_test, summarize_clusters_stc
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_test
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_1samp_test
from mne.stats import spatio_temporal_cluster_1samp_test
from scipy.sparse import diags
from mne.stats import combine_adjacency

import config_for_gogait
n_subs = len(config_for_gogait.subjects_list)
fs_dir = fetch_fsaverage(verbose=True)
template_subjects_dir = op.dirname(fs_dir)

# ampliNormalization = ['AmpliNorm', 'AmpliActual']
# ampliNormalization = 'AmpliActual'

event_type = ['target']  # 'cue'|'target'
baseline = 'bslnCorr' # 'bslnCorr' | 'bslnNoCorr'

# version_list = ['GOODremove','CHANremove']
version_list = ['CHANremove']
ep_extension = 'TF'


for ei, evnt in enumerate(event_type):
    
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['NoGo'] 
    condi_name = ['GOc', 'GOu', 'NoGo']
    
    if version_list[0] == 'GOODremove':
        n_chs = 128
    elif version_list[0] == 'CHANremove':
        n_chs = 103
   
    
    sampling_freq = 500 # in hz
    tfr_freqs = np.linspace(3,40,num = 40, endpoint= True)
    n_TF_freqs = len(tfr_freqs)
    
    for veri, version in enumerate(version_list):
        
        
        for sub_num, subject in enumerate(config_for_gogait.subjects_list): 
            print("\n Processing subject: %s" % subject)
            eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
       
            ## n_samples_esti = int(sampling_freq * (isi[ind_isi] + t_end - t_start + 0.002))  
            # estimate the num of time samples per condi/ISI to allocate numpy array
           
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
            # tf_itc_array_all_sub_all_condi = np.ones([n_subs, n_chs, n_TF_freqs, n_samples_esti])*np.nan
           
            for ci, condi in enumerate(condi_name): 
                
                print("condition: %s" % condi)  
                      
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
                
                "NB. Loading is longer for first time alone"
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
                
                # store condi in dim1, chs in dim2, freq in dim 3, time in dim4 
                pwr_per_sub_exp_dim = np.expand_dims(pwr_per_sub, axis = 0) 
                itc_per_sub_exp_dim = np.expand_dims(itc_per_sub, axis = 0) 
                
                if ci == 0:
                    tf_pwr_array_all_condi = pwr_per_sub_exp_dim
                    tf_itc_array_all_condi = itc_per_sub_exp_dim
                else:
                    tf_pwr_array_all_condi = np.vstack((tf_pwr_array_all_condi, pwr_per_sub_exp_dim))                     
                    tf_itc_array_all_condi = np.vstack((tf_itc_array_all_condi, itc_per_sub_exp_dim))                     
                    
            # # averaging TF arrays across all channels
            # tf_pwr_array_all_condi_avgCH = np.mean(tf_pwr_array_all_condi, axis = 1)
            # tf_itc_array_all_condi_avgCH = np.mean(tf_itc_array_all_condi, axis = 1)
            
            # store sub in dim1, condi in dim2, chs in dim2, freq in dim 4, time in dim5 
            tf_pwr_array_all_condi_exp_dim = np.expand_dims(tf_pwr_array_all_condi, axis = 0) 
            tf_itc_array_all_condi_exp_dim = np.expand_dims(tf_itc_array_all_condi, axis = 0) 
            
            if sub_num == 0:
                tf_pwr_array_all_condi_all_sub = tf_pwr_array_all_condi_exp_dim
                tf_itc_array_all_condi_all_sub = tf_itc_array_all_condi_exp_dim
            else:
                tf_pwr_array_all_condi_all_sub = np.vstack((tf_pwr_array_all_condi_all_sub, tf_pwr_array_all_condi_exp_dim))                     
                tf_itc_array_all_condi_all_sub = np.vstack((tf_itc_array_all_condi_all_sub, tf_itc_array_all_condi_exp_dim))                     
                
        #%% do the contrast
        ## https://mne.tools/stable/auto_tutorials/stats-sensor-space/50_cluster_between_time_freq.html#sphx-glr-auto-tutorials-stats-sensor-space-50-cluster-between-time-freq-py
        report = mne.Report()
        contrast_kind = ['GOu_GOc', 'NoGo_GOu']
        tf = 'pwr'  # ['pwr', 'itc']
        timeDur = 'poststim' # ['short', 'long', 'prestim', 'poststim']    
        threshType = 'auto'  # 'auto' |'tfce'| 'trshld' 
        
        
        for ci2, contrast in enumerate(contrast_kind):
            
            if contrast == 'GOu_GOc':
                
                X1_pwr = tf_pwr_array_all_condi_all_sub[:,1,:,:,:]
                X2_pwr = tf_pwr_array_all_condi_all_sub[:,0,:,:,:]
                Y1_itc = tf_itc_array_all_condi_all_sub[:,1,:,:,:]
                Y2_itc = tf_itc_array_all_condi_all_sub[:,0,:,:,:]
                
            
            elif contrast == 'NoGo_GOu': 
                X1_pwr = tf_pwr_array_all_condi_all_sub[:,2,:,:,:]
                X2_pwr = tf_pwr_array_all_condi_all_sub[:,1,:,:,:]
                Y1_itc = tf_itc_array_all_condi_all_sub[:,2,:,:,:]
                Y2_itc = tf_itc_array_all_condi_all_sub[:,1,:,:,:]
               
            
            #% take the diff b/w conditions and do spatio-temp test
            X_pwr = X1_pwr - X2_pwr
            Y_itc = Y1_itc - Y2_itc
            X_pwr_avg = X_pwr.mean(axis = 0) # avg across subs
            Y_itc_avg = Y_itc.mean(axis = 0)
            
            # transpose: observations × time × space) for evoked plots, 
            # or (observations × time × frequencies × space) for TF plots
            
            X = np.transpose(X_pwr, [0, 3, 2, 1])  # all subs
            Y = np.transpose(Y_itc, [0, 3, 2, 1])  # all subs
    
            
            print('Importing the TFR_times to disk')
            extension =  'TFR_times_'+ event_type[0] +'.npy'
            tfr_fname = op.join(config_for_gogait.eeg_dir_GOODremove,
                                    config_for_gogait.base_fname_generic.format(**locals()))
         
            print("Input: ", tfr_fname)
            tfr_times = np.load(tfr_fname)
            
            if timeDur == 'short' and  event_type[0] == 'cue':
                durStart = -0.2 # pre-stimulus(t2/S2/S4) duration in sec
                durEnd = 1.0 # post-stimulus (t3/S8/S16) duration in sec
            elif timeDur == 'short' and event_type[0] == 'target':
                durStart = -0.2 # pre-stimulus(t2/S2/S4) duration in sec
                durEnd = 0.7 # post-stimulus (t3/S8/S16) duration in sec
            elif timeDur == 'long' and  event_type[0] == 'cue':
                durStart = -0.8 # pre-stimulus(t2/S2/S4) duration in sec
                durEnd = 1.3 # post-stimulus (t3/S8/S16) duration in sec
            elif timeDur == 'long' and event_type[0] == 'target':
                durStart = -1.0 # pre-stimulus(t2/S2/S4) duration in sec
                durEnd = 0.7 # post-stimulus (t3/S8/S16) duration in sec
            elif timeDur == 'prestim' and  event_type[0] == 'cue':
                durStart = -0.5 # pre-stimulus(t2/S2/S4) duration in sec
                durEnd = 0.0 # post-stimulus (t3/S8/S16) duration in sec
            elif timeDur == 'poststim' and  event_type[0] == 'target':
                durStart = 0.0 # pre-stimulus(t2/S2/S4) duration in sec
                durEnd = 0.5 # post-stimulus (t3/S8/S16) duration in sec
            else:
                raise ValueError('check your timeDur and event_type combination')
           
            durStartInd = np.where(tfr_times == durStart)[0][0]
            durEndInd = np.where(tfr_times == durEnd)[0][0] + 1
            
            # recompute the arrays for windowed duration
            tfr_times = tfr_times[durStartInd:durEndInd]
            X = X[:,durStartInd:durEndInd,:,:] # time in dim2
            Y = Y[:,durStartInd:durEndInd,:,:]
            X_pwr_avg = X_pwr_avg[:,:,durStartInd:durEndInd] # time in dim3
            Y_itc_avg = Y_itc_avg[:,:,durStartInd:durEndInd]
            
            
            tfr_pwr_contrast_avg = mne.time_frequency.AverageTFR(info,X_pwr_avg,tfr_times,tfr_freqs, 1)
            tfr_itc_contrast_avg = mne.time_frequency.AverageTFR(info,Y_itc_avg,tfr_times,tfr_freqs, 1)
            
            #% setup sensor adjacency
           
            n_times, n_freqs, n_chans = (len(tfr_times), len(tfr_freqs), n_chs)
            chan_adj = diags([1., 1.], offsets=(-1, 1), shape=(n_chans, n_chans))
            adjM = combine_adjacency(
                n_times,  # regular lattice adjacency for times
                n_freqs,  # regular lattice adjacency for freqs
                chan_adj,  # custom matrix, or use mne.channels.find_ch_adjacency for MEG
            
                )  
            
            ## define threshold settings for forming significant cluster  
            if threshType == 'tfce':
                thresh = dict(start=0, step=1.0)
            elif threshType == 'auto':
                thresh = None
            elif threshType == 'trshld':
                # Here we set a cluster forming threshold based on a p-value for
                # the cluster based permutation test.
                # We use a two-tailed threshold, the "1 - p_threshold" is needed
                # because for two-tailed tests we must specify a positive threshold.
                p_threshold = 0.001 # 0.001
                df = n_subs - 1  # degrees of freedom for the test
                t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)
                thresh =  t_threshold
                
            
            # Now let's actually do the clustering. This can take a long time...
            print("Clustering." + tf)
            
            if tf == 'pwr':   
                data_for_stats = X
            elif tf == 'itc': 
                data_for_stats = Y
                
            T_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_1samp_test(
                data_for_stats,
                n_permutations = 100,
                adjacency=adjM,
                n_jobs=None,
                threshold = thresh,
                buffer_size=None,
                verbose=True,
            )
            
            #% Selecting “significant” clusters
            # Select the clusters that are statistically significant at p < 0.05
            good_clusters_idx = np.where(cluster_p_values < 0.05)[0]
            good_clusters = [clusters[idx] for idx in good_clusters_idx]
            
            # loop over clusters
            for i_clu, clu_idx in enumerate(good_clusters_idx):
                # unpack cluster information, get unique indices
                time_inds, freq_inds, space_inds = np.squeeze(clusters[clu_idx])
                ch_inds = np.unique(space_inds)
                time_inds = np.unique(time_inds)
                freq_inds = np.unique(freq_inds)
                pval = cluster_p_values[clu_idx]
                
                # get topography for F stat
                f_map_avg_time = T_obs[time_inds,:,:].mean(axis=0)
                f_map_avg_time_freq = f_map_avg_time[freq_inds,:].mean(axis=0)
                
                # get signals at the sensors contributing to the cluster
                sig_times = tfr_times[time_inds]
                sig_freqs = tfr_freqs[freq_inds]
                
                # create spatial mask
                mask = np.zeros((f_map_avg_time_freq.shape[0], 1), dtype=bool)
                mask[ch_inds, :] = True
                
                # initialize figure
                fig, ax = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
                
                # plot average test statistic and mark significant sensors
                f_evoked = mne.EvokedArray(f_map_avg_time_freq[:, np.newaxis], epochs.info, tmin=0)
                f_evoked.plot_topomap(
                    times=0,
                    mask=mask,
                    sphere = 0.4,
                    axes=ax[0],
                    cmap="Reds",
                    vlim=(np.min, np.max),
                    show=False,
                    colorbar=False,
                    mask_params=dict(markersize=10),
                )
                image = ax[0].images[0]
                
                # remove the title that would otherwise say "0.000 s"
                ax[0].set_title('p-value = ' + str(pval))
                
                # create additional axes (for ERF and colorbar)
                divider = make_axes_locatable(ax[0])
                
                # add axes for colorbar
                ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(image, cax=ax_colorbar)
                ax[0].set_xlabel(
                    "Averaged t-map ({:0.3f} - {:0.3f} s) \n ({:0.3f} - {:0.3f} Hz)".format(*sig_times[[0, -1]],
                                                                                            *sig_freqs[[0, -1]])
                )
                
                # add average TF power plot 
                # create TF mask 
                mask_TF = np.zeros((len(tfr_freqs),len(tfr_times)), dtype=bool)
                for tme in time_inds: 
                    for frq in freq_inds: 
                        mask_TF[frq,tme] = True
                        
                title = "Cluster #{0}, {1} sensor".format(i_clu + 1, len(ch_inds))
                if len(ch_inds) > 1:
                    title += "s (mean)"
                
                if tf == 'pwr':    
                    tfr_pwr_contrast_avg.plot(picks = ch_inds, baseline = None, title = title,
                                mask =  mask_TF, mask_style = 'contour', exclude = 'bads', 
                                combine = 'mean', axes = ax[1])
                elif tf == 'itc':
                    tfr_itc_contrast_avg.plot(picks = ch_inds, baseline = None, title = title,
                                mask =  mask_TF, mask_style = 'contour', exclude = 'bads', 
                                combine = 'mean', axes = ax[1])
                
                # fig.tight_layout()
                
                
                plt.show()
                report.add_figure(fig, title = contrast + "_cluster_{0}".format(i_clu + 1) , replace = True)
                
                     
                    
                plt.close('all')
           
           
    # finally saving the report after the for subject loop ends.     
    print('Saving the reports to disk')  
    report.title = 'Spatio-temp cluster TF_' + tf + ' :' + evnt+ '_' + version + '_' + timeDur + '_' + baseline +  '_' +  threshType
    #report.title = 'Group sub STC contrast at ' + evnt
    extension = 'spatio_temp cluster_TF_' + tf
    report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
    report.save(report_fname+'_' + evnt + '_' + version+ '_' + timeDur + '_' + baseline +  '_' +  threshType + '.html', overwrite=True)            
              
   
   



    #%%
    
    
            # # The overall adjacency we end up with is a square matrix with each
            # # dimension matching the data size (excluding observations) in an
            # # "unrolled" format, so: len(channels × frequencies × times)
            
            # # Compute statistic for tf_pwr and tf_itc
            # threshold = 6.0
            # F_obs_pwr, clusters_pwr, cluster_p_values_pwr, H0_pwr = permutation_cluster_test(
            #     X,
            #     out_type="mask",
            #     n_permutations=100,
            #     threshold=threshold,
            #     tail=0,
            #     seed=np.random.default_rng(seed=8675309),
            # )
            
            # F_obs_itc, clusters_itc, cluster_p_values_itc, H0_itc = permutation_cluster_test(
            #     Y,
            #     out_type="mask",
            #     n_permutations=100,
            #     threshold=threshold,
            #     tail=0,
            #     seed=np.random.default_rng(seed=8675309),
            # )
                            
    # #%% spatio_temporal_cluster_test
                

    # # We are running an F test, so we look at the upper tail
    # # see also: https://stats.stackexchange.com/a/73993
    # tail = 1
    
    # # We want to set a critical test statistic (here: F), to determine when
    # # clusters are being formed. Using Scipy's percent point function of the F
    # # distribution, we can conveniently select a threshold that corresponds to
    # # some alpha level that we arbitrarily pick.
    # alpha_cluster_forming = 0.001
    
    # # For an F test we need the degrees of freedom for the numerator
    # # (number of conditions - 1) and the denominator (number of observations
    # # - number of conditions):
    # n_conditions = 2
    # n_observations = len(X)
    # dfn = n_conditions - 1
    # dfd = n_observations - n_conditions
    
    # # Note: we calculate 1 - alpha_cluster_forming to get the critical value
    # # on the right tail
    # f_thresh = scipy.stats.f.ppf(1 - alpha_cluster_forming, dfn=dfn, dfd=dfd)
    
    # # run the cluster based permutation analysis
    # cluster_stats = spatio_temporal_cluster_test(
    #     X,
    #     n_permutations= 500,
    #     threshold=f_thresh,
    #     tail=tail,
    #     n_jobs=None,
    #     buffer_size=None,
    #     #adjacency=adjacency,
    # )
    # F_obs, clusters, p_values, _ = cluster_stats
    
    # #%%  permutation_cluster_1samp_test
    # # We want a two-tailed test
    # tail = 0
    
    # # In this example, we wish to set the threshold for including data bins in
    # # the cluster forming process to the t-value corresponding to p=0.001 for the
    # # given data.
    # #
    # # Because we conduct a two-tailed test, we divide the p-value by 2 (which means
    # # we're making use of both tails of the distribution).
    # # As the degrees of freedom, we specify the number of observations
    # # (here: epochs) minus 1.
    # # Finally, we subtract 0.001 / 2 from 1, to get the critical t-value
    # # on the right tail (this is needed for MNE-Python internals)
    # degrees_of_freedom = len(X) #len(epochs) - 1
    # t_thresh = scipy.stats.t.ppf(1 - 0.001 / 2, df=degrees_of_freedom)
    
    # # Set the number of permutations to run.
    # # Warning: 50 is way too small for a real-world analysis (where values of 5000
    # # or higher are used), but here we use it to increase the computation speed.
    # n_permutations = 5
    
    # # Run the analysis
    # T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    #     X,
    #     n_permutations=n_permutations,
    #     threshold=t_thresh,
    #     tail=tail,
    #     adjacency=adjacency,
    #     out_type="mask",
    #     verbose=True,
    # )
    
    # #%% Visualize the clusters
    
    # # We subselect clusters that we consider significant at an arbitrarily
    # # picked alpha level: "p_accept".
    # # NOTE: remember the caveats with respect to "significant" clusters that
    # # we mentioned in the introduction of this tutorial!
    # p_accept = 0.01
    # good_cluster_inds = np.where(p_values < p_accept)[0]
    
    # # loop over clusters
    # for i_clu, clu_idx in enumerate(good_cluster_inds):
    #     # unpack cluster information, get unique indices
    #     time_inds, space_inds = np.squeeze(clusters[clu_idx])
    #     ch_inds = np.unique(space_inds)
    #     time_inds = np.unique(time_inds)
    
    