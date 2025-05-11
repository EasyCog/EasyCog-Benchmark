import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, detrend, find_peaks, resample_poly, resample
from scipy.interpolate import interp1d
from scipy.signal.windows import hamming
from scipy.stats import kurtosis
from data_processing.analysis_utils import *
from utils import logger
from data_processing.display import *
from data_processing.excel_operation import read_xlsx_to_dict
from data_processing.process_eeglab import *
from data_processing.features import *
import pandas as pd
from openpyxl import load_workbook
import warnings
import pywt
import autoreject
import mne
from mne.preprocessing import ICA
import pandas as pd
import shutil

def reject_bad_segments(valid_exg, sps, epoch_len=1, method="autoreject"):
    """
    Reject bad segments from EEG data using autoreject or Ransac method.
    
    Args:
        valid_exg: EEG data array
        sps: Samples per second
        epoch_len: Length of each epoch in seconds
        method: Method to use for rejection ("autoreject" or "Ransac")
        
    Returns:
        Cleaned EEG data with bad segments removed
    """
    eeg_mne_amp_factor = 1e-6

    seg_len = epoch_len*125
    seg_data = []
    for i in range(valid_exg.shape[1]//seg_len):
        seg_data.append(valid_exg[:, i*seg_len:(i+1)*seg_len])
    seg_data = np.array(seg_data)

    # Modify ceegrid location
    info = mne.create_info(ch_names=STD_CH, sfreq=sps, ch_types='eeg')     
    epochs = mne.EpochsArray(seg_data*eeg_mne_amp_factor, info=info)
    data_montage = pd.read_excel('cEEGrid_sensor.xlsx')
    channels = np.array(data_montage)[:,0]
    value = np.array(data_montage)[:,1:]
    list_dic = dict(zip(channels, value))
    ceegrid_montage = mne.channels.make_dig_montage(ch_pos=list_dic, coord_frame='head')
    
    epochs.set_montage(ceegrid_montage)

    # step 1: apply auto reject to reject bad sensors and bad epochs (bad epochs are set to 0)
    if method=="autoreject":
        ar= autoreject.AutoReject(n_interpolate=[0], n_jobs=4, random_state=11, consensus=np.linspace(0.6,1.1,5))
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

        reject_log.plot('horizontal')
        epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=400e-6))
    elif method=="Ransac":
        rsc = autoreject.Ransac(n_jobs=4, random_state=11)
        epochs_clean, reject_log = rsc.fit_transform(epochs, return_log=True)

    # step 5 return 
    clean_epoch_data = epochs_clean.get_data()/eeg_mne_amp_factor
    selected_epochs = epochs_clean.selection

    clean_seg_data = np.zeros_like(seg_data)
    for i in range(clean_seg_data.shape[0]):
        if i in selected_epochs:
            idx = np.where(selected_epochs==i)
            clean_seg_data[i,:,:] = clean_epoch_data[idx,:,:]
    
    clean_data = clean_seg_data.transpose([1,0,2]).reshape(clean_seg_data.shape[1], -1)
    return clean_data

"""
Use Autoreject as baseline
method=["autoreject","Ransac"] 
"""
def eog_separation_baseline(valid_exg, sps, method="autoreject"):
    """
    Separate EOG artifacts from EEG data using baseline method.
    
    Args:
        valid_exg: EEG data array
        sps: Samples per second
        method: Method to use for separation ("autoreject" or "Ransac")
        
    Returns:
        Cleaned EEG data with EOG artifacts removed
    """
    eeg_mne_amp_factor = 5e-7

    seg_len = 1*125
    seg_data = []
    for i in range(valid_exg.shape[1]//seg_len):
        seg_data.append(valid_exg[:, i*seg_len:(i+1)*seg_len])
    seg_data = np.array(seg_data)

    # Modify ceegrid location
    info = mne.create_info(ch_names=STD_CH, sfreq=sps, ch_types='eeg')     
    epochs = mne.EpochsArray(seg_data*eeg_mne_amp_factor, info=info)
    data_montage = pd.read_excel('cEEGrid_sensor.xlsx')
    channels = np.array(data_montage)[:,0]
    value = np.array(data_montage)[:,1:]
    list_dic = dict(zip(channels, value))
    ceegrid_montage = mne.channels.make_dig_montage(ch_pos=list_dic, coord_frame='head')

    epochs.set_montage(ceegrid_montage)

    # step 1: apply auto reject to reject bad sensors and bad epochs (bad epochs are set to 0)
    if method=="autoreject":
        ar= autoreject.AutoReject(n_interpolate=[1,2,3,16], n_jobs=4, random_state=11)
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    elif method=="Ransac":
        rsc = autoreject.Ransac(n_jobs=4, random_state=11)
        epochs_clean, reject_log = rsc.fit_transform(epochs, return_log=True)

    # step 2: apply ICA
    ica = ICA(n_components=seg_data.shape[1],random_state=99,method='fastica')
    ica.fit(epochs[~reject_log.bad_epochs])

    # step 3: check each ICA to see which one is most likely eye artifact
    ica_components = ica.get_sources(epochs)    # get ndarray format
    ica_components = ica_components.get_data().transpose([1,0,2]).reshape(seg_data.shape[1], -1)  # transform to [n_ch, n_sample]

    blink_data = get_blink_channel(ica_components, [])  
    blink_data = smooth_data(blink_data, 20)

    corr_blink = np.zeros(16)
    for i in range(16):
        corr_blink[i] = np.corrcoef(channel_norm(ica_components[i,:]), channel_norm(blink_data))[0,1]
        print(f"IC {i+1}: Blink similarity: {corr_blink[i]}")

    eog_data = get_eog_channel(ica_components,[])
    hp_ft = butter(4, 15, 'lp', fs=sps, output='sos')
    eog_data = sosfiltfilt(hp_ft, eog_data)
    eog_data = smooth_data(eog_data, 5)

    corr_eog = np.zeros(16)
    for i in range(16):
        corr_eog[i] = np.corrcoef(channel_norm(ica_components[i,:]), channel_norm(eog_data))[0,1]
        if np.isnan(corr_eog[i]):
            corr_eog[i] = 0
        print(f"IC {i+1}: EOG similarity: {corr_eog[i]}")

    blink_IC = np.where(np.abs(corr_blink) > 0.9)[0]
    eog_IC = np.where(np.abs(corr_eog) > 0.9)[0]

    eeg = ica.apply(epochs, exclude=np.unique(np.hstack([blink_IC, eog_IC])))

    # step 4: conduct autoreject again (comment)
    # reject = autoreject.get_rejection_threshold(epochs)
    epochs_eeg = eeg

    # step 5 return 
    clean_epoch_data = epochs_eeg.get_data()/eeg_mne_amp_factor
    selected_epochs = epochs_clean.selection

    clean_seg_data = np.zeros_like(seg_data)
    for i in range(clean_seg_data.shape[0]):
        if i in selected_epochs:
            idx = np.where(selected_epochs==i)
            clean_seg_data[i,:,:] = clean_epoch_data[idx,:,:]
    
    clean_data = clean_seg_data.transpose([1,0,2]).reshape(clean_seg_data.shape[1], -1)
    return clean_data
    
def prepare_micro_states_idx_by_subject(raw_feat_path, intend_subject=None, save_path=None, logger_file=None):
    """
    Prepare microstate indices for each subject by clustering their EEG data.
    
    Args:
        raw_feat_path: Path to raw feature data
        intend_subject: List of subject IDs to process
        save_path: Path to save processed data
        logger_file: File to write log messages
        
    Returns:
        True if processing completed successfully
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    sub_name = []
    if intend_subject:
        if  isinstance(intend_subject[0], int):
            for s in intend_subject:
                sub_name.append("%03d_patient"%s)
        else:
            return
    raw_data = os.listdir(raw_feat_path)

    cnt = 0
    for file in raw_data:
        cnt += 1
        logger.info(f"Preparing micro state idx: {cnt}/{len(raw_data)}", file=logger_file)
        filepath = os.path.join(raw_feat_path, file)
        data = np.load(filepath, allow_pickle=True)
        subject = str(data['subject'])
        eeg_data = data['raw_eeg']
        if os.path.isfile(filepath):
            # fild other data with the same subject and in the same day
            same_subject_files = [f for f in raw_data if subject in f]
            for same_file in same_subject_files:
                same_filepath = os.path.join(raw_feat_path, same_file)
                if os.path.isfile(same_filepath):
                    same_data = np.load(same_filepath, allow_pickle=True)
                    # Process the same_data as needed
                    eeg_data = np.concatenate((eeg_data, same_data['raw_eeg']), axis=1)

            # generate microstate for each subject
            _, centroid = cluster_microstate(eeg_data, n_clusters=20)
            file_name = f"{subject}"
            np.savez(os.path.join(save_path,file_name),
                    subject=subject,
                    centroid=centroid,
                    )
    return True

def prepare_micro_states_idx_by_all(slice_feat_path, intend_subject=None, save_path=None, logger_file=None):
    """
    Prepare microstate indices for all subjects combined by clustering their EEG data.
    
    Args:
        slice_feat_path: Path to sliced feature data
        intend_subject: List of subject IDs to process
        save_path: Path to save processed data
        logger_file: File to write log messages
        
    Returns:
        True if processing completed successfully
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    sub_name = []
    if intend_subject:
        if  isinstance(intend_subject[0], int):
            for s in intend_subject:
                sub_name.append("%03d_patient"%s)
        else:
            return
    sliced_files = os.listdir(slice_feat_path)

    eeg_data_all = []
    cnt = 0
    for file in sliced_files:
        cnt += 1
        if cnt % 1000 == 0:
            logger.info(f"Preparing micro state idx: {cnt}/{len(sliced_files)}", file=logger_file)
        filepath = os.path.join(slice_feat_path, file)
        data = np.load(filepath, allow_pickle=True)
        
        # concatenate all eeg data
        eeg_data = data['eeg_seg']
        eeg_data_all.append(eeg_data)

    eeg_data_all = np.concatenate(eeg_data_all, axis=1)
    _, centroid = cluster_microstate(eeg_data_all, n_clusters=20)
    file_name = f"all"
    np.savez(os.path.join(save_path,file_name),
            subject='all',
            centroid=centroid,)
    return True


def fix_bad_channel_segments(data_folder_path, interp_param=None, intend_subject=None, logger_file=None):
    """
    Fix bad channel segments in EEG data using interpolation.
    
    Args:
        data_folder_path: Path to data folder
        interp_param: Interpolation parameters
        intend_subject: List of subject IDs to process
        logger_file: File to write log messages
        
    Returns:
        True if processing completed successfully
    """
    if not interp_param['if_interp']:
        return True
    
    win_size = interp_param['win_size']
    win_step = interp_param['win_step']
    sps = interp_param['sps']

    sub_name = []
    if intend_subject:
        if  isinstance(intend_subject[0], int):
            for s in intend_subject:
                sub_name.append("%03d_patient"%s)
        else:
            return
    unsliced_files = os.listdir(data_folder_path)

    cnt = 0
    for file in unsliced_files:
        cnt += 1
        filepath = os.path.join(data_folder_path, file)
        if os.path.isfile(filepath):
            unsliced_data = np.load(filepath, allow_pickle=True)
            subject = str(unsliced_data['subject'])
            if intend_subject is not None:
                if subject not in sub_name:
                    continue
            logger.info(f"Selecting bad channels for {cnt}/{len(unsliced_files)}: {file}", file=logger_file)
            unsliced_data_dict = dict(unsliced_data)
            raw_eeg = unsliced_data_dict['raw_eeg']

            # sliding window
            start, end = 0, int(win_size*sps)
            bad_chs = np.zeros((raw_eeg.shape[0], raw_eeg.shape[1]))
            while end <= raw_eeg.shape[1]:
                eeg_seg = raw_eeg[:, start:end]
                # detect bad channels
                for i in range(eeg_seg.shape[0]):
                    peak_value = np.max(eeg_seg[i,:])-np.min(eeg_seg[i,:])

                    data = eeg_seg[i,:]
                    # plot fft spectrum
                    f, Pxx = signal.welch(data, fs=sps, nperseg=1024)

                    Pxx = (Pxx[f>10].sum())/Pxx.sum()
                    if Pxx < 1e-2 or peak_value < 1e-4:
                        bad_chs[i, start:end] = 1
                start, end = start+int(win_step*sps), end+int(win_step*sps)

            start, end = 0, int(win_size*sps)
            while end <= raw_eeg.shape[1]:
                # bad_chs_seg is the bad channels in the current segment
                bad_chs_seg = np.where(bad_chs[:, start:end].sum(axis=1)>0)[0]
                if len(bad_chs_seg) == raw_eeg.shape[0]:
                    logger.info(f"All channels are bad: {bad_chs} in {start}-{end}", file=logger_file)
                elif len(bad_chs_seg) > 0:
                    eeg_seg = raw_eeg[:, start:end]
                    logger.info(f"Detect bad channels: {bad_chs_seg} in {start}-{end}", file=logger_file)
                    for ch in bad_chs_seg:
                        # find the nearest 2 good channel due to the STD_CH_DIST array
                        good_chs = [ch for ch in range(16) if ch not in bad_chs_seg]
                        if len(good_chs) < 2:
                            good_chs = [good_chs[0], good_chs[0]]
                        distances = np.array([np.linalg.norm(np.array(STD_CH_DIST[ch]) - np.array(STD_CH_DIST[good_ch])) for good_ch in good_chs])

                        nearest_good_chs = np.argsort(distances)[:2]

                        neareat_good_chs_dist = distances[nearest_good_chs]
                        # weight is calculated by the distance, the closer the channel, the more weight it has, and the sum of the weight should be 1
                        neareat_good_chs_weight = 1/neareat_good_chs_dist
                        neareat_good_chs_weight = neareat_good_chs_weight/neareat_good_chs_weight.sum()
                        nearest_good_chs = [good_chs[idx] for idx in nearest_good_chs]
                        # intepolate the bad channel with the nearest 2 good channels
                        for i in range(eeg_seg.shape[1]):
                            eeg_seg[ch,i] = neareat_good_chs_weight[0]*eeg_seg[nearest_good_chs[0],i] + neareat_good_chs_weight[1]*eeg_seg[nearest_good_chs[1],i]
                        logger.info(f"Interpolated channel {ch} with {nearest_good_chs}", file=logger_file)
                    raw_eeg[:, start:end] = eeg_seg
                start, end = start+int(win_step*sps), end+int(win_step*sps)
            
            # new eog
            eog_data = get_eog_channel(raw_eeg,[])
            hp_ft = butter(10, 15, 'lp', fs=sps,  output='sos')
            eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
            for i in range(eog_data.shape[0]):
                eog_data[i,:] = smooth_data(eog_data[i,:], 5)

            unsliced_data_dict['raw_eeg'] = raw_eeg
            unsliced_data_dict['raw_eog'] = eog_data

            # New task data
            data_type = str(unsliced_data_dict['type'])
            exg_timestamps = unsliced_data_dict['timestamp']
            valid_et_sync_rs = unsliced_data_dict['eye_tracking']
            
            if data_type == 'video':
                task_data, task_et = split_tasks_exg(raw_eeg, valid_et_sync_rs, sps, exg_timestamps[-1]-exg_timestamps[0], exg_timestamps)
                task_eog, _ = split_tasks_exg(eog_data,valid_et_sync_rs, sps, exg_timestamps[-1]-exg_timestamps[0], exg_timestamps)

                unsliced_data_dict['task_eeg'] = task_data
                unsliced_data_dict['task_eog'] = task_eog
                unsliced_data_dict['task_et'] = task_et

            np.savez(filepath, **unsliced_data_dict)
                
    return True


def fix_bad_channel_data(valid_exg, win_size=3, win_step=2, ratio=0.3, sps=125, mode='all', data_type='resting', logger_file=None):
    """
    Fix bad channels in EEG data using spatial or temporal interpolation.
    
    Args:
        valid_exg: EEG data array
        win_size: Window size for analysis
        win_step: Step size between windows
        ratio: Threshold ratio for bad channel detection
        sps: Samples per second
        mode: Interpolation mode ('all', 'spatial', or 'temporal')
        data_type: Type of data ('resting' or other)
        logger_file: File to write log messages
        
    Returns:
        Tuple of (bad channel indices, processed EEG data, bad segment mask)
    """
    if mode == 'all':
        if_spatial = True
    elif mode == 'spatial':
        if_spatial = True
    elif mode == 'temporal':
        if_spatial = False
    else:
        raise ValueError(f"Invalid interpolation mode: {mode}")

    if if_spatial:
        raw_eeg = valid_exg
        # sliding window
        start, end = 0, int(win_size*sps)
        bad_chs = np.zeros((raw_eeg.shape[0], raw_eeg.shape[1]))
        while end <= raw_eeg.shape[1]:
            eeg_seg = raw_eeg[:, start:end]
            # detect bad channels
            for i in range(eeg_seg.shape[0]):
                peak_value = np.max(eeg_seg[i,:])-np.min(eeg_seg[i,:])

                if peak_value < 1e-4 or peak_value > 1000:
                    bad_chs[i, start:end] = 1
            start, end = start+int(win_step*sps), end+int(win_step*sps)

        bad_chs_ret = []
        for i in range(raw_eeg.shape[0]):
            bad_ratio = bad_chs[i,:].sum()/raw_eeg.shape[1]
            if bad_ratio > ratio:
                logger.info(f"Channel {i} has {bad_ratio} bad channels", file=logger_file)
                bad_chs_ret.append(i)

        start, end = 0, int(win_size*sps)
        while end <= raw_eeg.shape[1]:
            eeg_seg = raw_eeg[:, start:end]
            bad_chs_seg = np.where(bad_chs[:, start:end].sum(axis=1)>0)[0]
            if len(bad_chs_seg) == 16:
                start, end = start+int(win_step*sps), end+int(win_step*sps)
                continue
            for ch in bad_chs_ret:
            # bad_chs_seg is the bad channels in the current segment
                if bad_chs[ch, start:end].sum() > 0:
                # find the nearest 2 good channel due to the STD_CH_DIST array
                    good_chs = [ch for ch in range(16) if ch not in bad_chs_seg]
                    if len(good_chs) < 2:
                        good_chs = [good_chs[0], good_chs[0]]
                    # print(ch, good_chs)
                    distances = np.array([np.linalg.norm(np.array(STD_CH_DIST[ch]) - np.array(STD_CH_DIST[good_ch])) for good_ch in good_chs])

                    nearest_good_chs = np.argsort(distances)[:2]

                    neareat_good_chs_dist = distances[nearest_good_chs]
                    # weight is calculated by the distance, the closer the channel, the more weight it has, and the sum of the weight should be 1
                    neareat_good_chs_weight = 1/neareat_good_chs_dist
                    neareat_good_chs_weight = neareat_good_chs_weight/neareat_good_chs_weight.sum()
                    nearest_good_chs = [good_chs[idx] for idx in nearest_good_chs]
                    # intepolate the bad channel with the nearest 2 good channels
                    for i in range(eeg_seg.shape[1]):
                        if bad_chs[ch,i+start] == 1:
                            eeg_seg[ch,i] = neareat_good_chs_weight[0]*eeg_seg[nearest_good_chs[0],i] + neareat_good_chs_weight[1]*eeg_seg[nearest_good_chs[1],i]
                    raw_eeg[:, start:end] = eeg_seg
            start, end = start+int(win_step*sps), end+int(win_step*sps)
                
        return bad_chs_ret, raw_eeg, bad_chs
    else:
        return None, valid_exg, None

def fix_burst_channels_segments(sliced_feat_train_path, proc_param=None, intend_subject=None, save_path=None, logger_file=None):
    """
    Fix burst artifacts in channel segments using interpolation.
    
    Args:
        sliced_feat_train_path: Path to sliced feature data
        proc_param: Processing parameters
        intend_subject: List of subject IDs to process
        save_path: Path to save processed data
        logger_file: File to write log messages
        
    Returns:
        True if processing completed successfully
    """
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    sps = proc_param['interp_param']['sps']
    sub_name = []

    if intend_subject:
        if isinstance(intend_subject[0], int):
            for s in intend_subject:
                sub_name.append("%03d_patient"%s)
        else:
            return
    unsliced_files = os.listdir(sliced_feat_train_path)

    cnt = 0
    for file in unsliced_files:
        cnt += 1
        filepath = os.path.join(sliced_feat_train_path, file)
        if os.path.isfile(filepath):
            unsliced_data = np.load(filepath, allow_pickle=True)
            subject = str(unsliced_data['subject'])
            date = str(unsliced_data['date'])
            data_type = str(unsliced_data['type'])
            task_no = unsliced_data['task_no']
            pic_no = unsliced_data['pic_no']
            start = file.split('.')[0].split('-')[-2]
            end = file.split('.')[0].split('-')[-1]
            if intend_subject is not None:
                if subject not in sub_name:
                    continue
            if cnt % 1000 == 0:
                logger.info(f"Selecting burst channels for {cnt}/{len(unsliced_files)}", file=logger_file)
            unsliced_data_dict = dict(unsliced_data)
            raw_eeg = unsliced_data_dict['eeg_seg']

            # sliding window
            bad_chs = np.zeros((raw_eeg.shape[0]))
            for i in range(raw_eeg.shape[0]):
                peak_value = np.max(np.abs(raw_eeg[i,:]))
                # group_0_to_500:  0.9990937673474303
                # group_500_to_1000:  0.9995777482464828
                # group_1000_to_2000:  0.9997559677641326
                # group_2000_to_3000:  0.999820860386227
                # group_3000_to_4000:  0.9998493316752581
                # group_larger_than_4000:  0.00015066832474190995
                if peak_value > 500:
                    bad_chs[i] = 1
            
            if bad_chs.sum() == raw_eeg.shape[0]:
                if data_type == 'resting':
                    #find the nearest data with the nearest pic_no, task_no, and start, end
                    nearest_data = []
                    for f in unsliced_files:
                        if f.startswith(f"{subject}-{date}-{data_type}"):
                            # record the nearest 5 data
                            dist = abs(int(start)-int(f.split('.')[0].split('-')[-2])) + abs(int(end)-int(f.split('.')[0].split('-')[-1]))
                            if len(nearest_data) < 6:
                                nearest_data.append((dist, f))
                                nearest_data.sort()  
                            else:
                                if dist < nearest_data[-1][0]:  
                                    nearest_data[-1] = (dist, f)  
                                    nearest_data.sort()  
                    if len(nearest_data) > 0:
                        for i in range(len(nearest_data)):
                            unsliced_data = np.load(os.path.join(sliced_feat_train_path, nearest_data[i][1]), allow_pickle=True)
                            raw_eeg = unsliced_data['eeg_seg']
                            if np.max(np.abs(raw_eeg),axis=(0,1)) < 500:
                                unsliced_data_dict['eeg_seg'] = raw_eeg
                                unsliced_data_dict['et_seg'] = unsliced_data['et_seg']
                                break
                else:
                    same_pic_data = []
                    same_task_data = []
                    for f in unsliced_files:
                        if f.startswith(f"{subject}-{date}-{data_type}-task{task_no}-pic{pic_no}"):
                            dist = abs(int(start)-int(f.split('.')[0].split('-')[-2])) + abs(int(end)-int(f.split('.')[0].split('-')[-1]))
                            if len(same_pic_data) < 6:
                                same_pic_data.append((dist, f, f.split('.')[0].split('-')[-4][-1], f.split('.')[0].split('-')[-3][-1],))
                                same_pic_data.sort()  
                        elif f.startswith(f"{subject}-{date}-{data_type}-task{task_no}"):
                            dist = 10e9*abs(int(pic_no)-int(f.split('.')[0].split('-')[-3][-1]))+ abs(int(start)-int(f.split('.')[0].split('-')[-2])) + abs(int(end)-int(f.split('.')[0].split('-')[-1]))
                            if len(same_task_data) < 6:
                                same_task_data.append((dist, f, f.split('.')[0].split('-')[-4][-1], f.split('.')[0].split('-')[-3][-1],))
                                same_task_data.sort()  
                            else:
                                if dist < same_task_data[-1][0]:  
                                    same_task_data[-1] = ((dist, f, f.split('.')[0].split('-')[-4][-1], f.split('.')[0].split('-')[-3][-1],))  
                                    same_task_data.sort() 


                    if len(same_pic_data) > 0:
                        for i in range(len(same_pic_data)):
                            unsliced_data = np.load(os.path.join(sliced_feat_train_path, same_pic_data[i][1]), allow_pickle=True)
                            raw_eeg = unsliced_data['eeg_seg']
                            if np.max(np.abs(raw_eeg),axis=(0,1)) < 500:
                                unsliced_data_dict['eeg_seg'] = raw_eeg
                                unsliced_data_dict['et_seg'] = unsliced_data['et_seg']
                                break
                    if len(same_task_data) > 0:
                        for i in range(len(same_task_data)):
                            unsliced_data = np.load(os.path.join(sliced_feat_train_path, same_task_data[i][1]), allow_pickle=True)
                            raw_eeg = unsliced_data['eeg_seg']
                            if np.max(np.abs(raw_eeg),axis=(0,1)) < 500:
                                unsliced_data_dict['eeg_seg'] = raw_eeg
                                unsliced_data_dict['et_seg'] = unsliced_data['et_seg']
                                break

                # new eog
                eog_data = get_eog_channel(unsliced_data_dict['eeg_seg'],[])
                hp_ft = butter(10, 15, 'lp', fs=sps,  output='sos')
                eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                for i in range(eog_data.shape[0]):
                    eog_data[i,:] = smooth_data(eog_data[i,:], 5)
                unsliced_data_dict['eog_seg'] = eog_data
            filepath = os.path.join(save_path, file)
            np.savez(filepath, **unsliced_data_dict)
                
    return True

def update_norm_param(raw_feat_train_path, sliced_feat_train_path, save_path=None, logger_file=None):
    """
    Update normalization parameters for EEG and EOG data.
    
    Args:
        raw_feat_train_path: Path to raw feature data
        sliced_feat_train_path: Path to sliced feature data
        save_path: Path to save processed data
        logger_file: File to write log messages
        
    Returns:
        True if processing completed successfully
    """
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    raw_files = os.listdir(raw_feat_train_path)
    sliced_files = os.listdir(sliced_feat_train_path)

    cnt = 0
    for file in raw_files:
        cnt += 1
        subject = str(file.split('-')[0])
        data_type = str(file.split('.')[0].split('-')[-1])
        date = str(file.split('.')[0].split('-')[1])
        sliced_file_list = []
        for f in sliced_files:
            if f.startswith(f"{subject}-{date}-{data_type}"):   # find all the sliced files with the same subject, date, and data_type
                sliced_file_list.append(f)
        if len(sliced_file_list) == 0:
            continue
        logger.info(f"Updating norm param for {cnt}/{len(raw_files)}", file=logger_file)
        eeg_all = []
        eog_all = []
        for f in sliced_file_list:
            sliced_data = np.load(os.path.join(sliced_feat_train_path, f), allow_pickle=True)
            eeg_seg = sliced_data['eeg_seg']
            eog_seg = sliced_data['eog_seg']
            eeg_all.append(eeg_seg)
            eog_all.append(eog_seg)
        # concat all the eeg and eog data
        eeg_all = np.concatenate(eeg_all, axis=1)
        eog_all = np.concatenate(eog_all, axis=1)
        new_eeg_mean = np.mean(eeg_all, axis=1)
        new_eeg_std = np.std(eeg_all, axis=1)
        new_eog_mean = np.mean(eog_all, axis=1)
        new_eog_std = np.std(eog_all, axis=1)
        new_eeg_std_all = np.std(eeg_all, axis=(0,1))
        new_eog_std_all = np.std(eog_all, axis=(0,1))
        new_eeg_mean_all = np.mean(eeg_all, axis=(0,1))
        new_eog_mean_all = np.mean(eog_all, axis=(0,1))
        for f in sliced_file_list:
            sliced_data = dict(np.load(os.path.join(sliced_feat_train_path, f), allow_pickle=True))
            sliced_data['eeg_std'] = new_eeg_std
            sliced_data['eog_std'] = new_eog_std
            sliced_data['eeg_mean'] = new_eeg_mean
            sliced_data['eog_mean'] = new_eog_mean
            sliced_data['eeg_std_all'] = new_eeg_std_all
            sliced_data['eog_std_all'] = new_eog_std_all
            sliced_data['eeg_mean_all'] = new_eeg_mean_all
            sliced_data['eog_mean_all'] = new_eog_mean_all
            filepath = os.path.join(save_path, f)
            np.savez(filepath, **sliced_data)
    return True

def avgref_data(sliced_feat_train_path, save_path=None, logger_file=None):
    """
    Apply average reference to EEG data.
    
    Args:
        sliced_feat_train_path: Path to sliced feature data
        save_path: Path to save processed data
        logger_file: File to write log messages
        
    Returns:
        True if processing completed successfully
    """
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    unsliced_files = os.listdir(sliced_feat_train_path)

    cnt = 0
    for file in unsliced_files:
        cnt += 1
        filepath = os.path.join(sliced_feat_train_path, file)
        if os.path.isfile(filepath):
            unsliced_data = np.load(filepath, allow_pickle=True)
            subject = str(unsliced_data['subject'])
            date = str(unsliced_data['date'])
            data_type = str(unsliced_data['type'])
            task_no = unsliced_data['task_no']
            pic_no = unsliced_data['pic_no']
            start = file.split('.')[0].split('-')[-2]
            end = file.split('.')[0].split('-')[-1]

            if cnt % 1000 == 0:
                logger.info(f"Avgrefing data for {cnt}/{len(unsliced_files)}", file=logger_file)
            unsliced_data_dict = dict(unsliced_data)
            raw_eeg = unsliced_data_dict['eeg_seg']

            raw_eeg -= np.mean(raw_eeg, axis=0, keepdims=True)
            unsliced_data_dict['eeg_seg'] = raw_eeg

            filepath = os.path.join(save_path, file)
            np.savez(filepath, **unsliced_data_dict)
    
    return True

def stack_feature(processed_feat_path, intend_subject=None, feature=None, logger_file=None):
    """
    Stack multiple features into a single array.
    
    Args:
        processed_feat_path: Path to processed feature data
        intend_subject: List of subject IDs to process
        feature: List of features to stack
        logger_file: File to write log messages
        
    Returns:
        None
    """
    sub_name = []
    if intend_subject:
        if  isinstance(intend_subject[0], int):
            for s in intend_subject:
                sub_name.append("%03d_patient"%s)
        else:
            return
    processed_files = os.listdir(processed_feat_path)

    cnt = 0
    for file in processed_files:
        cnt += 1
        filepath = os.path.join(processed_feat_path, file)
        if os.path.isfile(filepath):
            processed_data = np.load(filepath, allow_pickle=True)
            subject = str(processed_data['subject'])
            if intend_subject is not None:
                if subject not in sub_name:
                    continue
            if cnt % 1000 == 0:
                logger.info(f"Stacking features {cnt}/{len(processed_files)}", file=logger_file)
        
            processed_data_dict = dict(processed_data)
            stacked_feat = None
            for f in feature:
                value = processed_data_dict[f]
                # print(f, value.shape)
                if len(value.shape)==0:
                    value = np.tile(np.array([value]), (16))

                if value.shape[0] == 16:
                    if stacked_feat is None:
                        stacked_feat = value.reshape(16,-1)
                    else:
                        stacked_feat = np.concatenate((stacked_feat, value.reshape(16,-1)), axis=1)
                else:   # 20
                    if stacked_feat is None:
                        if len(value.shape)==1:
                            # remat 16 times
                            stacked_feat = np.tile(value, (16, 1))
                        else:
                            # micro_states_center
                            stacked_feat = value.transpose(1,0)
                    else:
                        if len(value.shape)==1:
                            # remat 16 times
                            stacked_feat = np.concatenate((stacked_feat, np.tile(value, (16, 1))), axis=1)
                        else:
                            # micro_states_center
                            stacked_feat = np.concatenate((stacked_feat, value.transpose(1,0)), axis=1)
            processed_data_dict['stacked_feat'] = stacked_feat
            np.savez(filepath, **processed_data_dict)

    logger.info(f"Stacking features completed", file=logger_file)
    return

# replace nan of each sliced data with the inteplota value
def interpolate_nan_eye_tracking(data_folder_path, intend_subject=None, logger_file=None):
    """
    Interpolate NaN values in eye tracking data.
    
    Args:
        data_folder_path: Path to data folder
        intend_subject: List of subject IDs to process
        logger_file: File to write log messages
        
    Returns:
        True if processing completed successfully
    """
    sub_name = []
    if intend_subject:
        if  isinstance(intend_subject[0], int):
            for s in intend_subject:
                sub_name.append("%03d_patient"%s)
        else:
            return
    sliced_files = os.listdir(data_folder_path)

    cnt = 0
    for file in sliced_files:
        cnt += 1
        filepath = os.path.join(data_folder_path, file)
        if os.path.isfile(filepath):
            unsliced_data = np.load(filepath, allow_pickle=True)
            subject = str(unsliced_data['subject'])
            if intend_subject is not None:
                if subject not in sub_name:
                    continue
            if str(unsliced_data['type']) == 'resting':
                continue
            if cnt % 1000 == 0:
                logger.info(f"Interpolating nan for {cnt}/{len(sliced_files)}", file=logger_file)
            unsliced_data_dict = dict(unsliced_data)
            eye_tracking = unsliced_data_dict['et_seg']
            if np.isnan(eye_tracking).any():
                eye_tracking = interpolate_nan_channel(eye_tracking.T).T
                unsliced_data_dict['et_seg'] = eye_tracking
                np.savez(filepath, **unsliced_data_dict)
            else:
                continue
    return True

def interpolate_nan_channel(data):
    """
    Interpolate NaN values in a single channel of data.
    
    Args:
        data: Data array with NaN values
        
    Returns:
        Data array with interpolated values
    """
    for i in range(data.shape[0]):
        nan_idx = np.where(np.isnan(data[i,:]))[0]
        if len(nan_idx) > 0:
            nan_idx = np.array(nan_idx)
            if nan_idx[0] == 0:
                nan_idx = nan_idx[1:]
            if nan_idx[-1] == data.shape[1]-1:
                nan_idx = nan_idx[:-1]
            if len(nan_idx) > 0:
                data[i,nan_idx] = np.interp(nan_idx, nan_idx, data[i,nan_idx])
    return data

def rough_interpolate_bad_channel(valid_exg, sps=125, logger_file=None):
    """
    Perform rough interpolation of bad channels using nearest good channels.
    
    Args:
        valid_exg: EEG data array
        sps: Samples per second
        logger_file: File to write log messages
        
    Returns:
        Processed EEG data with interpolated bad channels
    """
    bad_chs = []
    for i in range(valid_exg.shape[0]):
        exg_std = valid_exg[i,:].std()
        if  exg_std < 0.1:
            bad_chs.append(i)
    logger.info(f"Detect bad channels: {bad_chs}", file=logger_file)

    for ch in bad_chs:
        # find the nearest 2 good channel due to the STD_CH_DIST array
        good_chs = [ch for ch in range(16) if ch not in bad_chs]
        distances = np.array([np.linalg.norm(np.array(STD_CH_DIST[ch]) - np.array(STD_CH_DIST[good_ch])) for good_ch in good_chs])
        nearest_good_chs = np.argsort(distances)[:2]
        neareat_good_chs_dist = distances[nearest_good_chs]
        # weight is calculated by the distance, the closer the channel, the more weight it has, and the sum of the weight should be 1
        neareat_good_chs_weight = 1/neareat_good_chs_dist
        neareat_good_chs_weight = neareat_good_chs_weight/neareat_good_chs_weight.sum()
        nearest_good_chs = [good_chs[idx] for idx in nearest_good_chs]
        # intepolate the bad channel with the nearest 2 good channels
        for i in range(valid_exg.shape[1]):
            valid_exg[ch,i] = neareat_good_chs_weight[0]*valid_exg[nearest_good_chs[0],i] + neareat_good_chs_weight[1]*valid_exg[nearest_good_chs[1],i]
        logger.info(f"Interpolated channel {ch} with {nearest_good_chs}", file=logger_file)
    return valid_exg

def remove_common_interference(valid_exg, sps=125, corr_thr=0.6, seg=1, logger_file=None):
    """
    Remove common interference from EEG data using ICA.
    
    Args:
        valid_exg: EEG data array
        sps: Samples per second
        corr_thr: Correlation threshold for interference detection
        seg: Number of segments to process
        logger_file: File to write log messages
        
    Returns:
        Cleaned EEG data with common interference removed
    """
    eeg_mne_amp_factor = 5e-7

    bad_chs, temp_exg, bad_seg = fix_bad_channel_data(valid_exg.copy(), 3, 0.3, 0.3, sps, logger_file)

    seg_len = valid_exg.shape[1]//seg
    seg_data = []
    for i in range(valid_exg.shape[1]//seg_len):
        seg_data.append(temp_exg[:, i*seg_len:(i+1)*seg_len])
    seg_data = np.array(seg_data)
    clean_seg_data = np.zeros_like(seg_data)

    for j in range(seg_data.shape[0]):
        # Modify ceegrid location
        info = mne.create_info(ch_names=STD_CH, sfreq=sps, ch_types='eeg')     
        epochs = mne.EpochsArray(seg_data[j,None,:,:]*eeg_mne_amp_factor, info=info)
        data_montage = pd.read_excel('cEEGrid_sensor.xlsx')
        channels = np.array(data_montage)[:,0]
        value = np.array(data_montage)[:,1:]
        list_dic = dict(zip(channels, value))
        ceegrid_montage = mne.channels.make_dig_montage(ch_pos=list_dic, coord_frame='head')
        
        epochs.set_montage(ceegrid_montage)

        trial_times = 3
        random_state = [99, 23, 61]
        for i in range(trial_times):
            logger.info(f"Removing common interference: {i+1}/{trial_times}", file=logger_file)
        # Conduct ICA
            ica = ICA(n_components=16, random_state=random_state[i], method='fastica')
            ica.fit(epochs)

            # Get ICA components
            ica_components = ica.get_sources(epochs).get_data().transpose([1,0,2]).reshape(seg_data.shape[1], -1) 

            # Find common interference by calculating the correlation of each component with all original channels of in seg_data
            corr_matrix = np.corrcoef(ica_components, seg_data[j,None,:,:].transpose([1,0,2]).reshape(seg_data.shape[1], -1) )[:16,16:]

            # find the component with the highest correlation majority of the channels
            corr_matrix = np.abs(corr_matrix)
            corr_matrix = np.mean(corr_matrix, axis=1)
            # find the highest correlation component
            common_interference_idx = np.argmax(corr_matrix)
            # if larger than thr, mark as bad component
            if corr_matrix[common_interference_idx] > corr_thr:
                common_interference_idx = common_interference_idx
                logger.info(f"Common interference found: {common_interference_idx}", file=logger_file)
                epochs = ica.apply(epochs, exclude=[common_interference_idx])
            else:
                continue
        # step 5 return 
        clean_epoch_data = epochs.get_data()/eeg_mne_amp_factor
        clean_seg_data[j,:,:] = clean_epoch_data[0,:,:]

    
    clean_data = clean_seg_data.transpose([1,0,2]).reshape(clean_seg_data.shape[1], -1)

    if bad_chs:
        for ch in bad_chs:
            clean_data[ch,:] = valid_exg[ch,:] - (temp_exg[ch,:] - clean_data[ch,:])
            bad_seg_idx = np.where(bad_seg[ch,:]==1)[0]
            clean_data[ch,bad_seg_idx] = valid_exg[ch,bad_seg_idx]
    
    return clean_data


def read_subject(root_path, subject, proc_param, logger_file=None):
    """
    Read and process data for a single subject.
    
    Args:
        root_path: Root path to data
        subject: Subject ID
        proc_param: Processing parameters
        logger_file: File to write log messages
        
    Returns:
        List of processed data dictionaries
    """
    # read video task data
    path = os.path.join(root_path, subject)
    
    result = []
    found_flag = False
    for foldername, subfolders, _ in os.walk(path):
        for subfolder in subfolders:
            if "video_task" in os.listdir(os.path.join(foldername, subfolder)):
                if check_completence(os.path.join(path, subfolder, 'video_task'), type='video'):
                    date = subfolder
                    data_path = os.path.join(path, subfolder, 'video_task')
                    data_type = 'video'
                    found_flag = True
                    logger.info(f"found file {data_path}", file=logger_file)

            if "resting_task" in os.listdir(os.path.join(foldername, subfolder)):
                if check_completence(os.path.join(path, subfolder, 'resting_task'), type='resting'):
                    date = subfolder
                    data_path = os.path.join(path, subfolder, 'resting_task')
                    data_type = 'resting'
                    found_flag = True
                    logger.info(f"found file {data_path}", file=logger_file)

            if found_flag:
                found_flag = False
                exg_data, exg_timestamps, sps, starting_points, video_timestamps2, video_timestamps, et_ts, et_data = read_data(data_path, type=data_type)

                exg_data = channel_remapping(exg_data)

                cancel_prefix = proc_param['cancel_prefix'] # unit:s
                ft_exg_data = np.zeros_like(exg_data)
                
                # filter the drifting （# filter from LaBraM: 0.1~75）
                hp_ft = butter(proc_param['filter_param']['order'], proc_param['filter_param']['low_freq'], 'hp', fs=sps, output='sos')  #

                ft_exg_data[:, int(cancel_prefix*sps):] = sosfiltfilt(hp_ft, (exg_data[:, int(cancel_prefix*sps):]), axis=1)

                if proc_param['if_notch']:
                    bs_ft = butter(3, [31, 32], 'bs', fs=sps, output='sos')
                    ft_exg_data = sosfiltfilt(bs_ft, ft_exg_data, axis=1)

                start_timestamp = float(starting_points['video_start_time'])
                end_timestamp = float(starting_points['video_end_time'])
                start_exg_idx = np.argmin(np.abs(exg_timestamps - start_timestamp))
                end_exg_idx = np.argmin(np.abs(exg_timestamps - end_timestamp))

                if data_type == 'video':
                    start_et_idx = np.argmin(np.abs(et_ts - start_timestamp))
                    end_et_idx = np.argmin(np.abs(et_ts - end_timestamp))
                    valid_et = et_data[start_et_idx:end_et_idx]
                    valid_et_ts = et_ts[start_et_idx:end_et_idx]

                if data_type == 'video':
                    duration = TASK_DURATION
                    valid_exg = ft_exg_data[:, start_exg_idx:start_exg_idx+int(duration*sps)]
                    # craft exg timestamps due to mixing Openbci records
                    exg_timestamps = np.linspace(start_timestamp, end_timestamp, valid_exg.shape[1])
                else:
                    duration = RESTING_DURATION
                    valid_exg = ft_exg_data[:, end_exg_idx-int(duration*sps):end_exg_idx]
                    # craft exg timestamps due to mixing Openbci records
                    exg_timestamps = np.linspace(start_timestamp, end_timestamp, valid_exg.shape[1])

                ## Processing eye tracking data
                if data_type == 'video':
                    if proc_param['if_resample_et']:
                        if valid_et_ts[-1] < duration + valid_et_ts[0]:
                            new_ts_part1 = np.linspace(valid_et_ts[0], valid_et_ts[-1], int((valid_et_ts[-1] - valid_et_ts[0]) * sps))
                            new_ts_part2 = np.linspace(valid_et_ts[-1], duration + valid_et_ts[0], int((duration + valid_et_ts[0] - valid_et_ts[-1]) * sps))
                            new_ts = np.concatenate((new_ts_part1, new_ts_part2))
                            valid_et_rs_part1 = np.zeros([new_ts_part1.shape[0],2])
                            interpolator = interp1d(valid_et_ts, valid_et[:,0], kind='linear', fill_value='extrapolate')
                            valid_et_rs_part1[:,0] = interpolator(new_ts_part1)
                            interpolator = interp1d(valid_et_ts, valid_et[:,1], kind='linear', fill_value='extrapolate')
                            valid_et_rs_part1[:,1] = interpolator(new_ts_part1)

                            valid_et_rs_part2 = np.full((new_ts_part2.shape[0],2), 0)
                            valid_et_rs = np.concatenate((valid_et_rs_part1, valid_et_rs_part2))
                        else:
                            new_ts = np.linspace(valid_et_ts[0], valid_et_ts[0]+duration, int(duration*sps))
                            valid_et_rs = np.zeros([new_ts.shape[0],2])
                            interpolator = interp1d(valid_et_ts, valid_et[:,0], kind='linear', fill_value='extrapolate')
                            valid_et_rs[:,0] = interpolator(new_ts)
                            interpolator = interp1d(valid_et_ts, valid_et[:,1], kind='linear', fill_value='extrapolate')
                            valid_et_rs[:,1] = interpolator(new_ts)
                        valid_et_ts_rs = new_ts
                        
                        valid_et_sync_rs = np.zeros([valid_exg.shape[1],2])
                        for i in range(len(valid_et_sync_rs)):
                            valid_et_sync_rs[i,:] = valid_et_rs[np.argmin(np.abs(exg_timestamps[i]-valid_et_ts_rs))]
                        
                    else:
                        valid_et_sync_rs = np.zeros([valid_exg.shape[1],2])
                        for i in range(len(valid_et_sync_rs)):
                            valid_et_sync_rs[i,:] = valid_et[np.argmin(np.abs(exg_timestamps[i]-valid_et_ts))]
                else:
                    valid_et_sync_rs = None


                logger.info(f"Performing eye removal: {proc_param['eye_removal']}", file=logger_file)
                if proc_param['eye_removal'] == "baseline":
                    if data_type == "video":
                        start = int(np.mod(int(duration*sps), sps))
                        eog_data = get_eog_channel(valid_exg,[])
                        # eog_data = seg[0,:] - seg[8,:]
                        hp_ft = butter(4, 15, 'lp', fs=sps, output='sos')
                        eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                        for i in range(eog_data.shape[0]):
                            eog_data[i,:] = smooth_data(eog_data[i,:], 5)
                        valid_exg[:,start:] = eog_separation_baseline(valid_exg[:,start:], sps, method="autoreject")
                    else:
                        eog_data = get_eog_channel(valid_exg,[])
                        # eog_data = seg[0,:] - seg[8,:]
                        hp_ft = butter(4, 15, 'lp', fs=sps, output='sos')
                        eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                        for i in range(eog_data.shape[0]):
                            eog_data[i,:] = smooth_data(eog_data[i,:], 5)
                        valid_exg = reject_bad_segments(valid_exg, sps, method="autoreject")
                        
                elif proc_param['eye_removal'] == "None":
                    if data_type == "video":
                        valid_exg = valid_exg
                        eog_data = get_eog_channel(valid_exg,[])
                        # eog_data = seg[0,:] - seg[8,:]
                        hp_ft = butter(4, 15, 'lp', fs=sps, output='sos')
                        eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                        for i in range(eog_data.shape[0]):
                            eog_data[i,:] = smooth_data(eog_data[i,:], 5)
                    else:
                        valid_exg = valid_exg
                        eog_data = get_eog_channel(valid_exg,[])
                        # eog_data = seg[0,:] - seg[8,:]
                        hp_ft = butter(4, 15, 'lp', fs=sps, output='sos')
                        eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                        for i in range(eog_data.shape[0]):
                            eog_data[i,:] = smooth_data(eog_data[i,:], 5)

                elif proc_param['eye_removal'] == "autoreject":
                    if data_type == "video":
                        start = int(np.mod(int(duration*sps), sps))
                        eog_data = get_eog_channel(valid_exg,[])
                        # eog_data = seg[0,:] - seg[8,:]
                        hp_ft = butter(4, 15, 'lp', fs=sps,  output='sos')
                        eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                        for i in range(eog_data.shape[0]):
                            eog_data[i,:] = smooth_data(eog_data[i,:], 5)
                        valid_exg[:,start:] = reject_bad_segments(valid_exg[:,start:], sps, epoch_len=1,method="autoreject")
                    else:
                        eog_data = get_eog_channel(valid_exg,[])
                        # eog_data = seg[0,:] - seg[8,:]
                        hp_ft = butter(4, 15, 'lp', fs=sps,  output='sos')
                        eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                        for i in range(eog_data.shape[0]):
                            eog_data[i,:] = smooth_data(eog_data[i,:], 5)
                        valid_exg = reject_bad_segments(valid_exg, sps, method="autoreject")

                elif proc_param['eye_removal'] == "asr":
                    if data_type == "video":
                        eog_data = get_eog_channel(valid_exg,[])
                        hp_ft = butter(4, 15, 'lp', fs=sps,  output='sos') #https://www.researchgate.net/publication/226805296_Wavelet_based_compression_technique_of_Electro-oculogram_signals/figures?lo=1
                        eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                        for i in range(eog_data.shape[0]):
                            eog_data[i,:] = smooth_data(eog_data[i,:], 5)
                        valid_exg = eeg_optimization(valid_exg, 125)
                    else:
                        eog_data = get_eog_channel(valid_exg,[])
                        hp_ft = butter(4, 15, 'lp', fs=sps,  output='sos') #https://www.researchgate.net/publication/226805296_Wavelet_based_compression_technique_of_Electro-oculogram_signals/figures?lo=1
                        eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                        for i in range(eog_data.shape[0]):
                            eog_data[i,:] = smooth_data(eog_data[i,:], 5)
                        valid_exg = eeg_optimization(valid_exg, 125)
                    
                elif proc_param['eye_removal'] == "asr_asr_eog":
                    if data_type == "video":
                        valid_exg = eeg_optimization(valid_exg, 125)
                        eog_data = get_eog_channel(valid_exg,[])
                        hp_ft = butter(4, 15, 'lp', fs=sps,  output='sos')
                        eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                        for i in range(eog_data.shape[0]):
                            eog_data[i,:] = smooth_data(eog_data[i,:], 5)
                    else:
                        valid_exg = eeg_optimization(valid_exg, 125)
                        eog_data = get_eog_channel(valid_exg,[])
                        hp_ft = butter(4, 15, 'lp', fs=sps,  output='sos') #https://www.researchgate.net/publication/226805296_Wavelet_based_compression_technique_of_Electro-oculogram_signals/figures?lo=1
                        eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                        for i in range(eog_data.shape[0]):
                            eog_data[i,:] = smooth_data(eog_data[i,:], 5)

                if proc_param['interp_param']['if_interp']:
                    _, valid_exg, _ = fix_bad_channel_data(valid_exg, proc_param['interp_param']['win_size'],
                                                    proc_param['interp_param']['win_step'], proc_param['interp_param']['ratio'],
                                                    proc_param['interp_param']['sps'], proc_param['interp_param']['type'],
                                                    data_type=data_type, logger_file=logger_file)
                    eog_data = get_eog_channel(valid_exg,[])
                    hp_ft = butter(4, 15, 'lp', fs=sps,  output='sos') #https://www.researchgate.net/publication/226805296_Wavelet_based_compression_technique_of_Electro-oculogram_signals/figures?lo=1
                    eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                    for i in range(eog_data.shape[0]):
                        eog_data[i,:] = smooth_data(eog_data[i,:], 5)


                if proc_param['common_removal']['if_common_removal']:
                    if np.mean(np.abs(np.corrcoef(valid_exg))) > proc_param['common_removal']['corr_thr']:
                        # remove common interference for all channels
                        valid_exg = remove_common_interference(valid_exg, sps, corr_thr=proc_param['common_removal']['corr_thr'], seg=proc_param['common_removal']['seg'], logger_file=logger_file )
                        eog_data = get_eog_channel(valid_exg,[])
                        hp_ft = butter(4, 15, 'lp', fs=sps,  output='sos') #https://www.researchgate.net/publication/226805296_Wavelet_based_compression_technique_of_Electro-oculogram_signals/figures?lo=1
                        eog_data = sosfiltfilt(hp_ft, eog_data, axis=1)
                        for i in range(eog_data.shape[0]):
                            eog_data[i,:] = smooth_data(eog_data[i,:], 5)

                if exg_data.shape[1]/(exg_timestamps[-1]-exg_timestamps[0]) < 125:
                    print(f"Expected SPS: 125, real SPS: {exg_data.shape[1]/(exg_timestamps[-1]-exg_timestamps[0])}")
                
                if data_type == 'video':
                    task_data, task_et = split_tasks_exg(valid_exg, valid_et_sync_rs, sps, exg_timestamps[-1]-exg_timestamps[0], exg_timestamps)
                    task_eog, _ = split_tasks_exg(eog_data,valid_et_sync_rs, sps, exg_timestamps[-1]-exg_timestamps[0], exg_timestamps)
                else:
                    task_data = None
                    task_et = None
                    task_eog = None

                result.append({
                    'subject':subject,
                    'date': date,
                    'type' : data_type,
                    'raw_eeg' :valid_exg,
                    'raw_eog' : eog_data,
                    'task_eeg': task_data,
                    'task_eog': task_eog,
                    'task_et' : task_et,
                    'timestamp':exg_timestamps,
                    'eye_tracking': valid_et_sync_rs,
                    'eeg_std': np.std(valid_exg, axis=1),
                    'eog_std': np.std(eog_data, axis=1),
                    'eeg_mean': np.mean(valid_exg, axis=1),
                    'eog_mean': np.mean(eog_data, axis=1),
                    'eeg_std_all': np.std(valid_exg, axis=(0,1)),
                    'eog_std_all': np.std(eog_data, axis=(0,1)),
                    'eeg_mean_all': np.mean(valid_exg, axis=(0,1)),
                    'eog_mean_all': np.mean(eog_data, axis=(0,1)),
                })
                logger.info(f"Processed: {data_path}", file=logger_file)
    return result


def read_raw_data(root_path, intend_subject=None, proc_param=None, save_path=None, logger_file=None):
    """
    Read and process raw data for multiple subjects.
    
    Args:
        root_path: Root path to data
        intend_subject: List of subject IDs to process
        proc_param: Processing parameters
        save_path: Path to save processed data
        logger_file: File to write log messages
        
    Returns:
        List of processed data dictionaries
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    subjects = []
    if intend_subject is None:
        path = os.listdir(root_path)
        for p in path:
            if os.path.isdir(os.path.join(root_path,p)):
                subjects.append(p)

    elif isinstance(intend_subject,list) and isinstance(intend_subject[0], int):
        for s in intend_subject:
            subjects.append("%03d_patient"%s)
    else:
        assert("Wrong subject list!")

    data_ret = []
    for s in subjects:
        logger.info(f"Processing subject {s}",file=logger_file)
        try:
            subject_data = read_subject(root_path, s, proc_param=proc_param, logger_file=logger_file)
            data_ret.extend(subject_data)
        except Exception as e:
            logger.info(f"******* Error: {e} occur in {s} *************",file=logger_file)

        if proc_param['if_save']:
            for i in range(len(subject_data)):
                try:
                    item = subject_data[i]
                    subject, date, task_type = item['subject'], item['date'], item['type']
                    filename = os.path.join(save_path, f"{subject}-{date}-{task_type}")
                    np.savez(filename, 
                            subject=subject,
                            date=date,
                            type=task_type,
                            raw_eeg=item['raw_eeg'],
                            raw_eog=item['raw_eog'],
                            task_eeg=item['task_eeg'],
                            task_eog=item['task_eog'],
                            task_et=item['task_et'],
                            timestamp=item['timestamp'],
                            eye_tracking=item['eye_tracking'],
                            eeg_std=item['eeg_std'],
                            eog_std=item['eog_std'],
                            eeg_mean=item['eeg_mean'],
                            eog_mean=item['eog_mean'],
                            eeg_std_all=item['eeg_std_all'],
                            eog_std_all=item['eog_std_all'],
                            eeg_mean_all=item['eeg_mean_all'],
                            eog_mean_all=item['eog_mean_all'],
                            )
                    logger.info(f"Saved to {filename}",file=logger_file)
                except:
                    logger.info("Error!",file=logger_file)

    return data_ret


def slice_data(data_folder_path, proc_param=None, intend_subject=None, patient_info_xlsx=None, save_path=None, logger_file=None):
    """
    Slice EEG data into segments and add patient information.
    
    Args:
        data_folder_path: Path to data folder
        proc_param: Processing parameters
        intend_subject: List of subject IDs to process
        patient_info_xlsx: Path to patient information Excel file
        save_path: Path to save processed data
        logger_file: File to write log messages
        
    Returns:
        True if processing completed successfully
    """
    win_len = proc_param['slice_param']['win_len']
    win_step = proc_param['slice_param']['win_step']
    sps = proc_param['slice_param']['sps']
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    sub_name = []
    if intend_subject:
        if  isinstance(intend_subject[0], int):
            for s in intend_subject:
                sub_name.append("%03d_patient"%s)
        else:
            return
    unsliced_files = os.listdir(data_folder_path)

    patient_info = read_xlsx_to_dict(patient_info_xlsx)
    id = np.array(patient_info["id"])
    xlsx_date = patient_info["Date"]
    MoCA = patient_info["MoCA"]
    MMSE = patient_info["MMSE"]

    task_score = []
    for i in range(7):
        task_score.append(patient_info[f"Task{i+1}"])

    MMSE_task_score = []
    for i in range(6):
        MMSE_task_score.append(patient_info[f"MMSE_Task{i+1}"])  

    cnt = 0
    for file in unsliced_files:
        cnt += 1
        logger.info(f"Slicing {cnt}/{len(unsliced_files)}", file=logger_file)
        filepath = os.path.join(data_folder_path, file)
        if os.path.isfile(filepath):
            unsliced_data = np.load(filepath, allow_pickle=True)
            data_type = str(unsliced_data['type'])
            subject = str(unsliced_data['subject'])
            date = str(unsliced_data['date'])
            date_str = '.'.join(date.split('_')[:3])
            subject_idx_list = np.where(id == subject)[0]
            if len(subject_idx_list) > 0:
                for s_idx in subject_idx_list:
                    if date_str == xlsx_date[s_idx]:
                        subject_idx = s_idx
            else:
                subject_idx = subject_idx_list[0]

            if intend_subject is not None:
                if subject not in sub_name:
                    continue

            subject_moca = MoCA[subject_idx]
            if not isinstance(subject_moca,int):
                subject_moca = -1

            subject_mmse = MMSE[subject_idx]
            if not isinstance(subject_mmse,int):
                subject_mmse = -1

            subject_task_score = np.zeros((7), dtype=np.float32)
            for i in range(7):
                subject_task_score[i] = task_score[i][subject_idx] / MOCA_TASK_SCORE_MAX[i]
                if subject_task_score[i] < 0:
                    subject_task_score[i] = subject_moca / 30

            subject_MMSE_task_score = np.zeros((6), dtype=np.float32)
            for i in range(6):
                subject_MMSE_task_score[i] = MMSE_task_score[i][subject_idx] / MMSE_TASK_SCORE_MAX[i]
                if subject_MMSE_task_score[i] < 0:
                    subject_MMSE_task_score[i] = subject_mmse / 30

            if data_type == 'video':
                task_eeg = unsliced_data['task_eeg'].item()
                task_eog = unsliced_data['task_eog'].item()
                task_et = unsliced_data['task_et'].item()
                for i in range(9):
                    task_name = f"task{i+1}_pic"
                    single_task_eeg = task_eeg[task_name]
                    single_task_eog = task_eog[task_name]
                    single_task_et = task_et[task_name]
                    n_pics = len(single_task_eeg)

                    single_task_score = 0
                    for n_map in VIDEO_TASK_TO_MOCA_TASK_MAPPING[i]:
                        single_task_score += subject_task_score[n_map]
                    single_task_score /= len(VIDEO_TASK_TO_MOCA_TASK_MAPPING[i])
                    if single_task_score < 0:
                        single_task_score = subject_moca / 30

                    mmse_single_task_score = 0
                    for n_map in VIDEO_TASK_TO_MMSE_TASK_MAPPING[i]:
                        mmse_single_task_score += subject_MMSE_task_score[n_map]
                    mmse_single_task_score /= len(VIDEO_TASK_TO_MMSE_TASK_MAPPING[i])
                    if mmse_single_task_score < 0:
                        mmse_single_task_score = subject_mmse / 30

                    if single_task_score < 0 or single_task_score > 1 or mmse_single_task_score < 0 or mmse_single_task_score > 1:
                        if '010_patient' not in file:
                            logger.info(f"Error: file {file} single_task_score or mmse_single_task_score is out of range: {single_task_score} or {mmse_single_task_score}", file=logger_file)

                    for j in range(n_pics):
                        pics_eeg = single_task_eeg[j]
                        pics_eog = single_task_eog[j]
                        pics_et = single_task_et[j]
                        pics_len = pics_eeg.shape[1]
                        start, end = 0, win_len*sps
                        while end <= pics_len:
                            eeg_seg = pics_eeg[:, start:end]
                            eog_seg = pics_eog[:, start:end]
                            et_seg = pics_et[start:end]

                            if 'eeg_std' in unsliced_data.keys():
                                eeg_std = unsliced_data['eeg_std']
                                eog_std = unsliced_data['eog_std']
                                eeg_mean = unsliced_data['eeg_mean']
                                eog_mean = unsliced_data['eog_mean']
                                eeg_std_all = unsliced_data['eeg_std_all']
                                eog_std_all = unsliced_data['eog_std_all']
                                eeg_mean_all = unsliced_data['eeg_mean_all']
                                eog_mean_all = unsliced_data['eog_mean_all']
                            else:
                                eeg_std = None
                                eog_std = None
                                eeg_mean = None
                                eog_mean = None
                                eeg_std_all = None
                                eog_std_all = None
                                eeg_mean_all = None
                                eog_mean_all = None

                            file_name = f"{subject}-{date}-{data_type}-task{i}-pic{j}-{start}-{end}"
                            np.savez(os.path.join(save_path,file_name), 
                                    subject=subject,
                                    date=date,
                                    type=data_type,
                                    eeg_seg=eeg_seg,
                                    eog_seg=eog_seg,
                                    et_seg=et_seg,
                                    task_no=i,
                                    pic_no=j,
                                    moca=subject_moca,
                                    mmse=subject_mmse,
                                    eeg_std=eeg_std,
                                    eog_std=eog_std,
                                    eeg_mean=eeg_mean,
                                    eog_mean=eog_mean,
                                    eeg_std_all=eeg_std_all,
                                    eog_std_all=eog_std_all,
                                    eeg_mean_all=eeg_mean_all,
                                    eog_mean_all=eog_mean_all,
                                    moca_task_score=single_task_score,
                                    mmse_task_score=mmse_single_task_score
                                    )

                            start, end = start+win_step*sps, end+win_step*sps

            elif data_type == 'resting':
                raw_eeg = unsliced_data['raw_eeg']
                raw_eog = unsliced_data['raw_eog']
                raw_len = raw_eeg.shape[1]
                start, end = 0, win_len*sps
                single_task_score = subject_moca / 30
                mmse_single_task_score = subject_mmse / 30
                if single_task_score < 0 or mmse_single_task_score < 0:
                    if '010_patient' not in file:
                        logger.info(f"Error: file {file} single_task_score is out of range: {single_task_score}", file=logger_file)
                while end <= raw_len:
                    eeg_seg = raw_eeg[:, start:end]
                    eog_seg = raw_eog[:, start:end]
                    if 'eeg_std' in unsliced_data.keys():
                        eeg_std = unsliced_data['eeg_std']
                        eog_std = unsliced_data['eog_std']
                        eeg_mean = unsliced_data['eeg_mean']
                        eog_mean = unsliced_data['eog_mean']
                        eeg_std_all = unsliced_data['eeg_std_all']
                        eog_std_all = unsliced_data['eog_std_all']
                        eeg_mean_all = unsliced_data['eeg_mean_all']
                        eog_mean_all = unsliced_data['eog_mean_all']
                    else:
                        eeg_std = None  
                        eog_std = None
                        eeg_mean = None
                        eog_mean = None
                        eeg_std_all = None
                        eog_std_all = None
                        eeg_mean_all = None
                        eog_mean_all = None
                    file_name = f"{subject}-{date}-{data_type}-task9-pic0-{start}-{end}"
                    np.savez(os.path.join(save_path,file_name), 
                            subject=subject,
                            date=date,
                            type=data_type,
                            eeg_seg=eeg_seg,
                            eog_seg=eog_seg,
                            et_seg=None,
                            task_no=9,
                            pic_no=0,
                            moca=subject_moca,
                            mmse=subject_mmse,
                            eeg_std=eeg_std,
                            eog_std=eog_std,
                            eeg_mean=eeg_mean,
                            eog_mean=eog_mean,
                            eeg_std_all=eeg_std_all,
                            eog_std_all=eog_std_all,
                            eeg_mean_all=eeg_mean_all,
                            eog_mean_all=eog_mean_all,
                            moca_task_score=single_task_score,
                            mmse_task_score=mmse_single_task_score
                            )

                    start, end = start+win_step*sps, end+win_step*sps

    return True


def process_feature(sliced_feat_path, microstate_idx_path, microstate_mode='all',intend_subject=None, feature=None, save_path=None, logger_file=None):
    """
    Process features from sliced EEG data.
    
    Args:
        sliced_feat_path: Path to sliced feature data
        microstate_idx_path: Path to microstate indices
        microstate_mode: Mode for microstate processing ('all' or 'subject')
        intend_subject: List of subject IDs to process
        feature: List of features to process
        save_path: Path to save processed data
        logger_file: File to write log messages
        
    Returns:
        True if processing completed successfully
    """
    ##
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    sub_name = []
    if intend_subject:
        if  isinstance(intend_subject[0], int):
            for s in intend_subject:
                sub_name.append("%03d_patient"%s)
        else:
            return
        
    sliced_files = os.listdir(sliced_feat_path)

    cnt = 0
    for file in sliced_files:
        filepath = os.path.join(sliced_feat_path, file)
        if cnt % 100 == 0:
            logger.info(f"Calculating features {cnt}/{len(sliced_files)}", file=logger_file)
        cnt+=1
        if os.path.isfile(filepath):
            sliced_data = np.load(filepath, allow_pickle=True)
            data_type = str(sliced_data['type'])
            subject = str(sliced_data['subject'])
            date = str(sliced_data['date'])
            eeg_data = sliced_data['eeg_seg']
            task_no = int(sliced_data['task_no'])
            pic_no = int(sliced_data['pic_no'])

            if intend_subject is not None:
                if subject not in sub_name:
                    continue
            
            feature_generator = FeatureGenerator(eeg_data, 125)
            features = {'subject' : subject,
                        'date' : date,
                        'type' : data_type,
                        'task_no' : task_no,
                        'pic_no' : pic_no,
                        }
            for f in feature:
                if f == 'micro_states':
                    if microstate_mode == 'all':
                        micro_state_idx_filepath = os.path.join(microstate_idx_path, f"all.npz")
                    else:
                        micro_state_idx_filepath = os.path.join(microstate_idx_path, f"{subject}.npz")
                    micro_state_idx = np.load(micro_state_idx_filepath, allow_pickle=True)
                    centroid = micro_state_idx['centroid']
                    feat = feature_generator.calculate_feature(f)
                    feat = micro_state_mapping(feat, centroid)
                    features[f] = feat
                else:
                    if 'micro_states' in f:
                        continue
                    feat = feature_generator.calculate_feature(f)
                    features[f] = feat
            
            for f in feature:
                if f == 'micro_states_center':
                    features[f] = centroid
                if f == 'micro_states_occurrences':
                    # count the occurrences of each micro state
                    features[f] = np.zeros(20)
                    for i in range(20):
                        features[f][i] = np.sum(features['micro_states']==i)
                    
                if f == 'micro_states_transition':
                    # count the times of transition of each micro state
                    trans_cnt = 0
                    for i in range(len(features['micro_states'])-1):
                        if features['micro_states'][i] != features['micro_states'][i+1]:
                            trans_cnt += 1
                    features[f] = trans_cnt

                if f == 'micro_states_dist':
                    sum_dist = 0
                    for i in range(len(features['micro_states'])-1):
                        sum_dist += np.linalg.norm(centroid[features['micro_states'][i]]-centroid[features['micro_states'][i+1]])
                    features[f] = np.mean(sum_dist)

                if f == 'micro_states_entropy':
                    # calculate the entropy of the microstates sequence
                    features[f] = -np.sum(np.log2(np.bincount(features['micro_states']) / len(features['micro_states'])))

            file_str = file.split(".")[0]
            file_name = f"{file_str}-feat"
            np.savez(os.path.join(save_path,file_name), 
                                    **features,
                                    )

    return True


def pick_train_dataset_raw_features(raw_feat_path, raw_feat_train_path):
    """
    Select raw features for training dataset.
    
    Args:
        raw_feat_path: Path to raw feature data
        raw_feat_train_path: Path to save selected features
        
    Returns:
        True if processing completed successfully
    """
    if not os.path.exists(raw_feat_train_path):
        os.makedirs(raw_feat_train_path)
    incomplete_subjects = [1, 21, 53, 63]
    incomplete_subjects_str = ["%03d_patient"%s for s in incomplete_subjects]

    robustness_subjects = {16:'2024_12_13',     # except by clean data
                            17:'2024_12_10',        
                           68:'2025_01_13',
                             69:'2025_01_13', 
                             70:'2025_01_15', 
                             71:'2025_01_15',}
    
    robustness_subjects_str = ["%03d_patient"%s for s in robustness_subjects.keys()]
    robustness_subjects_date = robustness_subjects.values()
    new_robustness_dict = dict(zip(robustness_subjects_str, robustness_subjects_date))

    # copy all files in raw_feat_path to raw_feat_train_path
    raw_files = os.listdir(raw_feat_path)
    for file in raw_files:
        subject = file.split('-')[0]
        if subject in incomplete_subjects_str:
            continue
        elif subject in new_robustness_dict.keys():
            if new_robustness_dict[subject] not in file:
                continue
        shutil.copy(os.path.join(raw_feat_path, file), raw_feat_train_path)
    return True

def pick_perfect_data(raw_feat_train_path, selected_feat_train_path, include, exclude):
    """
    Select perfect data samples based on inclusion and exclusion criteria.
    
    Args:
        raw_feat_train_path: Path to raw feature data
        selected_feat_train_path: Path to save selected features
        include: List of subject-date-type combinations to include
        exclude: List of subject-date-type combinations to exclude
        
    Returns:
        None
    """
    if not os.path.exists(selected_feat_train_path):
        os.makedirs(selected_feat_train_path)

    if len(include) > 0:
        raw_files = os.listdir(raw_feat_train_path)
        for file in raw_files:
            unsliced_data = np.load(os.path.join(raw_feat_train_path, file), allow_pickle=True)
            subject = unsliced_data['subject']
            data_type = unsliced_data['type']
            date = unsliced_data['date']
            str_subject_date_type = f"{subject}-{date}-{data_type}"
            if str_subject_date_type in include:
                shutil.copy(os.path.join(raw_feat_train_path, file), selected_feat_train_path)
    else:
        raw_files = os.listdir(raw_feat_train_path)
        for file in raw_files:
            shutil.copy(os.path.join(raw_feat_train_path, file), selected_feat_train_path)

    if len(exclude) > 0:
        new_files = os.listdir(selected_feat_train_path)
        deleted_files = []
        for file in new_files:
            unsliced_data = np.load(os.path.join(selected_feat_train_path, file), allow_pickle=True)
            subject = str(unsliced_data['subject'])
            data_type = str(unsliced_data['type'])
            date = str(unsliced_data['date'])
            str_subject_date_type = f"{subject}-{date}-{data_type}"
            if str_subject_date_type in exclude:
                deleted_files.append(file)
                # also remove the file with the same subject
                for f in new_files:
                    if subject in f:
                        deleted_files.append(f)
        for f in deleted_files:
            try:
                os.remove(os.path.join(selected_feat_train_path, f))
            except:
                pass

def calculate_DTF_sliced_data(sliced_feat_train_path, intend_subject=None, logger_file=None):
    """
    Calculate Directed Transfer Function (DTF) for sliced data.
    
    Args:
        sliced_feat_train_path: Path to sliced feature data
        intend_subject: List of subject IDs to process
        logger_file: File to write log messages
        
    Returns:
        True if processing completed successfully
    """
    sliced_files = os.listdir(sliced_feat_train_path)
    cnt = 0
    none_cnt = 0
    sub_name = []
    if intend_subject:
        if  isinstance(intend_subject[0], int):
            for s in intend_subject:
                sub_name.append("%03d_patient"%s)
        else:
            return

    if len(sliced_files) > 10000:
        rep_freq = 1000
    else:
        rep_freq = 100
    for file in sliced_files:
        if cnt % rep_freq == 0:
            logger.info(f"Calculating DTF {cnt}/{len(sliced_files)}", file=logger_file)
        cnt+=1
        if os.path.isfile(os.path.join(sliced_feat_train_path, file)):
            if intend_subject:
                if file.split('-')[0] not in sub_name:
                    continue
            sliced_data = dict(np.load(os.path.join(sliced_feat_train_path, file), allow_pickle=True))
            if 'DTF' in sliced_data.keys():
                # print(f'{file} already has DTF')
                continue
            eeg_data = sliced_data['eeg_seg']
            DTF_data = cal_DTF(eeg_data.T, 2, 30, 4, 125)
            sliced_data['DTF'] = DTF_data
            np.savez(os.path.join(sliced_feat_train_path, file), **sliced_data)

    return True


def copy_sliced_feat(source_slice_feat_path, target_slice_feat_path, logger_file=None):
    """
    Copy sliced features from source to target directory.
    
    Args:
        source_slice_feat_path: Source directory path
        target_slice_feat_path: Target directory path
        logger_file: File to write log messages
        
    Returns:
        True if copying completed successfully
    """
    source_files = os.listdir(source_slice_feat_path)
    for file in source_files:
        shutil.copy(os.path.join(source_slice_feat_path, file), target_slice_feat_path)
    return True

def wrap_unslice_resting_data(raw_feat_train_path, intend_subject, resting_3min_feat_train_path, patient_info_xlsx, logger_file=None):
    """
    Process and wrap unsliced resting state data.
    
    Args:
        raw_feat_train_path: Path to raw feature data
        intend_subject: List of subject IDs to process
        resting_3min_feat_train_path: Path to save processed resting data
        patient_info_xlsx: Path to patient information Excel file
        logger_file: File to write log messages
        
    Returns:
        None
    """
    raw_files = os.listdir(raw_feat_train_path)

    os.makedirs(resting_3min_feat_train_path, exist_ok=True)
    sub_name = []
    if intend_subject:
        if  isinstance(intend_subject[0], int):
            for s in intend_subject:
                sub_name.append("%03d_patient"%s)
        else:
            return
        
    patient_info = read_xlsx_to_dict(patient_info_xlsx)
    id = np.array(patient_info["id"])
    xlsx_date = patient_info["Date"]
    MoCA = patient_info["MoCA"]
    MMSE = patient_info["MMSE"]

    task_score = []
    for i in range(7):
        task_score.append(patient_info[f"Task{i+1}"])

    MMSE_task_score = []
    for i in range(6):
        MMSE_task_score.append(patient_info[f"MMSE_Task{i+1}"])  

    for file in raw_files:
        if 'resting' not in file:
            continue
        raw_data = np.load(os.path.join(raw_feat_train_path, file), allow_pickle=True)
        subject = str(raw_data['subject'])
        date = str(raw_data['date'])
        data_type = str(raw_data['type'])
        date_str = '.'.join(date.split('_')[:3])
        subject_idx_list = np.where(id == subject)[0]
        if len(subject_idx_list) > 0:
            for s_idx in subject_idx_list:
                if date_str == xlsx_date[s_idx]:
                    subject_idx = s_idx
        else:
            subject_idx = subject_idx_list[0]

        if intend_subject is not None:
            if subject not in sub_name:
                continue

        subject_moca = MoCA[subject_idx]
        if not isinstance(subject_moca,int):
            subject_moca = -1

        subject_mmse = MMSE[subject_idx]
        if not isinstance(subject_mmse,int):
            subject_mmse = -1

        subject_task_score = np.zeros((7), dtype=np.float32)
        for i in range(7):
            subject_task_score[i] = task_score[i][subject_idx] / MOCA_TASK_SCORE_MAX[i]
            if subject_task_score[i] < 0:
                subject_task_score[i] = subject_moca / 30

        subject_MMSE_task_score = np.zeros((6), dtype=np.float32)
        for i in range(6):
            subject_MMSE_task_score[i] = MMSE_task_score[i][subject_idx] / MMSE_TASK_SCORE_MAX[i]
            if subject_MMSE_task_score[i] < 0:
                subject_MMSE_task_score[i] = subject_mmse / 30

        raw_eeg = raw_data['raw_eeg']
        raw_eog = raw_data['raw_eog']
        single_task_score = subject_moca / 30
        mmse_single_task_score = subject_mmse / 30
        if single_task_score < 0 or mmse_single_task_score < 0:
            if '010_patient' not in file:
                logger.info(f"Error: file {file} single_task_score is out of range: {single_task_score}", file=logger_file)
        eeg_seg = raw_eeg
        eog_seg = raw_eog
        if 'eeg_std' in raw_data.keys():
            eeg_std = raw_data['eeg_std']
            eog_std = raw_data['eog_std']
            eeg_mean = raw_data['eeg_mean']
            eog_mean = raw_data['eog_mean']
            eeg_std_all = raw_data['eeg_std_all']
            eog_std_all = raw_data['eog_std_all']
            eeg_mean_all = raw_data['eeg_mean_all']
            eog_mean_all = raw_data['eog_mean_all']
        else:
            eeg_std = np.std(raw_eeg, axis=1)
            eog_std = np.std(raw_eog, axis=1)
            eeg_mean = np.mean(raw_eeg, axis=1)
            eog_mean = np.mean(raw_eog, axis=1)
            eeg_std_all = np.std(raw_eeg, axis=(0,1))
            eog_std_all = np.std(raw_eog, axis=(0,1))
            eeg_mean_all = np.mean(raw_eeg, axis=(0,1))
            eog_mean_all = np.mean(raw_eog, axis=(0,1))

        file_name = f"{subject}-{date}-{data_type}-task9-pic0"
        np.savez(os.path.join(resting_3min_feat_train_path, file_name), 
                subject=subject,
                date=date,
                type=data_type,
                eeg_seg=eeg_seg,
                eog_seg=eog_seg,
                et_seg=None,
                task_no=9,
                pic_no=0,
                moca=subject_moca,
                mmse=subject_mmse,
                eeg_std=eeg_std,
                eog_std=eog_std,
                eeg_mean=eeg_mean,
                eog_mean=eog_mean,
                eeg_std_all=eeg_std_all,
                eog_std_all=eog_std_all,
                eeg_mean_all=eeg_mean_all,
                eog_mean_all=eog_mean_all,
                moca_task_score=single_task_score,
                mmse_task_score=mmse_single_task_score
                )

