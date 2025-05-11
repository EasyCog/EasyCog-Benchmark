"""
EEG Data Processing Pipeline

This module implements a comprehensive pipeline for processing EEG data, including:
- Raw data reading and preprocessing
- Eye artifact removal
- Signal filtering and interpolation
- Feature extraction
- Data slicing and segmentation
- Microstate analysis
- Directed Transfer Function (DTF) calculation

The pipeline supports both video task and resting state EEG data processing,
with various preprocessing options and feature extraction methods.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# Local imports
from data_processing.process_utils import *
from utils import logger

############ USER ENTRY ###############
MODE = 'Feat_Generation'    # 'Feat_Generation', 'Feat_Generation_3min', 'Raw_processing'
patient_info_xlsx = 'path/to/Patient_Info_dataset.xlsx'
feat_folder = f'path/to/dataset'
feat_name = 'asreog_filter_order3_all_data'


raw_data_path = 'path/to/raw_data'
############################

logger_file = None
sliced_feat_path = f'{feat_folder}/sliced/{feat_name}'
unsliced_feat_path = f'{feat_folder}/unsliced/{feat_name}'
micro_state_idx_path = f'{feat_folder}/micro_state_idx/{feat_name}'
processed_feat_path = f'{feat_folder}/stat_feat/{feat_name}'


if MODE == 'Feat_Generation':
    proc_param = {
        'feature_param': {
            'feature': FEATURE_NAMES,
        },
    }
    logger_file = None
    logger.info(f"***************************\nProcessing {feat_name}\n***************************", file=logger_file)
    logger.info(f"Param: {proc_param}", file=logger_file)
    ret = calculate_DTF_sliced_data(sliced_feat_path, logger_file=logger_file)
    ret = prepare_micro_states_idx_by_all(sliced_feat_path, intend_subject=None, save_path=micro_state_idx_path, logger_file=logger_file)
    ret = process_feature(sliced_feat_path, micro_state_idx_path,intend_subject=None, feature=proc_param['feature_param']['feature'], save_path=processed_feat_path, logger_file=logger_file)
    calculate_pca_feat(sliced_feat_path, processed_feat_path)
    stack_feature(processed_feat_path, intend_subject=None, feature=proc_param['feature_param']['feature'], logger_file=logger_file)
elif MODE == 'Feat_Generation_3min':
    resting_3min_feat_train_path = f'{feat_folder}/resting_3min_feat_train/{feat_name}'
    ret = wrap_unslice_resting_data(unsliced_feat_path, intend_subject=None, resting_3min_feat_train_path=resting_3min_feat_train_path, patient_info_xlsx=patient_info_xlsx, logger_file=logger_file)
    ret = calculate_DTF_sliced_data(resting_3min_feat_train_path, logger_file=logger_file)
elif MODE == 'Raw_processing':
    logger_file = None
    raw_feat_path = f'{feat_folder}/raw_feat/{feat_name}'
    raw_feat_train_path = f'{feat_folder}/raw_feat_train/{feat_name}'
    selected_feat_train_path = f'{feat_folder}/selected_raw_feat_train/{feat_name}'
    sliced_feat_path = f'{feat_folder}/sliced_feat/{feat_name}'
    sliced_feat_train_path = f'{feat_folder}/sliced_feat_train/{feat_name}'
    micro_state_idx_path = f'{feat_folder}/micro_state_idx/{feat_name}'
    processed_sliced_feat_path = f'{feat_folder}/processed_sliced_feat/{feat_name}'
    processed_sliced_feat_train_path = f'{feat_folder}/processed_sliced_feat_train/{feat_name}'

    proc_param = {
        'eye_removal': 'asr_asr_eog',   # default: asr_asr_eog
        'if_resample_et': True,         # default: True
        'if_save': True,                # default: True
        'cancel_prefix': 2,             # default: 2
        'filter_param': {
            'order': 3, 
            'low_freq': 0.1,
        },
        'if_notch': False,      
        'interp_param': {
            'if_interp': False,
            'win_size': 3,
            'win_step': 0.3,
            'ratio': 0.0,
            'sps': 125,
            'type': 'all',  # 'all', 'spatial', 'temporal'
        },
        'common_removal': {
            'if_common_removal': False,
            'corr_thr': 0.6,
            'seg': 1,
        },
        'slice_param': {
            'win_len': 3,
            'win_step': 1,
            'sps': 125,
        },
        'feature_param': {
            'feature': FEATURE_NAMES,
        },
        'include': [],
        'exclude': [],
        'if_avgref': False
    }

    logger_file = None
    logger.info(f"***************************\nProcessing {feat_name}\n***************************", file=logger_file)
    logger.info(f"Param: {proc_param}", file=logger_file)
    ret = read_raw_data(raw_data_path, None, proc_param=proc_param, save_path=raw_feat_path, logger_file=logger_file)
    ret = pick_train_dataset_raw_features(raw_feat_path, raw_feat_train_path)
    ret = pick_perfect_data(raw_feat_train_path, selected_feat_train_path, proc_param['include'], proc_param['exclude'])
    ret = slice_data(raw_feat_train_path, intend_subject=None, proc_param=proc_param, patient_info_xlsx=patient_info_xlsx, save_path=sliced_feat_train_path, logger_file=logger_file)
    if_update_norm=False
    if proc_param['interp_param']['if_interp']:
        if proc_param['interp_param']['type'] == 'all' or proc_param['interp_param']['type'] == 'temporal':
            # ret = fix_burst_channels_segments(sliced_feat_train_path, proc_param=proc_param, save_path=sliced_feat_train_path, logger_file=logger_file)
            ret = fix_burst_channels_segments(sliced_feat_train_path, intend_subject=None, proc_param=proc_param, save_path=sliced_feat_train_path, logger_file=logger_file)
            if_update_norm=True
    if proc_param['if_avgref']:
        ret = avgref_data(sliced_feat_train_path, save_path=sliced_feat_train_path, logger_file=logger_file)
        if_update_norm=True
    if_update_norm = True
    if if_update_norm:
        ret = update_norm_param(raw_feat_train_path, sliced_feat_train_path, save_path=sliced_feat_train_path, logger_file=None)
    ret = calculate_DTF_sliced_data(sliced_feat_train_path, logger_file=logger_file)
    ret = prepare_micro_states_idx_by_all(sliced_feat_train_path, intend_subject=None, save_path=micro_state_idx_path, logger_file=logger_file)
    ret = process_feature(sliced_feat_train_path, micro_state_idx_path,intend_subject=None, feature=proc_param['feature_param']['feature'], save_path=processed_sliced_feat_train_path, logger_file=logger_file)
    calculate_pca_feat(sliced_feat_train_path, processed_sliced_feat_train_path)
    stack_feature(processed_sliced_feat_train_path, intend_subject=None, feature=proc_param['feature_param']['feature'], logger_file=logger_file)

