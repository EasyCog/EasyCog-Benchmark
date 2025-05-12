import numpy as np
import json
import os
from tqdm import tqdm

def calculate_dataset_statistics(json_file):
    """
    Calculate statistics (min, max, mean, std) for all variables in the dataset.
    """
    # Load the dataset
    with open(json_file, 'r') as f:
        data_list = json.load(f)
    
    # Initialize statistics dictionary
    stats = {
        'EEG': {'sequence': {'min': float('inf'), 'max': float('-inf'), 'values': []}},
        'EOG': {'sequence': {'min': float('inf'), 'max': float('-inf'), 'values': []}},
        'Gaze_posi': {'sequence': {'min': float('inf'), 'max': float('-inf'), 'values': []}},
        'MoCA': {'value': {'min': float('inf'), 'max': float('-inf'), 'values': []}},
        'MMSE': {'value': {'min': float('inf'), 'max': float('-inf'), 'values': []}},
    }
    
    # Process each sample
    print("Calculating statistics...")
    for sample_id in tqdm(data_list):
        data_info = data_list[sample_id]
        data = np.load(os.path.join(data_info['root'], data_info['file']), allow_pickle=True)
        # Process EEG
        if 'eeg_seg' in data.keys():
            eeg_data = data['eeg_seg']
            if eeg_data is not None:
                # eeg_data = np.clip(eeg_data, -1000, 1000)
                stats['EEG']['sequence']['min'] = min(stats['EEG']['sequence']['min'], np.min(eeg_data))
                stats['EEG']['sequence']['max'] = max(stats['EEG']['sequence']['max'], np.max(eeg_data))
                stats['EEG']['sequence']['values'].extend(eeg_data.flatten())
        
        # Process EOG
        if 'eog_seg' in data.keys():
            eog_data = data['eog_seg']
            if eog_data.size == 1 and eog_data.item() is None:
                continue
            stats['EOG']['sequence']['min'] = min(stats['EOG']['sequence']['min'], np.min(eog_data))
            stats['EOG']['sequence']['max'] = max(stats['EOG']['sequence']['max'], np.max(eog_data))
            stats['EOG']['sequence']['values'].extend(eog_data.flatten())
        
        # Process Gaze position
        if 'et_seg' in data.keys():
            et_data = data['et_seg']
            if et_data.size == 1 and et_data.item() is None:
                continue
            stats['Gaze_posi']['sequence']['min'] = min(stats['Gaze_posi']['sequence']['min'], np.min(et_data))
            stats['Gaze_posi']['sequence']['max'] = max(stats['Gaze_posi']['sequence']['max'], np.max(et_data))
            stats['Gaze_posi']['sequence']['values'].extend(et_data.flatten())
        
        # Process MoCA
        if 'moca' in data.keys():
            moca_value = data['moca']
            if moca_value > -1: # valid value
                stats['MoCA']['value']['min'] = min(stats['MoCA']['value']['min'], moca_value)
                stats['MoCA']['value']['max'] = max(stats['MoCA']['value']['max'], moca_value)
                stats['MoCA']['value']['values'].append(moca_value)
            
        # Process MMSE
        if 'mmse' in data.keys():
            mmse_value = data['mmse']
            if mmse_value > -1: # valid value
                stats['MMSE']['value']['min'] = min(stats['MMSE']['value']['min'], mmse_value)
                stats['MMSE']['value']['max'] = max(stats['MMSE']['value']['max'], mmse_value)
                stats['MMSE']['value']['values'].append(mmse_value)
    
    # Calculate final statistics
    for modality in ['EEG', 'EOG', 'Gaze_posi']:
        values = np.array(stats[modality]['sequence']['values'])
        stats[modality]['sequence']['mean'] = np.mean(values)
        stats[modality]['sequence']['std'] = np.std(values)
        del stats[modality]['sequence']['values']  # Clean up to save memory
    
    for modality in ['MoCA', 'MMSE']:
        values = np.array(stats[modality]['value']['values'])
        stats[modality]['value']['mean'] = np.mean(values)
        stats[modality]['value']['std'] = np.std(values)
        del stats[modality]['value']['values']
    
    return stats 

def update_config_with_stats(config_file, stats):
    """
    Update the configuration file with calculated statistics
    
    Args:
        config_file (str): Path to the configuration YAML file
        stats (dict): Calculated statistics
    """
    import yaml
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update normalization parameters
    for modality in ['EEG', 'EOG', 'Gaze_posi']:
        if modality not in config['norm_params'].keys():
            config['norm_params'][modality] = {}
        config['norm_params'][modality]['sequence'].update({
            'max_value': float(stats[modality]['sequence']['max']),
            'min_value': float(stats[modality]['sequence']['min']),
            'mean_value': float(stats[modality]['sequence']['mean']),
            'std_value': float(stats[modality]['sequence']['std'])
        })
    
    for modality in ['MoCA', 'MMSE']:
        config['norm_params'][modality]['value'].update({
            'max_value': float(stats[modality]['value']['max']),
            'min_value': float(stats[modality]['value']['min']),
            'mean_value': float(stats[modality]['value']['mean']),
            'std_value': float(stats[modality]['value']['std'])
        })
    
    # Save updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
#     json_file = '/home/mmWave_group/EasyCog/data_json_files/no_eog_separation_v2.json'
#     stats = calculate_dataset_statistics(json_file)
#     print(stats)
#     update_config_with_stats('/home/mmWave_group/EasyCog/DL_pipeline/configs/cfg_eye_tracking_baseline.yaml', stats)

    # json_file = '/home/mmWave_group/EasyCog/data_json_files/ASR_resampleET.json'
    # stats = calculate_dataset_statistics(json_file)
    # print(stats)
    # update_config_with_stats('/home/mmWave_group/EasyCog/DL_pipeline/configs/cfg_ASR_exg_pretask_classification.yml', stats)

    # json_file = '/home/mmWave_group/EasyCog/data_json_files/train_json/ASR_ASR_EOG_resampleET_new.json'
    json_file = '/home/mmWave_group/EasyCog/data_json_files/proc_compare_json/asreog_filter_order3_clean_data_notch.json'
    stats = calculate_dataset_statistics(json_file)
    print(stats)

    
