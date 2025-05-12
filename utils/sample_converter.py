from utils import logger

def convert_sample_to_format(sample, dt, format):
    
    if dt == 'EEG':
        converted_sample = convert_EEG_to_format(sample, format)
    
    elif dt == 'EOG':
        converted_sample = convert_EOG_to_format(sample, format)
    
    elif dt == 'Pic_id':
        converted_sample = convert_Pic_id_to_format(sample, format)

    elif dt == 'Task_embed':
        converted_sample = convert_Task_embed_to_format(sample, format)

    elif dt == 'DTF':
        converted_sample = convert_DTF_to_format(sample, format)

    else:
        converted_sample = sample
        
    return converted_sample


def convert_EEG_to_format(sample, format):
    # logger.info(f'EEG {format}: Just return the original sample')
    if format == 'sequence':
        return sample
    if 'chs' in format:
        num_chs = int(format.split('_')[-1].split('chs')[0])
        if num_chs == 10:
            ### sample: [16, 375]
            ### Right: R1 R3 R8 R10
            ### Left: L1 L3 L8 L10
            ### Forehead: FR1 FR4
            ### STD_CH = ['R1', 'R2', 'R3', 'FR1', 'FR2', 'R8', 'R9', 'R10', 'L1', 'L2', 'L3', 'FR4', 'FR3', 'L8', 'L9', 'L10']
            channel_indices = [0, 2, 3, 5, 7, 8, 10, 11, 13, 15]
            sample = sample[channel_indices, :]
        elif num_chs == 6:
            ### Right: R3 R8 
            ### Left:  L3 L8 
            ### Forehead: FR1 FR4
            ### STD_CH = ['R1', 'R2', 'R3', 'FR1', 'FR2', 'R8', 'R9', 'R10', 'L1', 'L2', 'L3', 'FR4', 'FR3', 'L8', 'L9', 'L10']
            channel_indices = [2, 3, 5, 10, 11, 13]
            sample = sample[channel_indices, :]
        else:
            raise ValueError(f'Invalid number of channels: {num_chs}')

    return sample

def convert_EOG_to_format(sample, format):
    # logger.info(f'EOG {format}: Just return the original sample')
    return sample

def convert_Gaze_posi_to_format(sample, format):
    # logger.info(f'Gaze_posi {format}: Just return the original sample')
    return sample

def convert_Pic_id_to_format(sample, format):
    # logger.info(f'Pic_id {format}: Just return the original sample')
    return sample

def convert_Task_embed_to_format(sample, format):
    if format == 'last_token':
        sample = sample[:, -1, :].squeeze()
    elif format == 'all_tokens':
        sample = sample
    else:
        raise ValueError(f'Invalid format: {format}')
    # logger.info(f'Task_embed {format}: Just return the original sample')
    return sample

def convert_DTF_to_format(sample, format):
    # logger.info(f'DTF {format}: Just return the original sample')
    return sample

