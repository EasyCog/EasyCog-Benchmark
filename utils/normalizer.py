import numpy as np

def normalize(sample, data_type, format, norm_params):
    '''
    norm_params: 
    {
        'norm_type': str, # Type of normalization to apply
        'max_value': float, # Maximum value for normalization
        'min_value': float, # Minimum value for normalization  
        'mean_value': float, # Mean value for normalization
        'std_value': float, # Standard deviation for normalization
    }
    '''
    if norm_params['norm_type'] == 'no_norm':
        return sample
    norm_type, max_value, min_value, mean_value, std_value = norm_params['norm_type'], norm_params['max_value'], norm_params['min_value'], norm_params['mean_value'], norm_params['std_value']
    if data_type == 'EEG_StatFeat':     
        if format == 'sequence':
            if norm_type == 'norm_by_min_max':
                # start = 0
                # for i in range(len(norm_params['feat_len_array'])):
                #     min_value = norm_params['feat_min'][i]
                #     max_value = norm_params['feat_max'][i]
                #     for j in range(len(sample)):
                #         for k in range(len(sample[j])):
                #             sample[j][k][start:start+norm_params['feat_len_array'][i]] = (sample[j][k][start:start+norm_params['feat_len_array'][i]] - min_value) / (max_value - min_value)
                #     start += norm_params['feat_len_array'][i]
                # return sample.reshape(-1)
                start = 0
                for i in range(len(norm_params['feat_len_array'])):
                    temp_sample = []
                    for j in range(len(sample-1)):
                        for k in range(len(sample[j])):
                            temp_sample.append(sample[j][k][start:start+norm_params['feat_len_array'][i]])
                    temp_sample = np.array(temp_sample)
                    min_value = np.min(temp_sample)
                    max_value = np.max(temp_sample)
                    for j in range(len(sample-1)):
                        for k in range(len(sample[j])):
                            sample[j][k][start:start+norm_params['feat_len_array'][i]] = (sample[j][k][start:start+norm_params['feat_len_array'][i]] - min_value) / (max_value - min_value)
                    start += norm_params['feat_len_array'][i]

                start = 0
                for i in range(len(norm_params['feat_len_array'])):
                    mean_value = norm_params['feat_mean'][i]
                    std_value = norm_params['feat_std'][i]
                    for j in range(len(sample)):
                        for k in range(len(sample[j])):
                            sample[j][k][start:start+norm_params['feat_len_array'][i]] = (sample[j][k][start:start+norm_params['feat_len_array'][i]] - mean_value) / std_value
                    start += norm_params['feat_len_array'][i]
                return sample.reshape(-1)
            elif norm_type == 'norm_by_mean_std':
                # start = 0
                # for i in range(len(norm_params['feat_len_array'])):
                #     mean_value = norm_params['feat_mean'][i]
                #     std_value = norm_params['feat_std'][i]
                #     for j in range(len(sample)):
                #         for k in range(len(sample[j])):
                #             sample[j][k][start:start+norm_params['feat_len_array'][i]] = (sample[j][k][start:start+norm_params['feat_len_array'][i]] - mean_value) / std_value
                #     start += norm_params['feat_len_array'][i]
                # return sample.reshape(-1)

                start = 0
                for i in range(len(norm_params['feat_len_array'])):
                    temp_sample = []
                    for j in range(len(sample)):
                        for k in range(len(sample[j])):
                            temp_sample.append(sample[j][k][start:start+norm_params['feat_len_array'][i]])      
                    temp_sample = np.array(temp_sample)
                    mean_value = np.mean(temp_sample)
                    std_value = np.std(temp_sample)
                    for j in range(len(sample)):
                        for k in range(len(sample[j])):
                            sample[j][k][start:start+norm_params['feat_len_array'][i]] = (sample[j][k][start:start+norm_params['feat_len_array'][i]] - mean_value) / std_value
                    start += norm_params['feat_len_array'][i]
                return sample.reshape(-1)

            elif norm_type == 'no_norm':
                sample = sample
                return sample
            else:
                raise NotImplementedError
        
    
    if norm_type == 'norm_by_min_max':
        if format == 'sequence':
            sample = (sample - min_value) / (max_value - min_value)
        elif format == 'value':
            sample = (sample - min_value) / (max_value - min_value)
    
    elif norm_type == 'norm_by_mean_std':
        if format == 'sequence':
            sample = (sample - mean_value) / std_value
        elif format == 'value':
            sample = (sample - mean_value) / std_value

    elif norm_type == 'no_norm':
        sample = sample

    elif norm_type == 'norm_by_subject_task':
        #### TODO: modify it. I currently implement it in the dataloading part, but I need to modify it to here. 
        sample = sample
    else:
        raise NotImplementedError
    return sample