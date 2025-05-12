from utils.utils import aggregate_features_within_task
import numpy as np
import torch
def preprocess_sample(sample, dt, format, method='raw'):
    if dt == 'EEG':
        proc_sample = EEG_preprocess(sample, format, method)
    
    elif dt == 'EOG':
        proc_sample = EOG_preprocess(sample, format, method)

    elif dt == 'Task_embed':
        proc_sample = Task_embed_preprocess(sample, format, method)

    elif dt == 'Gaze_posi':
        proc_sample = Gaze_posi_preprocess(sample, format, method)

    elif dt == 'EEG_StatFeat':
        proc_sample = sample
    
    elif dt == 'DTF':
        proc_sample = DTF_preprocess(sample, format, method)

    return proc_sample

def remove_common_features(feat):
    # sample: list of numpy array, each numpy array is a task with uncertain number of pictures and slices
    # return: list of numpy array, each numpy array is a task with uncertain number of pictures and slices
    sum_task_data = []
    for task_idx in range(len(feat)-1): #last one is resting
        for j in range(len(feat[task_idx])):
            if torch.is_tensor(feat[task_idx][j]):
                sum_task_data.append(feat[task_idx][j].unsqueeze(0))
            else:
                sum_task_data.append(feat[task_idx][j].reshape(1, -1))

    if torch.is_tensor(sum_task_data[0]):
        sum_task_data = torch.cat(sum_task_data, dim=0)
        mean_task_data = torch.mean(sum_task_data, dim=0, keepdims=True)
    else:
        sum_task_data = np.concatenate(sum_task_data, axis=0)
        mean_task_data = np.mean(sum_task_data, axis=0, keepdims=True)

    for task_idx in range(len(feat)-1):
        feat[task_idx] = feat[task_idx] - mean_task_data
    return feat

def task_preprocessing(sample, contrast_task):
    # sample: list of numpy array, each numpy array is a task with uncertain number of pictures and slices
    # contrast_task: list of task indices to be used for contrast normalization
    # return: list of numpy array, each numpy array is a task with uncertain number of pictures and slices
    new_sample = []
    if contrast_task is None:
        contrast_task = []
    for task_idx in range(len(sample)):
        task_data = sample[task_idx]
        if task_idx == 8:
            if task_data.shape[0] < 10:
                if torch.is_tensor(task_data):
                    task_data = torch.concatenate([task_data, task_data[:(10 - task_data.shape[0]),:]], dim=0)
                else:
                    task_data = np.concatenate([task_data, task_data[:(10 - task_data.shape[0]),:]], axis=0)
        task_data = aggregate_features_within_task(task_data, method='split-1')
        # task_data: [n_pics, n_dim]
        if task_idx in contrast_task:
            if task_idx == 2:
                pos_idx = [0, 2, 4, 6, 8]     # social
                neg_idx = [1, 3, 5, 7, 9]  # paired no social
                pos_data = task_data[pos_idx]
                neg_data = task_data[neg_idx]
                task_data = pos_data - neg_data  # become 5 pics!

            elif task_idx == 3:
                pos_idx = [1, 3, 5, 7, 9]   # normal
                neg_idx = [0, 2, 4, 6, 8]  # paired abnormal pictures
                pos_data = task_data[pos_idx]
                neg_data = task_data[neg_idx]
                task_data = pos_data - neg_data  # become 5 pics!

            elif task_idx == 6:
                pos_idx = [0, 1, 2, 5, 7]
                neg_idx = [3, 4, 6, 8, 9]   
                if torch.is_tensor(task_data[pos_idx]):
                    pos_data_mean = torch.mean(task_data[pos_idx], dim=0, keepdim=True)
                else:
                    pos_data_mean = np.mean(task_data[pos_idx], axis=0, keepdims=True)
                neg_data = task_data[neg_idx]
                task_data = pos_data_mean - neg_data  # become 5 pics!
        new_sample.append(task_data)
    return new_sample


def Task_embed_preprocess(sample, format, method='raw'):
    if format == 'last_token':
        if method == 'raw':
            return sample
    else:
        raise ValueError(f'Invalid format: {format}')


def EEG_preprocess(sample, format, method='raw'):
    if format == 'sequence':
        if method == 'raw':
            return sample
        elif method == 'avg-re-referencing':
            return sample - sample.mean(axis=0, keepdims=True)
    else:
        raise ValueError(f'Invalid method: {method}')



def EOG_preprocess(sample, format, method='raw'):
    if format == 'sequence':
        if method == 'raw':
            return sample
    else:
        raise ValueError(f'Invalid method: {method}')


def DTF_preprocess(sample, format, method='raw'):
    if format == 'value':
        if method == 'raw':
            return sample
    else:
        raise ValueError(f'Invalid method: {method}')

def Gaze_posi_preprocess(sample, format, method='raw'):
    if format == 'sequence':
        if method == 'raw':
            return sample
    else:
        raise ValueError(f'Invalid method: {method}')
