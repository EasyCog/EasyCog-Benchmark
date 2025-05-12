import torch
import random
from utils.utils import aggregate_features_within_task
import numpy as np

def data_augmentation(input_dict, gt_dict, methods, mode='train', weight=1):
    """
        Augments the input data based on specified methods and times for different modalities.
        
        Args:
            input_dict (dict): The input data to augment, organized by modality.
            gt_dict (dict): The ground truth data, organized by modality.
            methods (dict): A dictionary where keys are modalities and values are lists of augmentation methods to apply.
            methods:{
            'train'/'test':{
                modality: {
                    method_name: {
                        aug_times: xxx,
                    }
                }
            }
            
        
        Returns:
            tuple: (augmented_data, augmented_gt) where each is a dict with modality keys
                   containing lists of augmented samples
    """
    augmented_data = {}
    augmented_gt = {}
    ratio = random.random()

    total_times = 1 # at least the no-aug version
    modality = list(input_dict.keys())[0].split('-')[0]
   
    for method_name, method_params in methods[mode][modality].items():
        total_times += int(method_params['aug_times'] * weight)
    IF_AUG = ratio > (1/total_times)
    # Apply augmentations to input data
    for modality_format in input_dict.keys():
        modality = modality_format.split('-')[0]
        if modality not in list(methods[mode].keys()):
            continue

        orig_data = input_dict[modality_format]
        augmented_data[modality] = []
        
        # Apply each augmentation method
        for method_name, method_params in methods[mode][modality].items():
            if method_name == 'mix_up':
                augmented_data[modality].append(apply_input_augmentation(orig_data, 'avg_intra_task_sample', method_params))
                continue
            aug_times = int(method_params['aug_times'] * weight)
            
            # Generate augmented samples
            for _ in range(aug_times):
                if IF_AUG:
                    aug_data = apply_input_augmentation(orig_data, method_name, method_params)
                ### no aug
                elif mode == 'test' and method_name == 'avg_intra_task_sample':
                    aug_data = apply_input_augmentation(orig_data, 'avg_intra_task_sample', method_params)
                elif mode == 'train' and modality == 'features':
                    ### default: no aug for extracted features
                    aug_data = apply_input_augmentation(orig_data, 'avg_intra_task_sample', method_params)
                else:
                    aug_data = orig_data
                augmented_data[modality].append(aug_data)
    
    # Apply augmentations to ground truth
    max_aug_len = max(len(aug_list) for aug_list in augmented_data.values())
    if gt_dict is None:
        return augmented_data, None
    for gt_format in gt_dict.keys():
        gt_type = gt_format.split('-')[0]
        if gt_type not in list(methods[mode].keys()):
            continue
            
        orig_gt = gt_dict[gt_format]
        augmented_gt[gt_type] = []
        
        # Apply each augmentation method
        for method_name, method_params in methods[mode][gt_type].items():
            if method_name == 'mix_up':
                augmented_gt[gt_type].append(orig_gt)
                continue
            # Generate augmented samples to match input data length
            while len(augmented_gt[gt_type]) < max_aug_len:
                if IF_AUG:
                    aug_gt = apply_gt_augmentation(orig_gt, method_name)
                else:
                    aug_gt = orig_gt
                augmented_gt[gt_type].append(aug_gt)

    return augmented_data, augmented_gt
     
def apply_input_augmentation(input_data, method_name, method_params):
    
    ### apply multiple methods in chain
    if "-" in method_name:
        methods = method_name.split("-")
        for method in methods:
            tmp = apply_input_augmentation(input_data, method, method_params)
            input_data = tmp
        return input_data

    if method_name == 'no_aug':
        return input_data
    
    elif method_name == 'random_noise':
        if 'amplitude' in method_params:
            return add_random_noise(input_data, method_params['amplitude'])
        else:
            return add_random_noise(input_data)
    elif method_name == 'pink_noise_subject_task_arrays':
        return add_pink_noise_subject_task_arrays(input_data, method_params['amplitude'])
    elif method_name == 'pink_noise':
        if len(input_data.shape) == 3:
            if 'amplitude' in method_params:
                return add_pink_noise_batch(input_data, method_params['amplitude'])
            else:
                return add_pink_noise_batch(input_data)
        else:
            if 'amplitude' in method_params:
                return add_pink_noise(input_data, method_params['amplitude'])
            else:
                return add_pink_noise(input_data)
    elif method_name == 'random_mask':
        if 'mask_ratio' in method_params:
            return add_random_mask(input_data, method_params['mask_ratio'])
        else:
            return add_random_mask(input_data)
    elif method_name == 'random_order':
        if isinstance(input_data, list):
            return random_order_sequence_dict(input_data, method_params['feat_len'], method_params['ratio'])
        else:
            return random_order_sequence(input_data, method_params['feat_len'], method_params['ratio'])
    
    elif method_name == 'random_task_mask':
        return random_task_mask_sequence(input_data, method_params['feat_len'], method_params['ratio'])

    elif method_name == 'random_order_numpy_array':
        return random_order_numpy_array(input_data, method_params['ratio'])
    
    elif method_name == 'random_region_re_reference':
        return random_region_re_reference(input_data)
    
    elif method_name == 'random_mask_sequence':
        return random_zero_mask_sequence(input_data, method_params['mask_ratio'])

    elif method_name == 'random_intra_task_sample':
        return random_intra_task_sample(input_data, method_params['sample_ratio'])

    elif method_name == 'avg_intra_task_sample':
        return avg_intra_task_sample(input_data)
    
    elif method_name == 'intra_task_pic_avg_sampe':
        return intra_task_pic_avg_sample(input_data, method_params['ratio'])
    
    else:
        raise ValueError(f"Unsupported augmentation method: {method_name}")

def apply_gt_augmentation(gt_data, method_name):
    if method_name == 'no_aug':
        return gt_data
    elif method_name == 'random_order':
        return gt_data
    else:
        raise ValueError(f"Unsupported augmentation method: {method_name}")
    
def apply_mixup_gt_augmentation(gt_data, batch_task_idx):
    return gt_data

def add_random_noise(input_data, amplitude=0.1):
    """
    Adds random noise to EEG data.
    
    Args:
        input_data (torch.Tensor): Input EEG data of shape (channels, samples)
        amplitude (float): Amplitude of the noise relative to the data
        
    Returns:
        torch.Tensor: EEG data with added random noise
    """
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    
    channels, samples = input_data.shape

    noise = torch.randn(channels, samples, device=input_data.device)

    noisy_data = input_data + amplitude * noise

    return noisy_data


def add_pink_noise(input_data, amplitude=0.1):
    """
    Adds pink noise to EEG data.
    
    Args:
        data (torch.Tensor): Input EEG data of shape (channels, samples)
        amplitude (float): Amplitude of the pink noise relative to the data
        
    Returns:
        torch.Tensor: EEG data with added pink noise
    """
   
    
    # Get the shape of the input data
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    
    channels, samples = input_data.shape
    
    # Generate white noise
    white = torch.randn(channels, samples, device=input_data.device)
    
    # Generate frequencies for the FFT
    freq = torch.fft.fftfreq(samples)
    
    # Convert to frequency domain
    white_fft = torch.fft.fft(white, dim=-1)
    
    # Create 1/f filter in frequency domain
    # Add small constant to avoid division by zero
    f_filter = 1 / (torch.abs(freq) + 1e-12)**0.5
    f_filter = f_filter.to(input_data.device)
    
    # Apply filter to create pink noise
    pink_fft = white_fft * f_filter[None, :]
    
    # Convert back to time domain
    pink = torch.fft.ifft(pink_fft, dim=-1).real
    
    # Normalize pink noise
    pink = pink * (torch.std(input_data) / torch.std(pink))
    
    # Add noise to the original signal
    noisy_data = input_data + amplitude * pink
    
    return noisy_data

def add_pink_noise_batch(input_data, amplitude=0.1):
    """
    Adds pink noise to batched EEG data.
    
    Args:
        input_data (torch.Tensor): Input EEG data of shape (batch_size, channels, samples)
        amplitude (float): Amplitude of the pink noise relative to the data
        
    Returns:
        torch.Tensor: EEG data with added pink noise
    """
    # 确保输入是torch张量
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    
    # 获取数据形状
    batch_size, channels, samples = input_data.shape
    
    # 生成白噪声
    white = torch.randn(batch_size, channels, samples, device=input_data.device)
    
    # 生成FFT的频率
    freq = torch.fft.fftfreq(samples)
    
    # 转换到频域 - 对于每个batch和每个通道
    white_fft = torch.fft.fft(white, dim=-1)
    
    # 创建1/f滤波器（频域）
    # 添加小常数避免除以零
    f_filter = 1 / (torch.abs(freq) + 1e-12)**0.5
    f_filter = f_filter.to(input_data.device)
    
    # 应用滤波器创建粉红噪声 - 为每个batch和channel应用相同的滤波器
    pink_fft = white_fft * f_filter[None, None, :]
    
    # 转回时域
    pink = torch.fft.ifft(pink_fft, dim=-1).real
    
    # 为每个batch样本单独标准化粉红噪声
    # 计算每个batch样本的标准差
    input_std = torch.std(input_data.reshape(batch_size, -1), dim=1).view(batch_size, 1, 1)
    pink_std = torch.std(pink.reshape(batch_size, -1), dim=1).view(batch_size, 1, 1)
    
    # 应用标准化
    pink_normalized = pink * (input_std / pink_std)
    
    # 添加噪声到原始信号
    noisy_data = input_data + amplitude * pink_normalized
    
    return noisy_data

def add_pink_noise_subject_task_arrays(input_data, amplitude=0.1):
    aug_data = []
    for task_data in input_data:
        aug_data.append(add_pink_noise_batch(task_data, amplitude))
    return aug_data


def add_random_mask(input_data, mask_ratio=0.2, mask_type='zero'):
    """
    Applies random masking to EEG data channels.
    
    Args:
        input_data (torch.*Tensor): Input EEG data of shape (channels, samples)
        mask_ratio (float): Ratio of channels to mask (0.0-1.0)
        mask_type (str): Type of masking ('zero', 'mean', or 'noise')
        
    Returns:
        torch.Tensor: EEG data with random channels masked
    """
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    
    channels, samples = input_data.shape # [16, 3*125]
    
    # Make a copy of the input data to avoid modifying the original
    masked_data = input_data.clone()
    
    # Determine number of channels to mask
    num_to_mask = max(1, int(channels * mask_ratio))
    
    # Randomly select channels to mask
    mask_indices = torch.randperm(channels)[:num_to_mask]
    
    # Apply masking based on the specified type
    if mask_type == 'zero':
        masked_data[mask_indices] = 0
    elif mask_type == 'mean':
        # Replace with channel mean
        for idx in mask_indices:
            channel_mean = torch.mean(input_data[idx])
            masked_data[idx] = channel_mean
    elif mask_type == 'noise':
        # Replace with random noise (scaled to match data distribution)
        for idx in mask_indices:
            channel_std = torch.std(input_data[idx])
            noise = torch.randn(samples, device=input_data.device) * channel_std * 0.1
            masked_data[idx] = noise
    else:
        raise ValueError(f"Unsupported mask type: {mask_type}")
    
    return masked_data

def test_time_data_augmentation():
    pass

def random_order_numpy_array(input_data, ratio=1):
    ### input_data: numpy array of shape (n_task, n_feat)
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    n_task = input_data.shape[0]

    # Calculate how many tasks to shuffle based on ratio
    n_rand_task = int(n_task * ratio)
    
    # Create a copy of the input data
    new_input = input_data.detach().clone()
    
    if n_rand_task > 0:
        # Select random tasks to shuffle
        tasks_to_shuffle = random.sample(range(n_task), n_rand_task)
        
        # Create a random permutation of the selected tasks
        shuffled_indices = tasks_to_shuffle.copy()
        random.shuffle(shuffled_indices)
        
        # Apply the shuffling
        for original_idx, new_idx in zip(tasks_to_shuffle, shuffled_indices):
            new_input[original_idx] = input_data[new_idx]
    
    return new_input


def random_order_sequence(input_data, feat_len, ratio=1):
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    n_task = (input_data.shape[-1]//feat_len)

    n_rand_task = int(n_task*ratio)
    rand_task = random.sample(range(n_task), n_rand_task)

    # random_order = torch.randperm(n_task)

    cnt = 0
    new_input = input_data.detach().clone()
    for i in range(n_task):
        if i in rand_task:
            idx = rand_task[cnt]
            cnt += 1
            new_input[..., i*feat_len:(i+1)*feat_len] = input_data[..., idx*feat_len:(idx+1)*feat_len]
    
    return new_input




def random_order_sequence_dict(input_data, feat_len, ratio=1):
    #input_data: [n_task, n_feat]
    n_task = len(input_data)

    n_rand_task = int(n_task*ratio)
    rand_task = random.sample(range(n_task), n_rand_task)

    # random_order = torch.randperm(n_task)

    cnt = 0
    # new_input = input_data.detach().clone()
    new_input = input_data.copy()
    for i in range(n_task):
        if i in rand_task:
            idx = rand_task[cnt]
            cnt += 1
            new_input[i] = input_data[idx]
    
    return new_input

def random_task_mask_sequence(input_data, feat_len, ratio=1):
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    n_task = (input_data.shape[-1]//feat_len)

    n_rand_task = int(n_task*ratio)
    rand_task = random.sample(range(n_task), n_rand_task)

    # random_order = torch.randperm(n_task)

    cnt = 0
    new_input = input_data.detach().clone()
    for i in range(n_task):
        if i in rand_task:
            idx = rand_task[cnt]
            cnt += 1
            new_input[..., i*feat_len:(i+1)*feat_len] = torch.zeros_like(new_input[..., i*feat_len:(i+1)*feat_len])
    
    return new_input




def random_region_re_reference(input_data):
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    
    ### [0, 1, 2, 5, 6, 7]: right ear
    ### [3, 4, 11 ,12]: forehead
    ### [8, 9, 10, 13, 14, 15]: left ear

    ### randomly select one of the three regions
    left_ear_channels = [8, 9, 10, 13, 14, 15]
    right_ear_channels = [0, 1, 2, 5, 6, 7]
    forehead_channels = [3, 4, 11 ,12]

    ### for each region, randomly select one channel as the reference channel
    left_ear_ref_channel = left_ear_channels[torch.randint(0, len(left_ear_channels), (1,))]
    right_ear_ref_channel = right_ear_channels[torch.randint(0, len(right_ear_channels), (1,))]
    forehead_ref_channel = forehead_channels[torch.randint(0, len(forehead_channels), (1,))]

    ### re-reference the selected region to the reference channel
    output_data = input_data.clone()
    output_data[left_ear_channels] = input_data[left_ear_channels] - input_data[left_ear_ref_channel]
    output_data[right_ear_channels] = input_data[right_ear_channels] - input_data[right_ear_ref_channel]
    output_data[forehead_channels] = input_data[forehead_channels] - input_data[forehead_ref_channel]


    output_data[left_ear_ref_channel] = input_data[left_ear_ref_channel]
    output_data[right_ear_ref_channel] = input_data[right_ear_ref_channel]
    output_data[forehead_ref_channel] = input_data[forehead_ref_channel]

    return output_data


def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask


def random_zero_mask_sequence(input_data, mask_ratio=0.2, mask_type='zero'):
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    
    channels, samples = input_data.shape # [16, 3*125]
    patch_length = 25 # 25 / 125 = 0.2 s
    patch_num = samples // patch_length
    
    input_data_patches = input_data.view(1, channels, patch_num, patch_length)

    mask = generate_mask(1, channels, patch_num, mask_ratio, input_data.device)

    mask_data = input_data_patches.clone()
    mask_data[mask == 1] = torch.zeros(patch_length, device=input_data.device, requires_grad=False)

    mask_data = mask_data.view(channels, samples)

    return mask_data
    
def random_intra_task_sample(input_data, sample_ratio=0.2):
    r = random.random()
    result = []
    for task in input_data:
        if r < 0.8:
            result.append(aggregate_features_within_task(task, method=f'random_sample_{sample_ratio}'))
        else:
            result.append(aggregate_features_within_task(task, method=f'mean'))

    # Handle both numpy arrays and torch tensors
    if isinstance(input_data[0], torch.Tensor):
        # If input is tensor, concatenate tensors
        result = torch.cat(result, dim=0)
        return result
    else:
        # For numpy arrays, use numpy concatenate and convert to tensor
        result = np.concatenate(result, axis=0)
        return torch.tensor(result, dtype=torch.float32)
    
def avg_intra_task_sample(input_data):
    result = []
    for task in input_data:
        result.append(aggregate_features_within_task(task, method=f'mean'))
    
    # Handle both numpy arrays and torch tensors
    if isinstance(input_data[0], torch.Tensor):
        # If input is tensor, concatenate tensors
        result = torch.cat(result, dim=0)
        return result
    else:
        # For numpy arrays, use numpy concatenate and convert to tensor
        result = np.concatenate(result, axis=0)
        return torch.tensor(result, dtype=torch.float32)

    
def intra_task_pic_avg_sample(input_data, ratio=1):
    result = []
    if isinstance(input_data, np.ndarray):
        for task in input_data:
            result.append(np.mean(aggregate_features_within_task(task, method=f'split-{ratio}')))
        result = np.concatenate(result, axis=0)
        return torch.tensor(result, dtype=torch.float32)
    elif isinstance(input_data, torch.Tensor):
        for task in input_data:
            result.append(torch.mean(aggregate_features_within_task(task, method=f'split-{ratio}')))
        result = torch.cat(result, dim=0)
        return result

def mixup_from_subject(input_data, gt_data, aug_times):
    # input_data: tensor of shape [n_augs, n_task, n_dim]
    # gt_data: list of modalities, each modality is a tensor of shape [n_augs, ...]
    mixup_data = torch.zeros(aug_times, input_data.shape[1], input_data.shape[2]).to(input_data.device)
    gt_shape = [list(gt_data[i].shape) for i in range(len(gt_data))]
    for i in range(len(gt_shape)):
        gt_shape[i][0] = aug_times
    mixup_gt = [torch.zeros(gt_shape[i]).to(input_data.device) for i in range(len(gt_data))]
    for i in range(aug_times):
        weight = torch.rand(1).to(input_data.device)
        rand_feat1 = torch.randint(0, input_data.shape[0], (1,)).to(input_data.device)
        rand_feat2 = torch.randint(0, input_data.shape[0], (1,)).to(input_data.device)
        mixup_data[i, ...] = weight * input_data[rand_feat1, ...] + (1 - weight) * input_data[rand_feat2, ...]
        for j in range(len(gt_data)):
            mixup_gt[j][i, ...] = weight * gt_data[j][rand_feat1, ...] + (1 - weight) * gt_data[j][rand_feat2, ...]
        
    return mixup_data, mixup_gt

def apply_mixup_augmentation(input_data, gt_data, aug_times, mode='uniform'):
    # input data: tensor of shape [B, n_augs, n_task, n_dim], and the list of length is the number of subjects
    # gt_data: a list of tensors, each tensor is a numpy array of shape [n_subject, n_augs, ...], and the list of length is the number of subjects
    n_subjects = input_data.shape[0]
    n_feats = input_data.shape[1]
    n_tasks = input_data.shape[2]
    n_dim = input_data.shape[3]

    for _ in range(aug_times):
        new_data = torch.zeros((n_subjects, 1, n_tasks, n_dim)).to(input_data.device)
        new_gt = [torch.zeros_like(gt_data[i][0,0]).unsqueeze(0).repeat(n_subjects, 1).to(input_data.device) for i in range(len(gt_data))]
        for subject in range(n_subjects):
            if mode == 'uniform':
                random_weight = torch.rand(1).to(input_data.device)
            elif mode == 'beta':
                random_weight = np.random.beta(0.5, 0.5)

            rand_subject1 = torch.randint(0, n_subjects, (1,)).to(input_data.device)
            rand_subject2 = torch.randint(0, n_subjects, (1,)).to(input_data.device)
            rand_augfeat1 = torch.randint(0, n_feats, (1,)).to(input_data.device)
            rand_augfeat2 = torch.randint(0, n_feats, (1,)).to(input_data.device)

            new_data[subject, 0, ...] = random_weight * input_data[rand_subject1, rand_augfeat1, ...] + (1 - random_weight) * input_data[rand_subject2, rand_augfeat2, ...]
            for i in range(len(gt_data)):
                new_gt[i][subject, ...] = random_weight * gt_data[i][rand_subject1, rand_augfeat1, ...] + (1 - random_weight) * gt_data[i][rand_subject2, rand_augfeat2, ...]
        input_data = torch.cat((input_data, new_data), dim=1)
        for i in range(len(gt_data)):
            if len(gt_data[i].shape) == 2:
                gt_data[i] = torch.cat((gt_data[i], new_gt[i]), dim=1)
            elif len(gt_data[i].shape) == 3:
                gt_data[i] = torch.cat((gt_data[i], new_gt[i].unsqueeze(1)), dim=1)
            
    return input_data, gt_data

if __name__ == '__main__':
    data = [0,1,2,3,4,5,6]
    print(data)
    print(random_order_sequence(data,1,0.4))