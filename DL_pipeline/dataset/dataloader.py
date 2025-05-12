### dataloader
"""
items include: 
    - EEG: 16 channels x L 
    - EOG: 2 channels x L
    - Gaze position: 2 channels x L
    - MoCA: 1
    - MMSE: 1
"""
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import logger, convert_sample_to_format, normalize
import os
import random
import json
from DL_pipeline.dataset.data_aug import data_augmentation
from utils.data_preproc import preprocess_sample
from utils.utils import aggregate_features_within_task
from utils.subject_split import get_moca_category

class EasyCog_Dataset(Dataset):
    def __init__(
        self,
        sliced_data_json_file,
        trial_indices,
        data_aug_methods,
        input_type_list=["EEG", "EOG", "Gaze_posi"],
        input_format_dict={
            "EEG": ["sequence"],
            "EOG": ["sequence"],
            "Gaze_posi": ["sequence"],
        },
        gt_type_list=["MoCA", "MMSE", "Task_id"],
        gt_format_dict={"MoCA": ["value"], "MMSE": ["value"], "Task_id": ["value"]},
        norm_params={
            "EEG": {
                "sequence": {
                    "norm_type": "norm_by_mean_std",
                    "max_value": None,
                    "min_value": None,
                    "mean_value": -1,
                    "std_value": -1,
                    "preproc_method": "raw",    
                }
            },
            "EOG": {
                "sequence": {
                    "norm_type": "norm_by_mean_std",
                    "max_value": None,
                    "min_value": None,
                    "mean_value": -1,
                    "std_value": -1,
                    "preproc_method": "raw",    
                }
            },
            "Gaze_posi": {
                "sequence": {
                    "norm_type": "norm_by_mean_std",
                    "max_value": None,
                    "min_value": None,
                    "mean_value": -1,
                    "std_value": -1,
                    "preproc_method": "raw",    
                }
            },
        },
        name="train_set",
        logger_file=None,
        user_list=None
    ):
        """
        Args:
            data_folder (str): Path to the folder containing the data files
            trial_indices (list): List of trial IDs to include in the dataset: [[subject, resting/video, task_id, pic_id]]
            input_type_list (list): List of input modalities to use, can include "EEG", "EOG", "Gaze_posi"
            gt_type_list (list): List of ground truth labels to predict, can include "MoCA", "MMSE", "Task_id"
            norm_params (dict): Dictionary containing normalization parameters for each modality
                Structure:
                {
                    modality: {
                        format: {
                            'norm_type': str, # Type of normalization to apply
                            'max_value': float, # Maximum value for normalization
                            'min_value': float, # Minimum value for normalization
                            'mean_value': float, # Mean value for normalization
                            'std_value': float, # Standard deviation for normalization
                        }
                    }
                }
            name (str): Name of the dataset (e.g. "train_set", "val_set", "test_set")
            logger_file (str): Path to log file for logging dataset information
        """
        f = open(sliced_data_json_file)
        self.sliced_data_list = json.load(f)
        self.trial_indices = trial_indices
        self.input_type_list = input_type_list
        self.input_format_dict = input_format_dict
        self.gt_type_list = gt_type_list
        self.gt_format_dict = gt_format_dict
        if self.gt_type_list is not None:
            self.data_type_list = self.input_type_list + self.gt_type_list
        else:
            self.data_type_list = self.input_type_list
        self.logger_file = logger_file
        self.norm_params = norm_params
        self.user_list = user_list

        unique_subject_ids = sorted(list(set(self.user_list)))
        self.subject_id_to_class_index = {original_id: index for index, original_id in enumerate(unique_subject_ids)}
        self.class_index_to_subject_id = {index: original_id for index, original_id in enumerate(unique_subject_ids)}
        
        for modality in self.norm_params:
            for format in self.norm_params[modality]:
                if 'preproc_method' not in self.norm_params[modality][format]:
                    # default preproc_method is 'raw'
                    self.norm_params[modality][format]['preproc_method'] = 'raw'
        logger.info(f'{name} norm_params: {self.norm_params}')
        self.data_aug = data_aug_methods
        self.name = name
        if 'Task_embed' in self.data_type_list:
            self.task_embeddings = dict(np.load("/home/mmWave_group/EasyCog/Task_embeddings.npz", allow_pickle=True))
            
        print("{} size is: {}".format(name, self.__len__()))

    def set_name(self, name):
        self.name = name

    def __len__(self):
        return len(self.trial_indices)

    def load_single_data(self, data_info):
        """
        data_info is a dict from the json file
        """
        data_dict = {}
        tmp = np.load(os.path.join(data_info["root"], data_info["file"]), allow_pickle=True)
        for indx, data_type in enumerate(self.data_type_list):
            if "Forehead_EEG" in data_type:
                indices = [4, 5, 12, 13]
                data_dict[data_type] = tmp["eeg_seg"][indices]
            elif "EEG" in data_type:
                data_dict[data_type] = tmp["eeg_seg"]
                if self.norm_params[data_type]["sequence"]["norm_type"] == "norm_by_subject_task":
                    eeg_mean, eeg_std = data_info['eeg_mean_all'], data_info['eeg_std_all']
                    data_dict[data_type] = (tmp["eeg_seg"] - eeg_mean) / eeg_std
            elif "EOG" in data_type:
                data_dict[data_type] = tmp["eog_seg"]
            elif "Gaze_posi" in data_type:
                if tmp['et_seg'].dtype == 'O':
                    data_dict[data_type] = np.zeros((2, 375))
                else:
                    data_dict[data_type] = tmp["et_seg"].transpose(1, 0)
                    data_dict[data_type] = tmp["et_seg"].transpose(1, 0)
            elif "MoCA" in data_type and "Task_Score" not in data_type:
                data_dict[data_type] = int(data_info["MoCA"])
            elif "MMSE" in data_type and "Task_Score" not in data_type:
                data_dict[data_type] = int(data_info["MMSE"])
            elif "Subject_id" in data_type:
                # data_dict[data_type] = int(data_info["subject"].split("_")[0])
                data_dict[data_type] = self.subject_id_to_class_index[data_info["subject"]]
            elif "Subject_Category" in data_type:
                
                data_dict[data_type] = get_moca_category(data_info["MoCA"])
            elif "Task_id" in data_type:
                data_dict[data_type] = int(data_info["task_no"].split("task")[1])
            elif "Pic_id" in data_type:
                data_dict[data_type] = int(data_info["pic_no"].split("pic")[1])
            elif "Task_score" in data_type and 'MoCA' not in data_type:
                data_dict[data_type] = float(data_info["task_score"])
            elif "MoCA_Task_Score" in data_type:
                data_dict[data_type] = float(data_info["moca_task_score"])
            elif "MMSE_Task_Score" in data_type:
                data_dict[data_type] = float(data_info["mmse_task_score"])
            elif 'Task_embed' in data_type:
                data_dict[data_type] = self.task_embeddings[f"{data_info['data_type']}_{data_info['task_no']}_{data_info['pic_no']}"]
            elif 'STFT' in data_type:
                feat_file = np.load(os.path.join(data_info["feat_file"]), allow_pickle=True)
                data_dict[data_type] = np.abs(feat_file["stft_spectrum"])
            elif 'PCA' in data_type:
                feat_file = np.load(os.path.join(data_info["feat_file"]), allow_pickle=True)
                data_dict[data_type] = feat_file["pca_reduced_features"]
            elif 'Raw_feat' in data_type:
                # load all features and stack to [16, xxx]
                feat_file = np.load(os.path.join(data_info["feat_file"]), allow_pickle=True)
                data_dict[data_type] = feat_file["stacked_feat"]
            elif 'DTF' in data_type:
                data_dict[data_type] = tmp['DTF']
        return data_dict

    def __getitem__(self, idx):
        data_info = self.sliced_data_list[str(self.trial_indices[idx])]
        data_dict = self.load_single_data(data_info)

        input_sample_dict, gt_sample_dict = {}, {}
        for dt in self.input_type_list:
            sample = data_dict[dt]
            for format in self.input_format_dict[dt]:
                c_sample = convert_sample_to_format(sample, dt, format)
                c_sample = normalize(c_sample, dt, format, self.norm_params[dt][format])
                c_sample = preprocess_sample(c_sample, dt, format, method=self.norm_params[dt][format]['preproc_method'])
                c_sample = torch.tensor(np.array(c_sample), dtype=torch.float32).contiguous().clone()
                input_sample_dict[f"{dt}-{format}"] = c_sample

        for gt in self.gt_type_list:
            sample = data_dict[gt]
            for format in self.gt_format_dict[gt]:
                c_sample = convert_sample_to_format(sample, gt, format)
                c_sample = normalize(c_sample, gt, format, self.norm_params[gt][format])
                if gt == "Task_id":
                    c_sample = torch.tensor(np.array(c_sample), dtype=torch.int64).contiguous().clone()
                else:
                    c_sample = torch.tensor(np.array(c_sample), dtype=torch.float32).contiguous().clone()
                gt_sample_dict[f"{gt}-{format}"] = c_sample
        
        if 'train' in self.name:
            input_sample_dict, gt_sample_dict = data_augmentation(input_sample_dict, gt_sample_dict, self.data_aug, 'train')
        else:
            input_sample_dict, gt_sample_dict = data_augmentation(input_sample_dict, gt_sample_dict, self.data_aug, 'test')
        
        return input_sample_dict, gt_sample_dict


class EasyCog_Dataloader(DataLoader):
    def __init__(self, sliced_data_json_file,
         train_trials=None, valid_trials=None, test_trials=None,
        batch_size=64,
        data_aug_methods=None,
        input_type_list=["EEG", "EOG", "Gaze_posi"],
        input_format_dict={
            "EEG": ["sequence"],
            "EOG": ["sequence"],
            "Gaze_posi": ["sequence"],
        },
        gt_type_list=["MoCA", "MMSE", "Task_id"],
        gt_format_dict={"MoCA": ["value"], "MMSE": ["value"], "Task_id": ["value"]},
        norm_params={
            "EEG": {
                "sequence": {
                    "norm_type": "norm_by_mean_std",
                    "max_value": None,
                    "min_value": None,
                    "mean_value": -1,
                    "std_value": -1,
                }
            },
            "EOG": {
                "sequence": {
                    "norm_type": "norm_by_mean_std",
                    "max_value": None,
                    "min_value": None,
                    "mean_value": -1,
                    "std_value": -1,
                }
            },
            "Gaze_posi": {
                "sequence": {
                    "norm_type": "norm_by_mean_std",
                    "max_value": None,
                    "min_value": None,
                    "mean_value": -1,
                    "std_value": -1,
                }
            },
        }, 
        logger_file='test.out',
        num_workers = 0,
        persistent_workers=True,
        prefetch_factor=2,
        train_user_list=None,
        test_user_list=None,
        valid_user_list=None,
        ):

        self.batch_size = batch_size
        self.train_set = None
        self.test_set = None
        self.valid_set = None
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.prefetch_factor=prefetch_factor
        
        logger.info('Use EasyCog_Dataloader')

        if train_trials is not None:
            self.train_set = EasyCog_Dataset(
                sliced_data_json_file=sliced_data_json_file,
                trial_indices=train_trials,
                data_aug_methods=data_aug_methods,
                input_type_list=input_type_list,
                input_format_dict=input_format_dict,
                gt_type_list=gt_type_list,
                gt_format_dict=gt_format_dict,
                norm_params=norm_params,
                name="train_set",
                logger_file=logger_file,
                user_list=train_user_list
            )
        
        if valid_trials is not None:
            self.valid_set = EasyCog_Dataset(
                sliced_data_json_file=sliced_data_json_file,
                trial_indices=valid_trials,
                data_aug_methods=data_aug_methods,
                input_type_list=input_type_list,
                input_format_dict=input_format_dict,
                gt_type_list=gt_type_list,
                gt_format_dict=gt_format_dict,
                norm_params=norm_params,
                name="valid_set",
                logger_file=logger_file,
                user_list=valid_user_list
            )
        
        if test_trials is not None:
            self.test_set = EasyCog_Dataset(
                sliced_data_json_file=sliced_data_json_file,
                trial_indices=test_trials,
                data_aug_methods=data_aug_methods,
                input_type_list=input_type_list,
                input_format_dict=input_format_dict,
                gt_type_list=gt_type_list,
                gt_format_dict=gt_format_dict,
                norm_params=norm_params,
                name="test_set",
                logger_file=logger_file,
                user_list=test_user_list
            )
    
    def __call__(self):
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        if self.train_set is not None:
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, 
                                           persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor, drop_last=True)
        if self.valid_set is not None:
            self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor)
        if self.test_set is not None:
            self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor)
        return self.train_loader, self.valid_loader, self.test_loader

class EasyCog_StatFeat_Dict_Dataset(Dataset):
    def __init__(
        self,
        subject_task_indices, 
        target_user_list,
        data_aug_methods,
        input_type_list=["EEG"],
        input_format_dict={
            "EEG": ["sequence"],
        },
        gt_type_list=["MoCA", "MMSE"],
        gt_format_dict={"MoCA": ["value"], "MMSE": ["value"]},
        norm_params={
            "EEG": {
                "sequence": {
                    "norm_type": "norm_by_mean_std",
                    "max_value": None,
                    "min_value": None,
                    "mean_value": -1,
                    "std_value": -1,
                    "preproc_method": "raw",    
                }
            }
        },
        name="train_set",
        logger_file=None,
    ):
        """
        Args:
            data_folder (str): Path to the folder containing the data files
            trial_indices (list): List of trial IDs to include in the dataset: [[subject, resting/video, task_id, pic_id]]
            input_type_list (list): List of input modalities to use, can include "EEG", "EOG", "Gaze_posi"
            gt_type_list (list): List of ground truth labels to predict, can include "MoCA", "MMSE", "Task_id"
            norm_params (dict): Dictionary containing normalization parameters for each modality
                Structure:
                {
                    modality: {
                        format: {
                            'norm_type': str, # Type of normalization to apply
                            'max_value': float, # Maximum value for normalization
                            'min_value': float, # Minimum value for normalization
                            'mean_value': float, # Mean value for normalization
                            'std_value': float, # Standard deviation for normalization
                        }
                    }
                }
            name (str): Name of the dataset (e.g. "train_set", "val_set", "test_set")
            logger_file (str): Path to log file for logging dataset information
        """
        self.subject_task_indices = subject_task_indices
        self.target_user_list = target_user_list
        self.input_type_list = input_type_list
        self.input_format_dict = input_format_dict
        self.gt_type_list = gt_type_list
        self.gt_format_dict = gt_format_dict
        if self.gt_type_list is not None:
            self.data_type_list = self.input_type_list + self.gt_type_list
        else:
            self.data_type_list = self.input_type_list
        self.logger_file = logger_file
        self.norm_params = norm_params
        for modality in self.norm_params:
            for format in self.norm_params[modality]:
                if 'preproc_method' not in self.norm_params[modality][format]:
                    # default preproc_method is 'raw'
                    self.norm_params[modality][format]['preproc_method'] = 'raw'
        logger.info(f'{name} norm_params: {self.norm_params}')
        self.data_aug = data_aug_methods
        self.name = name

    def set_name(self, name):
        self.name = name

    def __len__(self):
        return len(self.target_user_list)

    def __getitem__(self, idx):
        subject_str = self.target_user_list[idx]
        input_sample_dict, gt_sample_dict = {}, {}
        for dt in self.input_type_list:
            if dt == 'EEG_StatFeat':
                sample = []
                for it in range(10):
                    str_task = f'task{it}'
                    feat_all = self.subject_task_indices[subject_str]['session_session1'][str_task]['feat_all']
                    feat_len_array = self.subject_task_indices[subject_str]['session_session1'][str_task]['feat_len_array']
                    self.norm_params[dt]['sequence']['feat_len_array'] = feat_len_array
                    sample.append(np.array(aggregate_features_within_task(np.array(feat_all), method=self.norm_params[dt]['sequence']['preproc_method'])))
                sample = np.array(sample)
            for format in self.input_format_dict[dt]:
                c_sample = convert_sample_to_format(sample, dt, format)
                c_sample = normalize(c_sample, dt, format, self.norm_params[dt][format])
                c_sample = preprocess_sample(c_sample, dt, format, method=self.norm_params[dt][format]['preproc_method'])
                c_sample = torch.tensor(np.array(c_sample), dtype=torch.float32).contiguous().clone()
                input_sample_dict[f"{dt}-{format}"] = c_sample.squeeze()

        for gt in self.gt_type_list:
            if gt == 'MoCA':
                sample = self.subject_task_indices[subject_str]['session_session1']['cognitive_scores']['MoCA']
            elif gt == 'MMSE':
                sample = self.subject_task_indices[subject_str]['session_session1']['cognitive_scores']['MMSE']
            elif gt == 'MoCA_taskscore':
                sample = self.subject_task_indices[subject_str]['session_session1']['moca_task_score']
            elif gt == 'MMSE_taskscore':
                sample = self.subject_task_indices[subject_str]['session_session1']['mmse_task_score']
            for format in self.gt_format_dict[gt]:
                c_sample = convert_sample_to_format(sample, gt, format)
                c_sample = normalize(c_sample, gt, format, self.norm_params[gt][format])
                if gt == "Task_id":
                    c_sample = torch.tensor(np.array(c_sample), dtype=torch.int64).contiguous().clone()
                else:
                    c_sample = torch.tensor(np.array(c_sample), dtype=torch.float32).contiguous().clone()
                gt_sample_dict[f"{gt}-{format}"] = c_sample

        if 'train' in self.name:
            input_sample_dict, gt_sample_dict = data_augmentation(input_sample_dict, gt_sample_dict, self.data_aug, 'train')
        else:
            input_sample_dict, gt_sample_dict = data_augmentation(input_sample_dict, gt_sample_dict, self.data_aug, 'test')
        
        return input_sample_dict, gt_sample_dict

class EasyCog_StatFeat_Dataset(Dataset):
    def __init__(
        self,
        data,       # （n_sample, n_task*task_feat_len)
        labels,     #  (2, n_sample), where is moca and mmse 
        subscores,  #  (2, n_sample, 7)
        data_aug_methods,
        input_type_list=["EEG"],
        input_format_dict={
            "EEG": ["sequence"],
        },
        gt_type_list=["MoCA", "MMSE"],
        gt_format_dict={"MoCA": ["value"], "MMSE": ["value"]},
        norm_params={
            "EEG": {
                "sequence": {
                    "norm_type": "norm_by_mean_std",
                    "max_value": None,
                    "min_value": None,
                    "mean_value": -1,
                    "std_value": -1,
                    "preproc_method": "raw",    
                }
            }
        },
        name="train_set",
        logger_file=None,
    ):
        """
        Args:
            data_folder (str): Path to the folder containing the data files
            trial_indices (list): List of trial IDs to include in the dataset: [[subject, resting/video, task_id, pic_id]]
            input_type_list (list): List of input modalities to use, can include "EEG", "EOG", "Gaze_posi"
            gt_type_list (list): List of ground truth labels to predict, can include "MoCA", "MMSE", "Task_id"
            norm_params (dict): Dictionary containing normalization parameters for each modality
                Structure:
                {
                    modality: {
                        format: {
                            'norm_type': str, # Type of normalization to apply
                            'max_value': float, # Maximum value for normalization
                            'min_value': float, # Minimum value for normalization
                            'mean_value': float, # Mean value for normalization
                            'std_value': float, # Standard deviation for normalization
                        }
                    }
                }
            name (str): Name of the dataset (e.g. "train_set", "val_set", "test_set")
            logger_file (str): Path to log file for logging dataset information
        """
        self.data = data
        self.label = labels
        self.task_score = subscores
        self.input_type_list = input_type_list
        self.input_format_dict = input_format_dict
        self.gt_type_list = gt_type_list
        self.gt_format_dict = gt_format_dict
        if self.gt_type_list is not None:
            self.data_type_list = self.input_type_list + self.gt_type_list
        else:
            self.data_type_list = self.input_type_list
        self.logger_file = logger_file
        self.norm_params = norm_params
        for modality in self.norm_params:
            for format in self.norm_params[modality]:
                if 'preproc_method' not in self.norm_params[modality][format]:
                    # default preproc_method is 'raw'
                    self.norm_params[modality][format]['preproc_method'] = 'raw'
        logger.info(f'{name} norm_params: {self.norm_params}')
        self.data_aug = data_aug_methods
        self.name = name

    def set_name(self, name):
        self.name = name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_dict = {}
        data_dict[self.input_type_list[0]] = self.data[idx,:]
        input_sample_dict, gt_sample_dict = {}, {}
        for dt in self.input_type_list:
            sample = data_dict[dt]
            for format in self.input_format_dict[dt]:
                c_sample = convert_sample_to_format(sample, dt, format)
                c_sample = normalize(c_sample, dt, format, self.norm_params[dt][format])
                c_sample = preprocess_sample(c_sample, dt, format, method=self.norm_params[dt][format]['preproc_method'])
                c_sample = torch.tensor(np.array(c_sample), dtype=torch.float32).contiguous().clone()
                input_sample_dict[f"{dt}-{format}"] = c_sample

        for gt in self.gt_type_list:
            if gt == 'MoCA':
                sample = self.label[0, idx]
            elif gt == 'MMSE':
                sample = self.label[1, idx]
            elif gt == 'MoCA_taskscore':
                sample = self.task_score[0][idx,:]
            elif gt == 'MMSE_taskscore':
                sample = self.task_score[1][idx,:],
            for format in self.gt_format_dict[gt]:
                c_sample = convert_sample_to_format(sample, gt, format)
                c_sample = normalize(c_sample, gt, format, self.norm_params[gt][format])
                if gt == "Task_id":
                    c_sample = torch.tensor(np.array(c_sample), dtype=torch.int64).contiguous().clone()
                else:
                    c_sample = torch.tensor(np.array(c_sample), dtype=torch.float32).contiguous().clone()
                gt_sample_dict[f"{gt}-{format}"] = c_sample

        if 'train' in self.name:
            input_sample_dict, gt_sample_dict = data_augmentation(input_sample_dict, gt_sample_dict, self.data_aug, 'train')
        else:
            input_sample_dict, gt_sample_dict = data_augmentation(input_sample_dict, gt_sample_dict, self.data_aug, 'test')
        
        return input_sample_dict, gt_sample_dict


class EasyCog_StatFeat_Dataloader(DataLoader):
    def __init__(self, sliced_data_json_file,
         train_trials=None, valid_trials=None, test_trials=None,
        batch_size=64,
        data_aug_methods=None,
        input_type_list=["EEG", "EOG", "Gaze_posi"],
        input_format_dict={
            "EEG": ["sequence"],
            "EOG": ["sequence"],
            "Gaze_posi": ["sequence"],
        },
        gt_type_list=["MoCA", "MMSE", "Task_id"],
        gt_format_dict={"MoCA": ["value"], "MMSE": ["value"], "Task_id": ["value"]},
        norm_params={
            "EEG": {
                "sequence": {
                    "norm_type": "norm_by_mean_std",
                    "max_value": None,
                    "min_value": None,
                    "mean_value": -1,
                    "std_value": -1,
                }
            },
            "EOG": {
                "sequence": {
                    "norm_type": "norm_by_mean_std",
                    "max_value": None,
                    "min_value": None,
                    "mean_value": -1,
                    "std_value": -1,
                }
            },
            "Gaze_posi": {
                "sequence": {
                    "norm_type": "norm_by_mean_std",
                    "max_value": None,
                    "min_value": None,
                    "mean_value": -1,
                    "std_value": -1,
                }
            },
        }, 
        logger_file='test.out',
        num_workers = 0,
        persistent_workers=True,
        prefetch_factor=2,
        ):

        self.batch_size = batch_size
        self.train_set = None
        self.test_set = None
        self.valid_set = None
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.prefetch_factor=prefetch_factor
        logger.info('Use EasyCog_Dataloader')

        if train_trials is not None:
            self.train_set = EasyCog_StatFeat_Dataset(
                sliced_data_json_file=sliced_data_json_file,
                trial_indices=train_trials,
                data_aug_methods=data_aug_methods,
                input_type_list=input_type_list,
                input_format_dict=input_format_dict,
                gt_type_list=gt_type_list,
                gt_format_dict=gt_format_dict,
                norm_params=norm_params,
                name="train_set",
                logger_file=logger_file,
            )
        
        if valid_trials is not None:
            self.valid_set = EasyCog_StatFeat_Dataset(
                sliced_data_json_file=sliced_data_json_file,
                trial_indices=valid_trials,
                data_aug_methods=data_aug_methods,
                input_type_list=input_type_list,
                input_format_dict=input_format_dict,
                gt_type_list=gt_type_list,
                gt_format_dict=gt_format_dict,
                norm_params=norm_params,
                name="valid_set",
                logger_file=logger_file,
            )
        
        if test_trials is not None:
            self.test_set = EasyCog_StatFeat_Dataset(
                sliced_data_json_file=sliced_data_json_file,
                trial_indices=test_trials,
                data_aug_methods=data_aug_methods,
                input_type_list=input_type_list,
                input_format_dict=input_format_dict,
                gt_type_list=gt_type_list,
                gt_format_dict=gt_format_dict,
                norm_params=norm_params,
                name="test_set",
                logger_file=logger_file,
            )
    
    def __call__(self):
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        if self.train_set is not None:
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor,
                                           drop_last=True)
        if self.valid_set is not None:
            self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor)
        if self.test_set is not None:
            self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor)
        return self.train_loader, self.valid_loader, self.test_loader



if __name__ == "__main__":

    data_aug_methods = {
        "train": {
            "EEG": {"random_noise": {"aug_times": 2}},
            "EOG": {"random_noise": {"aug_times": 2}},
            "Gaze_posi": {"random_noise": {"aug_times": 2}},
            "MoCA": {"no_aug": {"aug_times": 2}},
            "MMSE": {"no_aug": {"aug_times": 2}},
            "Task_id": {"no_aug": {"aug_times": 2}},
        },
        "test": {
            "EEG": {"no_aug": {"aug_times": 1}},
            "EOG": {"no_aug": {"aug_times": 1}},
            "Gaze_posi": {"no_aug": {"aug_times": 1}},
            "MoCA": {"no_aug": {"aug_times": 1}},
            "MMSE": {"no_aug": {"aug_times": 1}},
            "Task_id": {"no_aug": {"aug_times": 1}},
        },
    }

    norm_params = {
        "EEG": {
            "sequence": {
                "norm_type": "norm_by_mean_std",
                "max_value": None,
                "min_value": None,
                "mean_value": -1,
                "std_value": -1,
            }
        },
        "EOG": {
            "sequence": {
                "norm_type": "norm_by_mean_std",
                "max_value": None,
                "min_value": None,
                "mean_value": -1,
                "std_value": -1,
            }
        },
        "Gaze_posi": {
            "sequence": {
                "norm_type": "norm_by_mean_std",
                "max_value": None,
                "min_value": None,
                "mean_value": -1,
                "std_value": -1,
            }
        },
        "MoCA": {
            "value": {
                "norm_type": "norm_by_min_max",
                "max_value": 30,
                "min_value": 0,
                "mean_value": None,
                "std_value": None,
            }
        },
        "MMSE": {
            "value": {
                "norm_type": "norm_by_min_max",
                "max_value": 30,
                "min_value": 0,
                "mean_value": None,
                "std_value": None,
            }
        },
        "Task_id": {
            "value": {
                "norm_type": "no_norm",
                "max_value": None,
                "min_value": None,
                "mean_value": None,
                "std_value": None,
            }
        }
    }

    dataset = EasyCog_Dataset(
        sliced_data_json_file="/home/mmWave_group/EasyCog/data_json_files/no_eog_separation.json",
        trial_indices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        data_aug_methods=data_aug_methods,
        input_type_list=["EEG", "EOG", "Gaze_posi"],
        input_format_dict={
            "EEG": ["sequence"],
            "EOG": ["sequence"],
            "Gaze_posi": ["sequence"],
        },
        gt_type_list=["MoCA", "MMSE", "Task_id"],
        gt_format_dict={"MoCA": ["value"], "MMSE": ["value"], "Task_id": ["value"]},
        norm_params=norm_params,
        name="train_set",
        logger_file=None,
    )

    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])
    print(dataset[4])
    print(dataset[5])
    print(dataset[6])
    print(dataset[7])
    print(dataset[8])
    print(dataset[9])