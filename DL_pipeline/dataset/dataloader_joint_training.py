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
from utils.utils import aggregate_features_within_task, intra_task_sample_indx_selection
from collections import defaultdict

class EasyCog_Joint_Training_Dataset(Dataset):
	def __init__(self, sliced_data_json_file,
			  	subject_task_indices, data_aug_methods, 
				 input_type_list=['features'], gt_type_list=['MoCA', 'MMSE'],
				 input_format_dict={'features': ['value']},
				 norm_params=None,
				 gt_format_dict={'MoCA': ['value'], 'MMSE': ['value']},
				 name='train_set',
				 user_list=None):
			
		f = open(sliced_data_json_file)
		self.sliced_data_list = json.load(f)
		self.input_type_list = input_type_list
		self.input_format_dict = input_format_dict
		self.gt_type_list = gt_type_list
		if self.gt_type_list is not None:
			self.data_type_list = self.input_type_list + self.gt_type_list
		else:
			self.data_type_list = self.input_type_list
		self.gt_format_dict = gt_format_dict
		self.norm_params = norm_params
		self.data_aug = data_aug_methods
		self.name = name
		self.user_list = user_list

		unique_subject_ids = sorted(list(set(self.user_list)))
		self.subject_id_to_class_index = {original_id: index for index, original_id in enumerate(unique_subject_ids)}
		self.class_index_to_subject_id = {index: original_id for index, original_id in enumerate(unique_subject_ids)}

		self.subject_task_indices, self.cognitive_scores_list = subject_task_indices[0], subject_task_indices[1]
		if 'Task_embed' in self.data_type_list:
			self.task_embeddings = dict(np.load("/home/mmWave_group/EasyCog/Task_embeddings.npz", allow_pickle=True))

	def load_single_data(self, data_indx, data_type='EEG'):
		data_info = self.sliced_data_list[str(data_indx)]
		tmp = np.load(os.path.join(data_info['root'], data_info['file']), allow_pickle=True)
		if data_type == 'EEG':
			data = tmp['eeg_seg']
			if self.norm_params[data_type]["sequence"]["norm_type"] == "norm_by_subject_task":
				# eeg_mean, eeg_std = np.expand_dims(tmp['eeg_mean'], axis=1), np.expand_dims(tmp['eeg_std'], axis=1)
				eeg_mean, eeg_std = data_info['eeg_mean_all'], data_info['eeg_std_all']
				data = (data - eeg_mean) / eeg_std
			return data
		elif 'Task_embed' in data_type:
			return self.task_embeddings[f"{data_info['data_type']}_{data_info['task_no']}_{data_info['pic_no']}"]
		raise ValueError(f"data_type {data_type} is not supported")


	def convert_task_indices_to_task_arrays(self, task_indices, data_type='EEG', format='sequence'):
		task_arrays = []
		for indices in task_indices:
			if data_type == 'EEG':
				task_ExG_array = np.zeros((len(indices), 16, 375))
				for ii, ind in enumerate(indices):
					data = convert_sample_to_format(self.load_single_data(ind, 'EEG'), 'EEG', format)
					task_ExG_array[ii] = normalize(data, 'EEG', format, self.norm_params['EEG'][format])
				task_arrays.append(task_ExG_array)
			elif data_type == 'Task_embed':
				if format == 'last_token':
					task_embeds = np.zeros((len(indices), 4096))
				else:
					task_embeds = np.zeros((len(indices), 636, 4096)) ### all tokens
				for ii, ind in enumerate(indices):
					data = convert_sample_to_format(self.load_single_data(ind, 'Task_embed'), 'Task_embed', format)
					task_embeds[ii] = normalize(data, 'Task_embed', format, self.norm_params['Task_embed'][format])
				task_arrays.append(task_embeds)
		return task_arrays
	


	def __len__(self):
		return len(self.subject_task_indices)
	
	def get_gt_sample(self, cognitive_scores, idx):
		gt_sample_dict = {}
		for gt in self.gt_type_list:
			if gt == 'MoCA':
				gt_sample = cognitive_scores[0]
			elif gt == 'MMSE':
				gt_sample = cognitive_scores[1]
			elif gt == 'MoCA_taskscore':
				gt_sample = cognitive_scores[2]
			elif gt == 'MMSE_taskscore':
				gt_sample = cognitive_scores[3]
			elif gt == 'Subject_id':
				gt_sample = self.subject_id_to_class_index[self.user_list[idx]]
			for format in self.gt_format_dict[gt]:
				c_sample = convert_sample_to_format(gt_sample, gt, format)
				c_sample = normalize(c_sample, gt, format, self.norm_params[gt][format])
				gt_sample_dict[f"{gt}-{format}"] = [c_sample]

		return gt_sample_dict
		
	
	def __getitem__(self, idx):
		### task_indices: indices of the subject with all tasks: [[task1_indices], [task2_indices], ...]
		task_indices = self.subject_task_indices[idx] #### indices of the subject with all tasks
		input_sample_dict, gt_sample_dict = {}, {}

		for data_type in self.input_type_list:
			for format in self.input_format_dict[data_type]:
				task_arrays = self.convert_task_indices_to_task_arrays(task_indices, data_type, format)
				# input_sample_dict[f"{data_type}-{format}"] = [task_arrays]
				input_sample_dict[f"{data_type}-{format}"] = task_arrays
		cognitive_scores = self.cognitive_scores_list[idx]
		
		# input_sample_dict['task_ExG_arrays'] = [ExG_task_arrays]
		gt_sample_dict = self.get_gt_sample(cognitive_scores, idx)
		
		### leave augmentation to the model
		### TODO: add augmentation on the EEG input data
		if 'train' in self.name:
			input_sample_dict, gt_sample_dict = data_augmentation(input_sample_dict, gt_sample_dict, self.data_aug, 'train')
		else:
			input_sample_dict, gt_sample_dict = data_augmentation(input_sample_dict, gt_sample_dict, self.data_aug, 'test')

		### single view currently
		return input_sample_dict, gt_sample_dict
			
# 自定义 collate_fn
def custom_collate_fn(batch):
	inputs = {}
	targets = {}
	input_keys = list(batch[0][0].keys())
	gt_keys = list(batch[0][1].keys())
	num_views = len(batch[0][0][input_keys[0]])

	for key in input_keys:
		if key not in inputs:
			inputs[key] = []
		for view_idx in range(num_views):
			view_list = []
			for sample in batch:
				view_list.append(sample[0][key][view_idx])
			inputs[key].append(view_list)

	for key in gt_keys:
		if key not in targets:
			targets[key] = []
		for view_idx in range(num_views):
			view_list = []
			for sample in batch:
				view_list.append(sample[1][key][view_idx])
			targets[key].append(view_list)

	return inputs, targets

class EasyCog_Joint_Training_Dataloader(DataLoader):
	def __init__(self, sliced_data_json_file, train_user_indices, valid_user_indices, test_user_indices, data_aug_methods, 
				batch_size=36, num_workers=0, persistent_workers=True, prefetch_factor=2,
			  	input_type_list=['features'], gt_type_list=['MoCA', 'MMSE'],
				input_format_dict={'features': ['value']},
				norm_params=None,
				gt_format_dict={'MoCA': ['value'], 'MMSE': ['value']},
				intra_task_agg_method='none', across_task_agg_method='none',
				logger_file='test.out',
				train_user_list=None,
				valid_user_list=None,
				test_user_list=None):
		self.batch_size = batch_size
		self.train_set = None
		self.test_set = None
		self.valid_set = None
		self.num_workers = num_workers
		self.persistent_workers = persistent_workers
		self.prefetch_factor=prefetch_factor
		logger.info('Use EasyCog_Joint_Training_Dataloader')
	
		if train_user_indices is not None:
			self.train_set = EasyCog_Joint_Training_Dataset(sliced_data_json_file, train_user_indices, data_aug_methods, 
				input_type_list=input_type_list, input_format_dict=input_format_dict, gt_type_list=gt_type_list, gt_format_dict=gt_format_dict,
				norm_params=norm_params, 
				name='train_set',
				user_list=train_user_list)
	
		if valid_user_indices is not None:
			self.valid_set = EasyCog_Joint_Training_Dataset(sliced_data_json_file, valid_user_indices, data_aug_methods, 
				input_type_list=input_type_list, input_format_dict=input_format_dict, gt_type_list=gt_type_list, gt_format_dict=gt_format_dict,
				norm_params=norm_params,
				name='valid_set',
				user_list=valid_user_list)
	
		if test_user_indices is not None:
			self.test_set = EasyCog_Joint_Training_Dataset(sliced_data_json_file, test_user_indices, data_aug_methods, 
				input_type_list=input_type_list, input_format_dict=input_format_dict, gt_type_list=gt_type_list, gt_format_dict=gt_format_dict,
				norm_params=norm_params, 
				name='test_set',
				user_list=test_user_list)
	
	def __call__(self):
		self.train_loader = None
		self.valid_loader = None
		self.test_loader = None
		if self.train_set is not None:
			self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, 
								  		persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor, collate_fn=custom_collate_fn)
		if self.valid_set is not None:
			self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, 
								  		persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor, collate_fn=custom_collate_fn)
		if self.test_set is not None:
			self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, 
								  		persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor, collate_fn=custom_collate_fn)
		return self.train_loader, self.valid_loader, self.test_loader
		