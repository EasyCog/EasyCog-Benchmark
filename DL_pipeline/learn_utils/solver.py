from collections import namedtuple
import torch
import torch.nn.functional as F
from utils import evaluate, AverageMeter, transpose_nested_list, transpose_aug_list
from utils import logger, generate_mask, resample, get_data_dict_list_idx, line_seg, get_data_dict_list_to_item
from utils.statistics import evaluate_c
from data_processing.analysis_utils import VIDEO_TASK_MAPPING_MMSE_MATRIX, VIDEO_TASK_MAPPING_MOCA_MATRIX, PINV_VIDEO_TASK_MAPPING_MMSE_MATRIX, PINV_VIDEO_TASK_MAPPING_MOCA_MATRIX
import os
import time
import numpy as np
from scipy.stats import linregress
import random
from DL_pipeline.learn_utils.EarlyStopping import EarlyStopping


__all__ = ['Trainer', 'Tester']

# Define result types for regression and classification tasks
RegressionResult = namedtuple('RegressionResult', ['epoch', 'loss', 'rho', 'mean_absolute_error', 'mae_ratio', 'var'], defaults=(None,) * 6)
ClassificationResult = namedtuple('ClassificationResult', ['epoch', 'loss', 'acc', 'f1', 'recall', 'precision'], defaults=(None,) * 6)
RSquareResult = namedtuple('RSquareResult', ['epoch', 'mae', 'loss', 'r2'], defaults=(None,) * 4)


class Trainer:
	def __init__(self, model, device, optimizer, criterion, scheduler, cfg=None, resume=None,
				 save_path='./checkpoints', print_freq=20, val_freq=10, test_freq=10, 
				 regression_model=None, regression_optimizer=None, regression_criterion=None,
				 regression_scheduler=None, regression_test_criterion=None, regression_save_path=None,
				 regression_print_freq=20,
				 reg_cfg=None,
				 adv_model=None, test_criterion=None):
		# Initialize basic components
		self.model = model
		self.device = device
		self.optimizer = optimizer
		self.criterion = criterion
		self.scheduler = scheduler
		self.adv_model = adv_model
		
		### Initialize regression components
		self.regression_model = regression_model ### the model for regression task
		self.regression_optimizer = regression_optimizer
		self.regression_criterion = regression_criterion
		self.regression_scheduler = regression_scheduler
		self.regression_test_criterion = regression_test_criterion
		self.regression_save_path = regression_save_path

		# File paths and frequencies
		self.resume_file = resume
		self.save_path = save_path
		os.makedirs(self.save_path, exist_ok=True)
		if self.regression_save_path is not None:
			os.makedirs(self.regression_save_path, exist_ok=True)
		self.print_freq = print_freq
		self.regression_print_freq = regression_print_freq
		self.val_freq = val_freq
		self.test_freq = test_freq
		
		# Training state
		self.cur_epoch = 1
		self.all_epoch = None
		self.train_loss = None
		self.val_loss = None
		self.test_loss = None

		self.train_loss_reg = None
		self.val_loss_reg = None
		self.test_loss_reg = None
		
		# Best results tracking
		
		self.best_acc = ClassificationResult()
		self.best_f1 = ClassificationResult()
		self.best_recall = ClassificationResult()
		self.best_precision = ClassificationResult()

		# Prepare for StatFeat Regression
		self.best_mae = RegressionResult()
		self.best_mae_ratio = RegressionResult()
		self.best_var = RegressionResult()
		self.best_mae2 = RegressionResult()
		self.best_mae_ratio2 = RegressionResult()
		self.best_var2 = RegressionResult()
		self.best_r2_moca = RSquareResult()
		self.best_r2_mmse = RSquareResult()



		# Best results tracking
		self.best_var_valid = RegressionResult()
		self.best_acc_valid = ClassificationResult()
		self.best_f1_valid = ClassificationResult()
		self.best_recall_valid = ClassificationResult()
		self.best_precision_valid = ClassificationResult()

		# Prepare for StatFeat Regression
		self.best_mae_valid = RegressionResult()
		self.best_mae_ratio_valid = RegressionResult()
		self.best_mae2_valid = RegressionResult()
		self.best_mae_ratio2_valid = RegressionResult()
		self.best_var2_valid = RegressionResult()
		self.best_r2_moca_valid = RSquareResult()
		self.best_r2_mmse_valid = RSquareResult()
		
		# Configuration
		self.cfg = cfg or {}
		self.reg_cfg = reg_cfg 
		self.record = os.path.join(self.save_path, self.cfg.get('record', 'record.txt'))
		if self.reg_cfg is not None:
			self.reg_record = os.path.join(self.regression_save_path, self.reg_cfg.get('record', 'record.txt'))
		self.task = self.cfg.get('task', 'regression')
		
		# Initialize tester
		self.tester = Tester(model, device, test_criterion, save_path=save_path, print_freq=print_freq, cfg=cfg)

		# Initialize tester
		if self.reg_cfg is not None:
			self.tester_reg = Tester(model, device, regression_test_criterion, save_path=regression_save_path, cfg=reg_cfg, 
									 print_freq=regression_print_freq)
		
		self.valid_loss_early_stopping = EarlyStopping(patience=5, verbose=True, delta=0)
		self.moca_early_stopping = EarlyStopping(patience=5, verbose=True, delta=0)
		self.mmse_early_stopping = EarlyStopping(patience=5, verbose=True, delta=0)
		logger.info(f"Model will be saved at: [{save_path}]", file=self.record)
		logger.info(f"Model configuration: [{cfg}]", file=self.record)
	

	
	def loop_joint_classification_regression_validation_saved(self, epochs, 
											cls_loaders=[], reg_loaders=[]):
		"""Joint classification and regression training loop"""
		self.all_epoch = epochs
		self._resume() ### TODO: add resume on multi-task and joint-training
		# train_loader, val_loader, test_loader = cls_loaders
		# train_loader_reg, val_loader_reg, test_loader_reg = reg_loaders
		train_loader, val_loader, test_loader = cls_loaders

		train_loader_reg, val_loader_reg, test_loader_reg = reg_loaders
		for ep in range(self.cur_epoch, self.all_epoch + 1):
			self.cur_epoch = ep

			logger.info(f"{line_seg} Training [classification] {line_seg}")
			self.train_loss = self.train(train_loader, mode='train_cls')
			logger.info(f"{line_seg} Training classification loss: {self.train_loss} {line_seg}")
		
			logger.info(f"{line_seg} Training [regression] {line_seg}")
			self.train_loss_reg = self.train(train_loader_reg, mode='train_reg')
			logger.info(f"{line_seg} Training regression loss: {self.train_loss_reg} {line_seg}")
			
			
			# Validation phase
			if ep % self.val_freq == 0 and val_loader:
				logger.info(f"{line_seg} Validating [classification] {line_seg}")
				self.task = self.cfg.get('task', 'regression')
				self.val_loss = self._run_tests(val_loader, mode='test_cls', saved=True, validation_save=True)
				logger.info(f"{line_seg} Validating [regression] {line_seg}")
				self.task = self.reg_cfg.get('task', 'regression')
				self.val_loss_reg, results = self._run_tests(val_loader_reg, mode='test_reg', saved=True, validation_save=True, return_results=True)
				moca_mae = results[1][0]
				mmse_mae = results[2][0]
				self.moca_early_stopping(moca_mae)
				self.mmse_early_stopping(mmse_mae)
   
			# Testing phase
			if ep % self.test_freq == 0:
				logger.info(f"{line_seg} Testing [classification] {line_seg}")
				self.task = self.cfg.get('task', 'regression')
				self.test_loss = self._run_tests(test_loader, mode='test_cls', saved=True, validation_save=False)
				logger.info(f"{line_seg} Testing [regression] {line_seg}")
				self.task = self.reg_cfg.get('task', 'regression')
				self.test_loss_reg = self._run_tests(test_loader_reg, mode='test_reg', saved=True, validation_save=False)
			
			if self.moca_early_stopping.early_stop and self.mmse_early_stopping.early_stop:
				logger.info(f"Early STOP at epoch {ep}")
				break

	def train(self, train_loader, target_loader=None, mode=None):
		"""Training iteration with support for different training modes"""
		self.model.train() ### reg and cls use the same model even in joint-training
		train_stage = self.cfg.get('train_stage', 'standard')
		

		if train_stage == 'mask_pretrain':
			return self._iterate_mask_pretrain(train_loader)
		else:
			return self._iterate(train_loader, target_loader=target_loader, is_training=True, mode=mode)

	def validate(self, val_loader):
		"""Validation iteration"""
		self.model.eval()
		with torch.no_grad():
			return self._iterate(val_loader, is_training=False)

	def _convert_data_dict_to_list(self, data_dict):
		"""Convert data dict to list"""
		keys = list(data_dict.keys())
		data_list = []
		for i in range(len(keys)):
			data_list.append(data_dict[keys[i]])
		return data_list

	def _iterate_mask_pretrain(self, data_loader):
		"""Mask pretrain iteration"""
		loss_meters = [AverageMeter(f'Loss{i}') for i in range(self.criterion.get_num_loss_items())]
		iter_time = AverageMeter('Time')
		
		for batch_idx, (inputs, targets) in enumerate(data_loader):
			time_start = time.time()
			
			# Convert data dict to list
			input_data = self._convert_data_dict_to_list(inputs)
			
			# Move data to device
			input_data = self._to_device(input_data)
			
			# transpose the nested list
			input_data = transpose_nested_list(input_data)
			
			# Forward pass
			# the model will handle the data **list**.

			loss_list = []
			for i in range(len(input_data)):    # aug times
				### outputs is a tuple
				### the first item is the classification output
				### the second item is the regression output if the output length > 1
				ExG_input = input_data[i][0]
				B, input_chs, seq_len = ExG_input.shape
				ExG_input = ExG_input.view(B, input_chs, 3, seq_len//3)
				ExG_input = resample(ExG_input, 200, axis=-1)
				mask = generate_mask(ExG_input.shape[0], ExG_input.shape[1], ExG_input.shape[2], self.cfg['mask_ratio'], self.device)
				outputs = self.model(ExG_input, mask=mask)
				masked_outputs = outputs[0][mask==1]
				masked_inputs = ExG_input[mask==1]
				loss = self.criterion(masked_outputs, masked_inputs)
				loss_list.append(loss)

			losses = [0] * len(loss_list)
			for loss in loss_list:
				for i, loss_item in enumerate(loss):
					losses[i] += loss_item
			losses = [loss / len(loss_list) for loss in losses]

			# Backward pass (training only)
			self.optimizer.zero_grad()
			### loss is a tuple. the first item is the total loss value.
			losses[0].backward()
			self.optimizer.step()
			self.scheduler.step()
				
			# Update metrics
			for i, loss_meter in enumerate(loss_meters):
				loss_meter.update(losses[i].item())

			iter_time.update(time.time() - time_start)
			
			# Print progress
			if (batch_idx + 1) % self.print_freq == 0:
				self._print_progress(batch_idx, len(data_loader), loss_meters, iter_time)
		
		return loss_meters[0].avg
	
	def _iterate(self, data_loader, target_loader=None, is_training=True, mode=None):
		"""Generic iteration function for both training and validation"""
		if mode is None or 'cls' in mode:
			loss_meters = [AverageMeter(f'Loss{i}') for i in range(self.criterion.get_num_loss_items())]
		elif 'reg' in mode:
			loss_meters = [AverageMeter(f'Loss{i}') for i in range(self.regression_criterion.get_num_loss_items())]
			if self.reg_cfg["DA"]["if_use"] is True:
				loss_meters.append(AverageMeter('Domain_Loss'))
				loss_meters.append(AverageMeter('Domain_Acc'))
		iter_time = AverageMeter('Time')
		
		for batch_idx, (inputs, targets) in enumerate(data_loader):
			time_start = time.time()
			
			# Convert data dict to list
			input_data = self._convert_data_dict_to_list(inputs)
			gt_data = self._convert_data_dict_to_list(targets)
			
			# Move data to device
			input_data = self._to_device(input_data)
			gt_data = self._to_device(gt_data)
			
			# transpose the nested list
			input_data = transpose_nested_list(input_data)
			gt_data = transpose_nested_list(gt_data) 
			
			# Forward pass
			# the model will handle the data **list**.
			if 'moco' in self.model._get_name():
				self.cfg['model']['contrast_method'] = self.cfg['model']['contrast_method'] if 'contrast_method' in self.cfg['model'].keys() else 'sample'
				if 'Joint_CLS_REG_Model' in self.model._get_name():
					if mode == 'train_cls':
						qk = input_data
						if 'label' in self.cfg['model']['contrast_method']:
							gt = get_data_dict_list_idx(targets, 0) ### assume the ground truth maintains the same for augmentation
							outputs = self.model(qk, gt, mode=mode)
							loss = self.criterion(outputs, Y=(gt_data[0][0]))
						else:
							outputs = self.model(qk, mode=mode)
							loss = self.criterion(outputs, Y=(gt_data[0][0]))
					elif 'reg' in mode:
						i = 0 # TODO: need to modify the index
						del(gt_data)
						gt_data = get_data_dict_list_idx(targets, i)
						gt_data = get_data_dict_list_to_item(gt_data)
						outputs, aug_gts, _ = self.model(input_data[i], gt_data, mode=mode) 
						### subject_feats: [B, [task1_feats, task2_feats, ...]]
						gt = aug_gts
						loss = self.regression_criterion(outputs, gt) ### regression loss


			else:
				for i in range(len(input_data)):    # aug times
					### outputs is a tuple
					### the first item is the classification output
					### the second item is the regression output if the output length > 1
					if 'Joint_CLS_REG_Model' in self.model._get_name():
						if mode == 'train_cls':
							outputs = self.model(input_data[i], mode=mode)
							gt = gt_data[i]
							loss = self.criterion(outputs, gt) ### classification loss
						elif 'reg' in mode: 
							# input_data[0][0] = [B, [n_tasks, [n_slice, n_ch, n_sample]]]
							gt_data = get_data_dict_list_idx(targets, i)
							gt_data = get_data_dict_list_to_item(gt_data)
							outputs, aug_gts = self.model(input_data[i], gt_data, mode=mode) 
							### subject_feats: [B, [task1_feats, task2_feats, ...]]
							gt = aug_gts
							loss = self.regression_criterion(outputs, gt) ### regression loss

								
					else:
						outputs = self.model(input_data[i])
						gt = gt_data[i]
						loss = self.criterion(outputs, gt)
					
			
			if is_training:
				if mode is None or 'cls' in mode:
					self.optimizer.zero_grad()
					### loss is a tuple. the first item is the total loss value.
					loss[0].backward()
					self.optimizer.step()
					self.scheduler.step()
				elif 'reg' in mode:
					self.regression_optimizer.zero_grad()
					loss[0].backward()
					self.regression_optimizer.step()
					self.regression_scheduler.step()
			
			# Update metrics
			for i, loss_meter in enumerate(loss_meters):
				loss_meter.update(loss[i].item())

			iter_time.update(time.time() - time_start)
			
			# Print progress
			if mode is None or 'cls' in mode:
				if (batch_idx + 1) % self.print_freq == 0:
					self._print_progress(batch_idx, len(data_loader), loss_meters, iter_time, mode=mode)
			elif 'reg' in mode:
				if (batch_idx + 1) % self.regression_print_freq == 0:
					self._print_progress(batch_idx, len(data_loader), loss_meters, iter_time, mode=mode)
		
		return loss_meters[0].avg

	
	def _run_tests(self, test_loader, mode=None, saved=True, validation_save=False, return_results=False):
		"""Run tests and save best models"""
		if mode is None or 'cls' in mode:
			results = self.tester(test_loader, mode=mode)
		elif 'reg' in mode:
			if self.reg_cfg is None:
				results = self.tester(test_loader, mode=mode)
			else:
				results = self.tester_reg(test_loader, mode=mode)
		
		if mode is None or 'cls' in mode:
			self.multi_tasks = self.cfg['multi-task'] if 'multi-task' in self.cfg.keys() else None
		elif 'reg' in mode:
			self.multi_tasks = self.reg_cfg['multi-task'] if 'multi-task' in self.reg_cfg.keys() else None
		
		if self.multi_tasks is not None:
			if saved:
				self._save_best_multi_tasks(results)
		else:
			if self.task == 'regression' and (mode is None or 'test_reg' in mode):
				if self.reg_cfg is None and (self.cfg['train_stage'] == 'mix_up' or self.cfg['train_stage'] == 'mix_up_v2' or self.cfg['train_stage'] == 'cog_regression'):
					if saved:
						self._save_best_statfeat_regression(results[0], results[1], results[2], results[3], results[4], validation_save=validation_save)
				elif self.reg_cfg is not None and 'reg' in mode:
					if saved:
						self._save_best_statfeat_regression(results[0], results[1], results[2], results[3], results[4], validation_save=validation_save)
				else:
					# losses.avg, mae_meter.avg, mean_error_meter.avg, mae_ratio_meter.avg, var_error_meter.avg, rho_meter.avg, outputs.numpy(), targets.numpy()
					loss, mae, me, mae_ratio, var, rho = results[:6]  # Take first 6 values for regression
					# result_object = RegressionResult(self.cur_epoch, loss, -1, mae, me, mae_ratio, var)
					if saved:
						self._save_best_regression(loss, mae, me, mae_ratio, var=var, rho=rho)
			elif self.task == 'classification' and (mode is None or 'cls' in mode):  # classification
				# losses.avg, acc, f1, recall, precision, predictions.numpy(), ground_truths.numpy()
				loss, acc, f1, recall, precision = results[:5]  # Take first 5 values for metrics
				if saved:
					self._save_best_classification(loss, acc, f1, recall, precision, validation_save=validation_save)
		if return_results is True: 
			return results[0], results
		else:
			return results[0] ### loss

	def _save_best_multi_tasks(self, results):
		"""Save best multi-task models"""
		loss, metrics, predictions, ground_truths = results
		# Create a list to store result objects for each task
		result_objects = []
		
		for ii, task in enumerate(self.multi_tasks):
			if 'classification' in task:
				result_objects.append(ClassificationResult(self.cur_epoch, loss, metrics[ii][0], metrics[ii][1], metrics[ii][2], metrics[ii][3]))
			elif 'regression' in task:
				result_objects.append(RegressionResult(self.cur_epoch, loss, -1, metrics[ii][0], metrics[ii][2], metrics[ii][3]))

		for ii, task in enumerate(self.multi_tasks):
			if 'classification' in task:
				save_models = ['best_accuracy_model.pth', 'best_f1_model.pth', 'last.pth']
				# accuracy, f1_score, recall, precision
				self._save_best_classification(loss, acc=metrics[ii][0], f1=metrics[ii][1], recall=metrics[ii][2], precision=metrics[ii][3], 
											   model_name=f"_classification_{ii}",
											   result_object=result_objects[ii], save_models=save_models)
			elif 'regression' in task:
				save_models = ['best_mae_model.pth', 'best_var_model.pth']
				# mae, mae_ratio, mean_error, var_error, rho
				self._save_best_regression(loss, mae=metrics[ii][0], me=metrics[ii][2], mae_ratio=metrics[ii][1], var=metrics[ii][3], rho=metrics[ii][4],
										   model_name=f"_regression_{ii}",
										   result_object=result_objects[ii], save_models=save_models)
				
	def _save_best_statfeat_regression(self, loss, metric1, metric2, moca_r2, mmse_r2, 
							  model_name="",
							  result_object_moca=None,
							  result_object_mmse=None,
							  save_models=['best_mae_model.pth', 'best_mae_ratio_model.pth', 'best_var_model.pth', 'best_r2_model.pth','last.pth'],
							  validation_save=False):
		"""Save best regression models"""
		### rho=-1 means rho is not used
		state = self._get_state_dict()
		rho=-1
		mae1, _, mae_ratio1, var1, _, _, _ = metric1
		mae2, _, mae_ratio2, var2, _, _, _ = metric2
		if validation_save is True:
			prefix = 'valid'
		else:
			prefix = 'test'

		if result_object_moca is None:
			result_object_moca = RegressionResult(self.cur_epoch, loss, rho, mae1, mae_ratio1, var1)
			result_object_moca_r2 = RSquareResult(self.cur_epoch, loss, mae1, moca_r2)
		
		if result_object_mmse is None:
			result_object_mmse = RegressionResult(self.cur_epoch, loss, rho, mae2, mae_ratio2, var2)
			result_object_mmse_r2 = RSquareResult(self.cur_epoch, loss, mae2, mmse_r2)

		# Update and save best MAE model
		if validation_save is False:
			if 'best_mae_model.pth' in save_models:
				if self.best_mae.mean_absolute_error is None or mae1 < self.best_mae.mean_absolute_error:
					self.best_mae = result_object_moca
					self._save_checkpoint(state, f"best_moca_mae_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_mae)

				if self.best_mae2.mean_absolute_error is None or mae2 < self.best_mae2.mean_absolute_error:
					self.best_mae2 = result_object_mmse
					self._save_checkpoint(state, f"best_mmse_mae_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_mae2)

			# Update and save best MAE ratio model
			if 'best_mae_ratio_model.pth' in save_models:
				if self.best_mae_ratio.mae_ratio is None or mae_ratio1 < self.best_mae_ratio.mae_ratio:
					self.best_mae_ratio = result_object_moca
					self._save_checkpoint(state, f"best_moca_mae_ratio_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_mae_ratio)
				if self.best_mae_ratio2.mae_ratio is None or mae_ratio2 < self.best_mae_ratio2.mae_ratio:
					self.best_mae_ratio2 = result_object_mmse
					self._save_checkpoint(state, f"best_mmse_mae_ratio_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_mae_ratio2)

			# Update and save best Var model    
			if 'best_var_model.pth' in save_models:
				if self.best_var.var is None or var1 < self.best_var.var:
					self.best_var = result_object_moca
					self._save_checkpoint(state, f"best_moca_var_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_var)
				if self.best_var2.var is None or var2 < self.best_var2.var:
					self.best_var2 = result_object_mmse
					self._save_checkpoint(state, f"best_mmse_var_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_var2)
	
			# Update and save best r2 model    
			if 'best_r2_model.pth' in save_models:
				if self.best_r2_moca.r2 is None or moca_r2 > self.best_r2_moca.r2:
					self.best_r2_moca = result_object_moca_r2
					self._save_checkpoint(state, f"best_moca_r2_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_r2_moca)
				if self.best_r2_mmse.r2 is None or mmse_r2 > self.best_r2_mmse.r2:
					self.best_r2_mmse = result_object_mmse_r2
					self._save_checkpoint(state, f"best_mmse_r2_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_r2_mmse)
		else:
			if 'best_mae_model.pth' in save_models:
				if self.best_mae_valid.mean_absolute_error is None or mae1 < self.best_mae_valid.mean_absolute_error:
					self.best_mae_valid = result_object_moca
					self._save_checkpoint(state, f"best_moca_mae_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_mae_valid)

				if self.best_mae2_valid.mean_absolute_error is None or mae2 < self.best_mae2_valid.mean_absolute_error:
					self.best_mae2_valid = result_object_mmse
					self._save_checkpoint(state, f"best_mmse_mae_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_mae2_valid)

			# Update and save best MAE ratio model
			if 'best_mae_ratio_model.pth' in save_models:
				if self.best_mae_ratio_valid.mae_ratio is None or mae_ratio1 < self.best_mae_ratio_valid.mae_ratio:
					self.best_mae_ratio_valid = result_object_moca
					self._save_checkpoint(state, f"best_moca_mae_ratio_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_mae_ratio_valid)
				if self.best_mae_ratio2_valid.mae_ratio is None or mae_ratio2 < self.best_mae_ratio2_valid.mae_ratio:
					self.best_mae_ratio2_valid = result_object_mmse
					self._save_checkpoint(state, f"best_mmse_mae_ratio_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_mae_ratio2_valid)

			# Update and save best Var model    
			if 'best_var_model.pth' in save_models:
				if self.best_var_valid.var is None or var1 < self.best_var_valid.var:
					self.best_var_valid = result_object_moca
					self._save_checkpoint(state, f"best_moca_var_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_var_valid)
				if self.best_var2_valid.var is None or var2 < self.best_var2_valid.var:
					self.best_var2_valid = result_object_mmse
					self._save_checkpoint(state, f"best_mmse_var_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_var2_valid)

			# Update and save best r2 model    
			if 'best_r2_model.pth' in save_models:
				if self.best_r2_moca_valid.r2 is None or moca_r2 > self.best_r2_moca_valid.r2:
					self.best_r2_moca_valid = result_object_moca_r2
					self._save_checkpoint(state, f"best_moca_r2_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_r2_moca_valid)
				if self.best_r2_mmse_valid.r2 is None or mmse_r2 > self.best_r2_mmse_valid.r2:
					self.best_r2_mmse_valid = result_object_mmse_r2
					self._save_checkpoint(state, f"best_mmse_r2_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_r2_mmse_valid)
		# Save last model
		if 'last.pth' in save_models:
			self._save_checkpoint(state, f"last{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth")



	def _save_best_regression(self, loss, mae, me, mae_ratio, var, rho=-1, 
							  model_name="",
							  result_object=None,
							  save_models=['best_mae_model.pth', 'best_mae_ratio_model.pth', 'best_var_model.pth', 'last.pth']
							  , validation_save=False):
		"""Save best regression models"""
		### rho=-1 means rho is not used
		state = self._get_state_dict()
		if validation_save is True:
			prefix = 'valid'
		else:
			prefix = 'test'

		if result_object is None:
			result_object = RegressionResult(self.cur_epoch, loss, rho, mae, mae_ratio, var)
		
		if validation_save is False:
			# Update and save best MAE model
			if 'best_mae_model.pth' in save_models:
				if self.best_mae.mean_absolute_error is None or mae < self.best_mae.mean_absolute_error:
					self.best_mae = result_object
					self._save_checkpoint(state, f"best_mae_model{model_name}_{prefix}.pth", self.best_mae)
			
			
			# Update and save best MAE ratio model
			if 'best_mae_ratio_model.pth' in save_models:
				if self.best_mae_ratio.mae_ratio is None or mae_ratio < self.best_mae_ratio.mae_ratio:
					self.best_mae_ratio = result_object
					self._save_checkpoint(state, f"best_mae_ratio_model{model_name}_{prefix}.pth", self.best_mae_ratio)
			
			# Update and save best Var model    
			if 'best_var_model.pth' in save_models:
				if self.best_var.var is None or var < self.best_var.var:
					self.best_var = result_object
					self._save_checkpoint(state, f"best_var_model{model_name}_{prefix}.pth", self.best_var)

		else:
			# Update and save best MAE model
			if 'best_mae_model.pth' in save_models:
				if self.best_mae_valid.mean_absolute_error is None or mae < self.best_mae_valid.mean_absolute_error:
					self.best_mae_valid = result_object
					self._save_checkpoint(state, f"best_mae_model{model_name}_{prefix}.pth", self.best_mae_valid)

			# Update and save best MAE ratio model
			if 'best_mae_ratio_model.pth' in save_models:
				if self.best_mae_ratio_valid.mae_ratio is None or mae_ratio < self.best_mae_ratio_valid.mae_ratio:
					self.best_mae_ratio_valid = result_object
					self._save_checkpoint(state, f"best_mae_ratio_model{model_name}_{prefix}.pth", self.best_mae_ratio_valid)
			
			# Update and save best Var model    
			if 'best_var_model.pth' in save_models:
				if self.best_var_valid.var is None or var < self.best_var_valid.var:
					self.best_var_valid = result_object
					self._save_checkpoint(state, f"best_var_model{model_name}_{prefix}.pth", self.best_var_valid)
		
		# Save last model
		if 'last.pth' in save_models:
			self._save_checkpoint(state, f"last{model_name}_{prefix}.pth")

	def _save_best_classification(self, loss, acc, f1, recall, precision, result_object=None,
								  model_name="",
								  save_models=['best_accuracy_model.pth', 'best_f1_model.pth', 'best_recall_model.pth', 'best_precision_model.pth', 'last.pth'],
								  validation_save=False):
		"""Save best classification models"""
		state = self._get_state_dict()
		if result_object is None:
			result_object = ClassificationResult(self.cur_epoch, loss, acc, f1, recall, precision)
		if validation_save is True:
			prefix = 'valid'
		else:
			prefix = 'test'
		# Update and save best accuracy model
		if 'best_accuracy_model.pth' in save_models:
			if self.best_acc.acc is None or acc > self.best_acc.acc:
				self.best_acc = result_object
				self._save_checkpoint(state, f"best_accuracy_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_acc)
		
		# Update and save best F1 model
		if 'best_f1_model.pth' in save_models:
			if self.best_f1.f1 is None or f1 > self.best_f1.f1:
				self.best_f1 = result_object
				self._save_checkpoint(state, f"best_f1_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_f1)
		
		# Update and save best recall model
		if 'best_recall_model.pth' in save_models:
			if self.best_recall.recall is None or recall > self.best_recall.recall:
				self.best_recall = result_object
				self._save_checkpoint(state, f"best_recall_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_recall)
		
		# Update and save best precision model
		if 'best_precision_model.pth' in save_models:
			if self.best_precision.precision is None or precision > self.best_precision.precision:
				self.best_precision = result_object
				self._save_checkpoint(state, f"best_precision_model{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth", self.best_precision)
		
		# Save last model
		if 'last.pth' in save_models:
			self._save_checkpoint(state, f"last{model_name}_{prefix}_testset{self.cfg['test_user_list_idx']}.pth")

	def _get_state_dict(self):
		"""Get current model state"""
		return {
			'epoch': self.cur_epoch,
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'scheduler': self.scheduler.state_dict(),
			'best_mae': self.best_mae,
			'best_var': self.best_var,
			'best_mae_ratio': self.best_mae_ratio,
			'best_acc': self.best_acc,
			'best_f1': self.best_f1,
			'best_recall': self.best_recall,
			'best_precision': self.best_precision
		}

	def _save_checkpoint(self, state, filename, best_result=None):
		"""Save a checkpoint"""
		if best_result:
			state[f'best_{filename.split("_")[1].split(".")[0]}'] = best_result
		torch.save(state, os.path.join(self.save_path, filename))
		logger.info(f'Successfully saved checkpoint to {os.path.join(self.save_path, filename)}', file=self.record)

	def _resume(self):
		"""Resume from checkpoint if available"""
		if not self.resume_file or not os.path.isfile(self.resume_file):
			return
		
		logger.info(f'Loading checkpoint {self.resume_file}', file=self.record)
		checkpoint = torch.load(self.resume_file)
		
		self.cur_epoch = checkpoint['epoch'] + 1
		self.model.load_state_dict(checkpoint['state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer'])
		self.scheduler.load_state_dict(checkpoint['scheduler'])
		
		if self.task == 'regression':
			self.best_mae = checkpoint.get('best_mae', self.best_mae)
			# self.best_rho = checkpoint.get('best_rho', self.best_rho)
			self.best_mae_ratio = checkpoint.get('best_mae_ratio', self.best_mae_ratio)
			self.best_var = checkpoint.get('best_var', self.best_var)
		else:
			self.best_acc = checkpoint.get('best_acc', self.best_acc)
			self.best_f1 = checkpoint.get('best_f1', self.best_f1)
			self.best_recall = checkpoint.get('best_recall', self.best_recall)
			self.best_precision = checkpoint.get('best_precision', self.best_precision)
		
		logger.info(f'Successfully loaded checkpoint from epoch {checkpoint["epoch"]}', file=self.record)

	def _to_device(self, data):
		"""Move data to device, handling nested structures"""
		if isinstance(data, dict):
			return {k: self._to_device(v) for k, v in data.items()}
		elif isinstance(data, (list, tuple)):
			return [self._to_device(x) for x in data]
		elif hasattr(data, 'to'):  # Check if it's a tensor-like object with 'to' method
			return data.to(self.device)
		else:
			return data  # Return as is if it's not a tensor or container
	

	def _print_progress(self, batch_idx, total_batches, loss_meters, iter_time, mode=None):
		"""Print training progress"""
		loss_str = ' | '.join([f'{loss_meter.name}: {loss_meter.avg:.3e}' for loss_meter in loss_meters])
		if mode is None or 'cls' in mode:
			logger.info(
				f'mode: cls | '
				f'Epoch: [{self.cur_epoch}/{self.all_epoch}] | '
				f'[{batch_idx + 1}/{total_batches}] | '
				f'lr: {self.scheduler.get_lr()[0]:.2e} | '
				f'{loss_str} | '
				f'Time: {iter_time.avg:.3f}',
				file=self.record
			)
		elif 'reg' in mode:
			logger.info(
				f'mode: reg | '
				f'Epoch: [{self.cur_epoch}/{self.all_epoch}] | '
				f'[{batch_idx + 1}/{total_batches}] | '
				f'lr: {self.regression_scheduler.get_lr()[0]:.2e} | '
				f'{loss_str} | '
				f'Time: {iter_time.avg:.3f}',
			file=self.record
		)
	

class Tester:
	def __init__(self, model, device, criterion, save_path=None, cfg=None, print_freq=10):
		self.model = model
		self.device = device
		self.criterion = criterion
		self.print_freq = print_freq
		self.cfg = cfg or {}
		self.save_path = save_path
		self.record = os.path.join(save_path, self.cfg.get('record', 'record.txt')) if save_path else None
		self.task = self.cfg.get('task', 'regression')

	def __call__(self, test_loader, verbose=True, mode=None):
		"""Run test evaluation"""
		self.model.eval()
		with torch.no_grad():
			self.multi_tasks = self.cfg['multi-task'] if 'multi-task' in self.cfg.keys() else None
			if self.multi_tasks is not None:
				results = self._evaluate_multi_tasks(test_loader, mode=mode)
			else:
				if self.task == 'regression':
					if self.cfg['train_stage'] == 'cog_regression' or self.cfg['train_stage'] == 'mix_up_v2' or self.cfg['train_stage'] == 'mix_up':
						results = self._evaluate_feats_regression(test_loader, mode=mode)
					else:
						results = self._evaluate_regression(test_loader)
				else:
					results = self._evaluate_classification(test_loader)
		 
			
			if verbose and self.record:
				self._log_results(results)
			
			return results

	def _evaluate_regression(self, test_loader, mode=None):
		"""Evaluate regression metrics"""
		losses = AverageMeter('Loss')
		mae_meter = AverageMeter('MAE')
		mae_ratio_meter = AverageMeter('MAE Ratio')
		rho_meter = AverageMeter('Rho')
		mean_error_meter = AverageMeter('Mean Error')
		var_error_meter = AverageMeter('Var Error')

		if self.cfg['train_stage'] == 'mix_up':
			tensor_VIDEO_TASK_MAPPING_MOCA_MATRIX = self._to_device(torch.from_numpy(VIDEO_TASK_MAPPING_MOCA_MATRIX)).to(torch.float32)
			tensor_VIDEO_TASK_MAPPING_MMSE_MATRIX = self._to_device(torch.from_numpy(VIDEO_TASK_MAPPING_MMSE_MATRIX)).to(torch.float32)
			tensor_PINV_VIDEO_TASK_MAPPING_MOCA_MATRIX = self._to_device(torch.from_numpy(PINV_VIDEO_TASK_MAPPING_MOCA_MATRIX)).to(torch.float32)
			tensor_PINV_VIDEO_TASK_MAPPING_MMSE_MATRIX = self._to_device(torch.from_numpy(PINV_VIDEO_TASK_MAPPING_MMSE_MATRIX)).to(torch.float32)

		for inputs, targets in test_loader:
			input_data = self._convert_data_dict_to_list(inputs)
			gt_data = self._convert_data_dict_to_list(targets)

			# Move data to device
			input_data = self._to_device(input_data)
			gt_data = self._to_device(gt_data)

			input_data = transpose_nested_list(input_data)
			gt_data = transpose_nested_list(gt_data)
			
			if self.cfg['train_stage'] == 'mask_pretrain':
				ExG_input = input_data[0][0]
				B, input_chs, seq_len = ExG_input.shape
				# ExG_input = ExG_input.view(B, input_chs, 3, seq_len//3)
				ExG_input = ExG_input.view(B, input_chs, 3, seq_len//3)
				ExG_input = resample(ExG_input, 200, axis=-1)
				mask = generate_mask(ExG_input.shape[0], ExG_input.shape[1], ExG_input.shape[2], self.cfg['mask_ratio'], self.device)
				outputs = self.model(ExG_input, mask=mask)
				masked_outputs = outputs[0][mask==1]
				masked_inputs = ExG_input[mask==1]
				loss = self.criterion(masked_outputs, masked_inputs)
				mae, mae_ratio, mean_error, var_error, rho = evaluate(masked_outputs, masked_inputs)
			
		return losses.avg, mae_meter.avg, mean_error_meter.avg, mae_ratio_meter.avg, var_error_meter.avg, rho_meter.avg, 0 ,0


	def _evaluate_feats_regression(self, test_loader, mode=None):
		"""Evaluate regression metrics"""
		losses = AverageMeter('Loss')
		mae_meter = AverageMeter('MAE')
		mae_ratio_meter = AverageMeter('MAE Ratio')
		rho_meter = AverageMeter('Rho')
		mean_error_meter = AverageMeter('Mean Error')
		var_error_meter = AverageMeter('Var Error')

		mae_meter2 = AverageMeter('MAE2')
		mae_ratio_meter2 = AverageMeter('MAE Ratio2')
		rho_meter2 = AverageMeter('Rho2')
		mean_error_meter2 = AverageMeter('Mean Error2')
		var_error_meter2 = AverageMeter('Var Error2')
		
		r2_output_list = []
		r2_label_list = []
		if self.cfg['train_stage'] == 'mix_up':
			tensor_VIDEO_TASK_MAPPING_MOCA_MATRIX = self._to_device(torch.from_numpy(VIDEO_TASK_MAPPING_MOCA_MATRIX)).to(torch.float32)
			tensor_VIDEO_TASK_MAPPING_MMSE_MATRIX = self._to_device(torch.from_numpy(VIDEO_TASK_MAPPING_MMSE_MATRIX)).to(torch.float32)
			tensor_PINV_VIDEO_TASK_MAPPING_MOCA_MATRIX = self._to_device(torch.from_numpy(PINV_VIDEO_TASK_MAPPING_MOCA_MATRIX)).to(torch.float32)
			tensor_PINV_VIDEO_TASK_MAPPING_MMSE_MATRIX = self._to_device(torch.from_numpy(PINV_VIDEO_TASK_MAPPING_MMSE_MATRIX)).to(torch.float32)


		for inputs, targets in test_loader:
			input_data = self._convert_data_dict_to_list(inputs)
			gt_data = self._convert_data_dict_to_list(targets)

			# Move data to device
			input_data = self._to_device(input_data)
			gt_data = self._to_device(gt_data)

			input_data = transpose_nested_list(input_data)
			gt_data = transpose_nested_list(gt_data)

			if isinstance(gt_data[0][0], list):
				max_value = np.array(gt_data[0][0]).max()
			elif isinstance(gt_data[0][0], torch.Tensor):
				max_value = gt_data[0][0].max()
			max_norm_moca = 30 if max_value <= 1 else 1
			max_norm_mmse = 30 if max_value <= 1 else 1

			if mode is None:
				### outputs is a tuple
				### the first item is the classification output
				### the second item is the regression output if the output length > 1
				if len(input_data[0]) == 1:
					ExG_statfeat = input_data[0][0]
				else:
					ExG_statfeat = input_data[0]
				target = gt_data[0]
				outputs = self.model(ExG_statfeat)
				loss = self.criterion(outputs, target)
			
			elif mode == 'test_reg':
				if len(input_data) == 1:
					input_d = input_data[0]
					gt_data = get_data_dict_list_idx(targets, 0) ### test only one view
					gt_data = get_data_dict_list_to_item(gt_data)
					outputs, target = self.model(input_d, gt_data, mode='test_reg')
					loss = self.criterion(outputs, target)
			# Calculate metrics
			mae, mae_ratio, mean_error, var_error, rho = evaluate(outputs[0]*max_norm_moca, target[0]*max_norm_moca)
			mae2, mae_ratio2, mean_error2, var_error2, rho2 = evaluate(outputs[1]*max_norm_mmse, target[1]*max_norm_mmse)
			
			# Update meters
			batch_size = self._get_batch_size(input_data[0])
			losses.update(loss[0].item(), batch_size)
			mae_meter.update(mae, batch_size)
			mae_ratio_meter.update(mae_ratio, batch_size)
			mean_error_meter.update(mean_error, batch_size)
			var_error_meter.update(var_error, batch_size)
			rho_meter.update(rho, batch_size)

			mae_meter2.update(mae2, batch_size)
			mae_ratio_meter2.update(mae_ratio2, batch_size)
			mean_error_meter2.update(mean_error2, batch_size)
			var_error_meter2.update(var_error2, batch_size)
			rho_meter2.update(rho2, batch_size)
			r2_output_list.append(outputs)
			r2_label_list.append(target)

		r2_moca_pred = torch.tensor([i for j in r2_output_list for i in j[0]])
		r2_mmse_pred = torch.tensor([i for j in r2_output_list for i in j[1]])    
		r2_moca_label = torch.tensor([i for j in r2_label_list for i in j[0]])  
		r2_mmse_label = torch.tensor([i for j in r2_label_list for i in j[1]])

		mae, mae_ratio, mean_error, var_error, rho = evaluate(r2_moca_pred.cpu()*max_norm_moca, r2_moca_label.cpu()*max_norm_moca)
		mae2, mae_ratio2, mean_error2, var_error2, rho2 = evaluate(r2_mmse_pred.cpu()*max_norm_mmse, r2_mmse_label.cpu()*max_norm_mmse)
		
		if r2_moca_pred.cpu().max()*max_norm_moca == r2_moca_pred.cpu().min()*max_norm_moca:
			r_value_moca = 0
		else:
			_, _, r_value_moca, _, _ = linregress(r2_moca_pred.cpu()*max_norm_moca, r2_moca_label.cpu()*max_norm_moca)
		if r2_mmse_pred.cpu().max()*max_norm_mmse == r2_mmse_pred.cpu().min()*max_norm_mmse:
			r_value_mmse = 0
		else:
			_, _, r_value_mmse, _, _ = linregress(r2_mmse_pred.cpu()*max_norm_mmse, r2_mmse_label.cpu()*max_norm_mmse)

		print(f"moca preds:{np.round((r2_moca_pred.cpu()*max_norm_moca).detach().numpy(), 4)}")
		print(f"moca label:{np.round((r2_moca_label.cpu()*max_norm_moca).detach().numpy(), 4)}")

		print(f"mmse preds:{np.round((r2_mmse_pred.cpu()*max_norm_mmse).detach().numpy(), 4)}")
		print(f"mmse label:{np.round((r2_mmse_label.cpu()*max_norm_mmse).detach().numpy(), 4)}")

		return losses.avg, [mae, mean_error, mae_ratio, var_error, rho, 0 ,0], \
			  [mae2, mean_error2, mae_ratio2, var_error2, rho2, 0 ,0],\
				  r_value_moca, r_value_mmse

	def _convert_data_dict_to_list(self, data_dict):
		"""Convert data dict to list"""
		keys = list(data_dict.keys())
		data_list = []
		for i in range(len(keys)):
			data_list.append(data_dict[keys[i]])
		return data_list

	def _evaluate_classification(self, test_loader):
		"""Evaluate classification metrics"""
		loss_meters = [AverageMeter(f'Loss{i}') for i in range(self.criterion.get_num_loss_items())]
		predictions = []
		ground_truths = []
		iter_time = AverageMeter('Time')
		
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			time_start = time.time()
			# Convert data dict to list
			input_data = self._convert_data_dict_to_list(inputs)
			gt_data = self._convert_data_dict_to_list(targets)
			
			# Move data to device
			input_data = self._to_device(input_data)
			gt_data = self._to_device(gt_data)

			input_data = transpose_nested_list(input_data)
			gt_data = transpose_nested_list(gt_data)

			
			if len(input_data) == 1:
				### no aug, so here length is one.
				input_data = input_data[0]
				gt_data = gt_data[0]
				
				# Forward pass
				### outputs is a tuple
				### the first item is the classification output
				### the second item is the regression output if the output length > 1
				outputs = self.model(input_data)
				loss = self.criterion(outputs, gt_data)
			
			for i, loss_meter in enumerate(loss_meters):
				loss_meter.update(loss[i].item())
			predictions.append(outputs[0].cpu())
			
			ground_truths.append(gt_data[0].cpu())
			iter_time.update(time.time() - time_start)
			
			# Print progress
			if (batch_idx + 1) % self.print_freq == 0:
				self._print_progress(batch_idx, len(test_loader), loss_meters, iter_time)
		# Concatenate all predictions and ground truths
		predictions = torch.cat(predictions)
		ground_truths = torch.cat(ground_truths)
		
		# Calculate metrics
		acc, f1, recall, precision = evaluate_c(predictions, ground_truths, num_classes=self.cfg['classes'])
		
		return loss_meters[0].avg, acc, f1, recall, precision, predictions.numpy(), ground_truths.numpy()

	def _evaluate_multi_tasks(self, test_loader, mode=None):
		"""Evaluate multi-task metrics"""
		
		loss_meters = [AverageMeter(f'Loss{i}') for i in range(self.criterion.get_num_loss_items())]
		predictions = []
		ground_truths = []
		iter_time = AverageMeter('Time')
		
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			time_start = time.time()
			# Convert data dict to list
			input_data = self._convert_data_dict_to_list(inputs)
			gt_data = self._convert_data_dict_to_list(targets)
			

			# Move data to device
			input_data = self._to_device(input_data)
			gt_data = self._to_device(gt_data)

			input_data = transpose_nested_list(input_data)
			gt_data = transpose_nested_list(gt_data)

			
			if len(input_data) == 1:
				input_data = input_data[0]
				gt_data = gt_data[0]
			
			# Forward pass
			### outputs is a tuple
			### the first item is the classification output
			### the second item is the regression output if the output length > 1

			if (mode is None or 'cls' in mode) and 'Joint_CLS_REG_Model_moco' not in self.model._get_name():
				outputs = self.model(input_data)
				loss = self.criterion(outputs, gt_data)
			elif 'Joint_CLS_REG_Model_moco' in self.model._get_name() or 'Joint_CLS_REG_Model_moco_v2' in self.model._get_name():
				if 'cls' in mode:
					outputs = self.model(input_data, mode='test_cls')
					loss = self.criterion(outputs, gt_data)
				elif 'reg' in mode:
					gt_data = get_data_dict_list_idx(targets, 0) ### test only one view
					outputs, gt_data = self.model(input_data, gt_data, mode='test_reg')
					loss = self.criterion(outputs, gt_data)
			elif 'reg' in mode:
				gt_data = get_data_dict_list_idx(targets, 0) ### test only one view
				outputs, gt_data = self.model(input_data, gt_data, mode='test_reg')
				loss = self.criterion(outputs, gt_data)
			
			
			for i, loss_meter in enumerate(loss_meters):
				loss_meter.update(loss[i].item())

			predictions.append(self.detach_and_move_to_cpu(outputs))
			ground_truths.append(self.detach_and_move_to_cpu(gt_data))
			iter_time.update(time.time() - time_start)
			
			# Print progress
			if (batch_idx + 1) % self.print_freq == 0:
				self._print_progress(batch_idx, len(test_loader), loss_meters, iter_time)
		# Concatenate all predictions and ground truths
		metrics = []
		for ii, task in enumerate(self.multi_tasks):
			preds_list = torch.cat([preds[ii] for preds in predictions])
			ground_truths_list = torch.cat([ground_truth[ii] for ground_truth in ground_truths])
			if 'classification' in task:
				num_classes = int(task.split('-')[-1])
				# accuracy, f1_score, recall, precision
				metrics.append(evaluate_c(preds_list, ground_truths_list, num_classes=num_classes))
			elif 'regression' in task:
				# mae, mae_ratio, mean_error, var_error, rho
				metrics.append(evaluate(preds_list, ground_truths_list))
	
		return loss_meters[0].avg, metrics, self.detach_and_move_to_cpu(predictions), self.detach_and_move_to_cpu(ground_truths)
	
	def _to_device(self, data):
		"""Move data to device, handling nested structures"""
		if isinstance(data, dict):
			return {k: self._to_device(v) for k, v in data.items()}
		elif isinstance(data, (list, tuple)):
			return [self._to_device(x) for x in data]
		elif hasattr(data, 'to'):  # Check if it's a tensor-like object with 'to' method
			return data.to(self.device)

		else:
			return data  # Return as is if it's not a tensor or container


	# Process outputs and ground truths for saving
	def detach_and_move_to_cpu(self, data):
		if isinstance(data, tuple) or isinstance(data, list):
			return [item.detach().cpu() if hasattr(item, 'detach') else item for item in data]
		else:
			return data.detach().cpu() if hasattr(data, 'detach') else data
				

	def _get_batch_size(self, inputs):
		"""Get batch size from inputs"""
		if isinstance(inputs, dict):
			return next(iter(inputs.values())).size(0)
		elif isinstance(inputs, (list, tuple)):
			if isinstance(inputs[0], list):
				return len(inputs[0])
			return inputs[0].size(0)
		return inputs.size(0)
	
	def _print_progress(self, batch_idx, total_batches, loss_meters, iter_time):
		"""Print training progress"""
		loss_str = ' | '.join([f'{loss_meter.name}: {loss_meter.avg:.3e}' for loss_meter in loss_meters])
		logger.info(
			f'Testing:  '
			f'[{batch_idx + 1}/{total_batches}] | '
			f'{loss_str} | '
			f'Time: {iter_time.avg:.3f}',
			file=self.record
		)

	def _log_classification_results(self, loss, acc, f1, recall, precision):
		"""Log classification results"""
		logger.info(
			f'\n=>Test result:\n'
			f'loss:{loss:.3e}, acc:{acc:.3e}, '
			f'f1:{f1:.3e}, recall:{recall:.3e}, '
			f'precision:{precision:.3e}',
			file=self.record
		) 
	def _log_regression_results(self, loss, mae, mean_error, mae_ratio, var_error, rho):
		"""Log regression results"""
		logger.info(
			f'\n=>Test result:\n'
			f'loss:{loss:.3e}, mae:{mae:.3e}, '
			f'mae_ratio:{mae_ratio:.3e}, '
			f'mean_error:{mean_error:.3e}, '
			f'var_error:{var_error:.3e}, '
			f'rho:{rho:.3e}',
			file=self.record
		)
	
	def _log_results(self, results):
		"""Log evaluation results"""
		if self.multi_tasks is not None:
			loss = results[0]
			for ii, task in enumerate(self.multi_tasks):
				if 'classification' in task:
					acc, f1, recall, precision = results[1][ii]
					self._log_classification_results(loss, acc, f1, recall, precision)
				elif 'regression' in task:
					mae, mae_ratio, mean_error, var_error, rho = results[1][ii]
					self._log_regression_results(loss, mae, mean_error, mae_ratio, var_error, rho)
		else:
			if self.task == 'regression':
				if self.cfg['train_stage'] == 'mix_up' or self.cfg['train_stage'] == 'mix_up_v2' or self.cfg['train_stage'] == 'cog_regression':
					loss, moca_metric, mmse_metric, moca_r2, mmse_r2 = results
					logger.info(f'*************MoCA Test:\n'
								f'R_value: {moca_r2}',file=self.record)
					self._log_regression_results(loss, moca_metric[0], moca_metric[1], moca_metric[2], moca_metric[3], moca_metric[4])
					logger.info(f'*************MMSE Test:\n'
								f'R_value: {mmse_r2}',file=self.record)
					self._log_regression_results(loss, mmse_metric[0], mmse_metric[1], mmse_metric[2], mmse_metric[3], mmse_metric[4])
				else:
					loss, mae, mean_error, mae_ratio, var_error, rho = results[:6]
					self._log_regression_results(loss, mae, mean_error, mae_ratio, var_error, rho)
			else:
				loss, acc, f1, recall, precision = results[:5]
				self._log_classification_results(loss, acc, f1, recall, precision)