from DL_pipeline.learn_utils import *
from DL_pipeline.dataset import *
from utils import *
from DL_pipeline.learn_utils.get_features_utils import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
# from DL_pipeline.Cog_Regression import *
from DL_pipeline.learn_utils import Trainer, init_device, init_model, Tester
from DL_pipeline.learn_utils import FakeLR, WarmUpCosineAnnealingLR
from DL_pipeline.dataset import select_dataloader
from DL_pipeline.losses import loss_selection
from utils import read_task_score, update_subject_task_indices


def Cog_Joint_CLS_REG_Validation_Saved(cfg, cfg_file):
	logger.info('=> PyTorch Version: {}'.format(torch.__version__))
	reg_cfg_file = cfg['reg_cfg_file']
	reg_cfg = read_config(reg_cfg_file)
	cls_cfg = cfg
	reg_cfg['data_type'] = cfg['data_type']

	reg_cfg['test_user_list_idx'] = cfg['test_user_list_idx']
	reg_cfg['valid_user_list_idx'] = cfg['valid_user_list_idx']

	reg_cfg['sliced_data_folder'], reg_cfg['sliced_trials_json'] = select_trials_json(reg_cfg, cfg['data_type'])
	print(f"{line_seg}")
	print(f"reg_cfg: Using {cfg['data_type']} data | sliced_data_folder: {reg_cfg['sliced_data_folder']} | sliced_trials_json: {reg_cfg['sliced_trials_json']}")
	print(f"{line_seg}")

	test_user_list_idx = cfg['test_user_list_idx']
	valid_user_list_idx = cfg['valid_user_list_idx']
	data_set = select_data_set(cfg['data_type'])
	total_users_lists = data_set.total_users_lists
	train_user_list_idxs = [i for i in range(len(total_users_lists)) if i != test_user_list_idx and i != valid_user_list_idx]
	train_user_list = []
	for user_list in train_user_list_idxs:
		train_user_list.extend(total_users_lists[user_list])
	test_user_list = total_users_lists[test_user_list_idx]	
	valid_user_list = total_users_lists[valid_user_list_idx]

	cls_cfg['train_user_list'] = train_user_list
	cls_cfg['test_user_list'] = test_user_list
	cls_cfg['valid_user_list'] = valid_user_list
	reg_cfg['train_user_list'] = train_user_list
	reg_cfg['test_user_list'] = test_user_list
	reg_cfg['valid_user_list'] = valid_user_list
	
	### get cls dataloaders
	train_records, val_records, test_records = get_records(cls_cfg)

	if 'debug' in cls_cfg.keys() and cls_cfg['debug'] is True:
		train_records = train_records[:100]
	
	cls_cfg['pic_finetune'] = cls_cfg['pic_finetune'] if 'pic_finetune' in cls_cfg.keys() else False

	if cls_cfg['train'] != None:
		if cls_cfg['pic_finetune'] is True:
			train_records = np.concatenate((train_records, test_records))
		cls_train_loader, cls_valid_loader, cls_test_loader = select_dataloader(train_records, val_records, test_records, cls_cfg)
	

	# prepare subject_taskscore_dict
	### TODO
	subject_taskscore_dict = read_task_score('/data/mmWave_group/EasyCog/Ours_v2/Patient Info.xlsx')
	subject_task_indices = get_cfg_indices_with_each_user(cls_cfg)
	subject_task_indices = update_subject_task_indices(subject_task_indices, subject_taskscore_dict)
	train_indices, train_labels = get_raw_subject_task_indices(subject_task_indices, train_user_list, 'none', 'none', is_features=False)
	test_indices, test_labels = get_raw_subject_task_indices(subject_task_indices, test_user_list, 'none', 'none', is_features=False)
	valid_indices, valid_labels = get_raw_subject_task_indices(subject_task_indices, valid_user_list, 'none', 'none', is_features=False)
	reg_train_loader, reg_valid_loader, reg_test_loader = select_dataloader([train_indices, train_labels], [valid_indices, valid_labels], [test_indices, test_labels], reg_cfg)


	### init model
	device, pin_memory = init_device(
	cls_cfg["seed"], cls_cfg["cpu"], cls_cfg["gpu"], cls_cfg["cpu_affinity"])
	

	for k in reg_cfg['model'].keys():
		assert cls_cfg['model'][k] == reg_cfg['model'][k], f'cls and reg model must be unified one: {k}'
	model = init_model(reg_cfg)
	model.to(device)

	### get cls criterion
	cls_criterion = loss_selection.get_loss(cls_cfg, key='Loss')
	if 'Test_Loss' in cls_cfg.keys():
		test_cls_criterion = loss_selection.get_loss(cls_cfg, key='Test_Loss')
	else:
		test_cls_criterion = None


	reg_criterion = loss_selection.get_loss(reg_cfg, 'Loss')
	if 'Test_Loss' in reg_cfg.keys():
		test_reg_criterion = loss_selection.get_loss(reg_cfg, key='Test_Loss')
	else:
		test_reg_criterion = None

	cls_lr_init = cls_cfg['lr_init']
	cls_weight_decay = cls_cfg['weight_decay'] if 'weight_decay' in cls_cfg.keys() else 1e-4
	if cls_cfg["optimizer"] == "Adam":
		cls_optimizer = torch.optim.AdamW(
			[{'params': model.cls_model.parameters(), 'lr': cls_lr_init, 'weight_decay': cls_weight_decay}])
		# optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr_init, weight_decay=1e-5)
	else:
		cls_optimizer = torch.optim.sgd([{'params': model.cls_model.parameters()}], cls_lr_init)
	
	reg_lr_init = reg_cfg['lr_init']
	reg_weight_decay = reg_cfg['weight_decay'] if 'weight_decay' in cls_cfg.keys() else 1e-4
	if reg_cfg["optimizer"] == "Adam":
		reg_optimizer = torch.optim.AdamW(
			[{'params': model.parameters(), 'lr': reg_lr_init, 'weight_decay': reg_weight_decay}])
		# optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr_init, weight_decay=1e-5)
	else:
		reg_optimizer = torch.optim.sgd([{'params': model.parameters()}], cls_lr_init)


	cls_warmup_epochs = cls_cfg['warmup_epochs'] if 'warmup_epochs' in cls_cfg.keys() else 30
	if cls_cfg["scheduler"] == "const":
		cls_scheduler = FakeLR(optimizer=cls_optimizer)
	else:
		cls_scheduler = WarmUpCosineAnnealingLR(optimizer=cls_optimizer, T_max=cls_cfg['epochs'] * len(cls_train_loader),
											T_warmup=cls_warmup_epochs * len(cls_train_loader), eta_min=cls_lr_init/20)
	
	reg_warmup_epochs = reg_cfg['warmup_epochs'] if 'warmup_epochs' in reg_cfg.keys() else 30
	if reg_cfg["scheduler"] == "const":
		reg_scheduler = FakeLR(optimizer=reg_optimizer)
	else:
		reg_scheduler = WarmUpCosineAnnealingLR(optimizer=reg_optimizer, T_max=reg_cfg['epochs'] * len(reg_train_loader),
											T_warmup=reg_warmup_epochs * len(reg_train_loader), eta_min=reg_lr_init/20)

	reg_save_path = generate_save_file_folder(f"{reg_cfg['save_path']}_{reg_cfg_file}")
	save_path = generate_save_file_folder(f"{cls_cfg['save_path']}_{cfg_file}")	

	### todo: modify the trainer to joint cls and reg
	trainer = Trainer(model=model, device=device,
						optimizer=cls_optimizer, criterion=cls_criterion,
						scheduler=cls_scheduler, cfg=cls_cfg,
						regression_optimizer=reg_optimizer,
						regression_criterion=reg_criterion,
						regression_scheduler=reg_scheduler,
						regression_test_criterion=test_reg_criterion,
						regression_save_path=reg_save_path,
						regression_print_freq=1,
						reg_cfg=reg_cfg,
						resume=cls_cfg['resume'], save_path=save_path,
						print_freq=cls_cfg['print_freq'], val_freq=1, test_freq=cls_cfg['test_freq'],
						test_criterion=test_cls_criterion,
						)
		
	assert cfg['epochs'] == reg_cfg['epochs'], 'cls and reg epochs must be unified'

	trainer.loop_joint_classification_regression_validation_saved(cfg['epochs'], 
											  cls_loaders=[cls_train_loader, cls_valid_loader, cls_test_loader],
											  reg_loaders=[reg_train_loader, reg_valid_loader, reg_test_loader]
											  )