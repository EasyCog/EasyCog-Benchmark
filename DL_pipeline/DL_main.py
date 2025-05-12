import torch
import torch.nn as nn
import os
from utils import *
from DL_pipeline.learn_utils import Trainer, init_device, init_model, Tester
from DL_pipeline.learn_utils import FakeLR, WarmUpCosineAnnealingLR
from DL_pipeline.dataset import select_dataloader
from DL_pipeline.losses import loss_selection


def DL_run(cfg, cfg_file):

	logger.info('=> PyTorch Version: {}'.format(torch.__version__))
	
	device, pin_memory = init_device(
		cfg["seed"], cfg["cpu"], cfg["gpu"], cfg["cpu_affinity"])

	test_user_list_idx = cfg['test_user_list_idx']
	valid_user_list_idx = cfg['valid_user_list_idx']
	data_set = select_data_set(cfg['data_type'])
	total_users_lists = data_set.total_users_lists
	train_user_list_idxs = [i for i in range(len(total_users_lists)) if i != test_user_list_idx and i != valid_user_list_idx]
	train_user_list = []
	for user_list in train_user_list_idxs:
		train_user_list.extend(total_users_lists[user_list])
	test_user_list = total_users_lists[test_user_list_idx]	
	if valid_user_list_idx is not None:
		valid_user_list = total_users_lists[valid_user_list_idx]
	else:
		valid_user_list = None

	cfg['train_user_list'] = train_user_list
	cfg['valid_user_list'] = valid_user_list
	cfg['test_user_list'] = test_user_list
	train_records, val_records, test_records, adaptation_data_records = get_records(cfg)

	if 'debug' in cfg.keys() and cfg['debug'] is True:
		train_records = train_records[:100]
	
	if cfg['train'] != None:
		train_loader, valid_loader, test_loader = select_dataloader(train_records, val_records, test_records, cfg)
	
	if cfg['DA']['if_use']:
		target_loader, _, _ = select_dataloader(adaptation_data_records, None, None, cfg)
	else:
		target_loader = None
	

	model = init_model(cfg)
	model.to(device)

	criterion = loss_selection.get_loss(cfg, key='Loss')

	if 'Test_Loss' in cfg.keys():
		test_criterion = loss_selection.get_loss(cfg, key='Test_Loss')
	else:
		test_criterion = None

	if cfg['evaluate']:
		_, _, test_loader = select_dataloader(cfg=cfg)
		t = Tester(model, device, test_criterion, cfg=cfg)(test_loader)
		return
	
	lr_init = cfg["lr_init"]               
	weight_decay = cfg["weight_decay"] if "weight_decay" in cfg.keys() else 1e-4
	if cfg["optimizer"] == "Adam":
		optimizer = torch.optim.AdamW(
			[{'params': model.parameters(), 'lr': lr_init, 'weight_decay': weight_decay}])
	else:
		optimizer = torch.optim.sgd([{'params': model.parameters()}], lr_init)

	warmup_epochs = cfg['warmup_epochs'] if 'warmup_epochs' in cfg.keys() else 30
	if cfg["scheduler"] == "const":
		scheduler = FakeLR(optimizer=optimizer)
	else:
		scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer, T_max=cfg['epochs'] * len(train_loader),
											T_warmup=warmup_epochs * len(train_loader), eta_min=lr_init/20)
		
	save_path = generate_save_file_folder(f"{cfg['save_path']}_{cfg_file}")

	
	trainer = Trainer(model=model, device=device,
					  optimizer=optimizer, criterion=criterion,
					  scheduler=scheduler, cfg=cfg,
					  resume=cfg['resume'], save_path=save_path,
					  print_freq=cfg['print_freq'], val_freq=cfg['valid_freq'], test_freq=cfg['test_freq'],
					  test_criterion=test_criterion,
					  )
	
	
	trainer.loop(cfg['epochs'], train_loader, test_loader, val_loader=valid_loader, target_loader=target_loader)