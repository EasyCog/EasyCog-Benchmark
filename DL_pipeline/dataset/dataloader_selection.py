from DL_pipeline.dataset.dataloader import *
from DL_pipeline.dataset.dataloader_joint_training import *

def select_dataloader(train_records, valid_records, test_records, cfg):

    if cfg['dataloader'] == 'EasyCog_Dataloader':
        train_loader, valid_loader, test_loader = EasyCog_Dataloader(sliced_data_json_file=cfg['sliced_trials_json'],
                                                                     train_trials=train_records,
                                                                     valid_trials=valid_records,
                                                                     test_trials=test_records,
                                                                     batch_size=cfg['batch_size'],
                                                                     data_aug_methods=cfg['data_aug_methods'],
                                                                     input_type_list=cfg['input_type_list'],
                                                                     input_format_dict=cfg['input_format_dict'],
                                                                     gt_type_list=cfg['gt_type_list'],
                                                                     gt_format_dict=cfg['gt_format_dict'],
                                                                     norm_params=cfg['norm_params'],
                                                                     logger_file=cfg['logger_file'],
                                                                     num_workers=cfg['num_workers'],
                                                                     persistent_workers=cfg['persistent_workers'],
                                                                     prefetch_factor=cfg['prefetch_factor'],
                                                                     train_user_list=cfg['train_user_list'],
                                                                     valid_user_list=cfg['valid_user_list'],
                                                                     test_user_list=cfg['test_user_list'],
                                                                     )()
    

    
    elif cfg['dataloader'] == 'EasyCog_Joint_Training_Dataloader':
        train_loader, valid_loader, test_loader = EasyCog_Joint_Training_Dataloader(sliced_data_json_file=cfg['sliced_trials_json'],
                                                                     train_user_indices=train_records,
                                                                     valid_user_indices=valid_records,
                                                                     test_user_indices=test_records,
                                                                     batch_size=cfg['batch_size'],
                                                                     data_aug_methods=cfg['data_aug_methods'],
                                                                     input_type_list=cfg['input_type_list'],
                                                                     input_format_dict=cfg['input_format_dict'],
                                                                     gt_type_list=cfg['gt_type_list'],
                                                                     gt_format_dict=cfg['gt_format_dict'],
                                                                     norm_params=cfg['norm_params'],
                                                                     logger_file=cfg['logger_file'],
                                                                     num_workers=cfg['num_workers'],
                                                                     persistent_workers=cfg['persistent_workers'],
                                                                     prefetch_factor=cfg['prefetch_factor'],
                                                                     train_user_list=cfg['train_user_list'],
                                                                     valid_user_list=cfg['valid_user_list'],
                                                                     test_user_list=cfg['test_user_list'],
                                                                     )()
    
    else:
        raise NotImplementedError('Not implemented')

    return train_loader, valid_loader, test_loader


