from DL_pipeline.DL_main import DL_run
from utils.utils import read_config, str_to_bool, select_data_set, select_trials_json, select_data_set
from utils.logger import line_seg
from DL_pipeline.Cog_Regression import Cog_Regression
from DL_pipeline.Cog_Regression_Together import Cog_Regression_Together
from DL_pipeline.Cog_joint_train import Cog_Joint_CLS_REG_Validation_Saved
import argparse
import os
import time
from utils import logger, test_gpu


def parse_args():
    parser = argparse.ArgumentParser(description='EasyCog')
    parser.add_argument('--data_type', type=str, default='all_0426', help='choose data: clean_0426 / all_0426 ...')
    parser.add_argument('--test_user_list_idx', type=int, help='test user list index')
    parser.add_argument('--valid_user_list_idx', type=int, help='valid user list index')
    parser.add_argument('--log_file', type=str, help='log file')
    parser.add_argument('--task_pretrain', type=str_to_bool, help='Whether to conduct task pretraining (true/false)')
    parser.add_argument('--cog_regression', type=str_to_bool, help='Whether to conduct cog regression (true/false)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--cfg_file', type=str, help='config file path')
    parser.add_argument('--cog_regression_cfg_file', default=None, type=str, help='cog regression config file path')
    parser.add_argument('--joint_cls_reg_train', type=str_to_bool, default=False, help='Whether to conduct joint cls reg train (true/false)')
    parser.add_argument('--wait_time', type=int, default=0, help='wait time in seconds')
    parser.add_argument('--gpu_memory', type=int, default=23, help='gpu memory requirement in GB')
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    all_users = select_data_set(args.data_type).total_users_lists

    wait_time = args.wait_time
    test_user_list_idx = args.test_user_list_idx
    valid_user_list_idx = args.valid_user_list_idx
    log_file = args.log_file
    task_pretrain = args.task_pretrain
    cog_regression = args.cog_regression
    gpu = args.gpu
    cfg_file = os.path.join('DL_pipeline/configs', args.cfg_file)

    
    while True:
        if test_gpu(gpu, memory=args.gpu_memory):
            logger.info(f"GPU {gpu} is available. Starting training...")
            break
        else:
            logger.info(f"No available GPU found. Waiting for 10 minutes...")
            time.sleep(10*60)
    
    
    logger.info(f"Waiting for {wait_time} seconds")
    time.sleep(wait_time)
    cfg = read_config(cfg_file)

    cfg['sliced_data_folder'], cfg['sliced_trials_json'] = select_trials_json(cfg, args.data_type)
    print(f"{line_seg}")
    print(f"cfg: Using {args.data_type} data | sliced_data_folder: {cfg['sliced_data_folder']} | sliced_trials_json: {cfg['sliced_trials_json']}")
    print(f"{line_seg}")


    if args.cog_regression_cfg_file is not None:
        cfg['reg_cfg_file'] = os.path.join('DL_pipeline/configs', args.cog_regression_cfg_file)
    else:
        cfg['reg_cfg_file'] = None

        

    cfg['test_user_list_idx'] = test_user_list_idx
    cfg['test_subject'] = all_users[test_user_list_idx]
    cfg['valid_user_list_idx'] = valid_user_list_idx
    cfg['valid_subject'] = all_users[valid_user_list_idx]
    cfg['gpu'] = gpu
    cfg['data_type'] = args.data_type

    cfg['joint_cls_reg_train'] = args.joint_cls_reg_train
    
    if args.cog_regression_cfg_file is None:
        DL_run(cfg, cfg_file)
    
    elif args.cog_regression_cfg_file is not None and cog_regression is True:

        Cog_Joint_CLS_REG_Validation_Saved(cfg, cfg_file)