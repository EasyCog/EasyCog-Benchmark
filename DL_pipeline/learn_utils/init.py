import os
import torch
from utils import logger, line_seg, get_user_list
import random
import numpy as np
from DL_pipeline.learn_utils import model_selection

__all__ = ['init_device', 'init_model']

def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        # torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info('Running on GPU%d' % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info('Running on CPU')

    return device, pin_memory

def init_model(cfg):
    model = model_selection.get_model(cfg)
    if cfg['model']['pretrained'] != None:
        assert os.path.isfile(cfg['model']['pretrained'])
        state_dict = torch.load(
            cfg['model']['pretrained'], map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict, strict=True)
        logger.info('pretrained model loaded from {}'.format(
            cfg['model']['pretrained']))

    # Model info logging
    logger.info(
        f'=> Model Name: {cfg["model"]["name"]} [pretrained: {cfg["model"]["pretrained"]}]')
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    return model

def get_model(cfg, model_dict, subject):
    """
    model_dict:{
        test_users_idx: model_path
    }
    """
    user_list_idx = get_user_list(subject)
    if user_list_idx is None:
        raise ValueError(f"Subject {subject} not found in any test user list")
    model_path = model_dict[str(user_list_idx)]
    cfg['model']['pretrained'] = model_path
    from DL_pipeline.learn_utils import init_model
    model = init_model(cfg)
    return model