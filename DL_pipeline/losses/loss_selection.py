from DL_pipeline.losses.losses import *

def get_loss(cfg, key='Loss'):
    if cfg[key]['name'] == 'CrossEntropy':
        return Cross_Entropy()
    elif cfg[key]['name'] == 'Cross_Entropy_L1':
        return Cross_Entropy_L1(cfg[key]['params']['alpha'])
    elif cfg[key]['name'] == 'Cross_Entropy_L1_Task_score':
        return Cross_Entropy_L1_Task_score(cfg[key]['params']['alpha'], cfg[key]['params']['beta'])
    elif cfg[key]['name'] == 'Cross_Entropy_L1_Task_score_EyeTracking':
        return Cross_Entropy_L1_Task_score_EyeTracking(cfg[key]['params']['alpha'], cfg[key]['params']['beta'], cfg[key]['params']['gamma'])
    elif cfg[key]['name'] == 'Cross_Entropy_smooth_L1':
        return Cross_Entropy_smooth_L1(cfg[key]['params']['alpha'])
    elif cfg[key]['name'] == 'Cross_Entropy_L1_MoCA_MMSE':
        return Cross_Entropy_L1_MoCA_MMSE(cfg[key]['params']['alpha'], cfg[key]['params']['beta'])
    elif cfg[key]['name'] == 'ContrastiveLoss':
        return ContrastiveLoss(
            alpha=cfg[key]['params']['alpha'],
            beta=cfg[key]['params']['beta'],
            gamma=cfg[key]['params']['gamma'],
            theta=cfg[key]['params']['theta'],
            temperature=cfg[key]['params']['temperature']
        )
    elif cfg[key]['name'] == 'Cross_Entropy_L1_Contrastive_Loss':
        return Cross_Entropy_L1_Contrastive_Loss(
            alpha=cfg[key]['params']['alpha'],
            beta=cfg[key]['params']['beta'],
            gamma=cfg[key]['params']['gamma'],
            Reg_dim=cfg[key]['params']['Reg_dim']
        )
    elif cfg[key]['name'] == 'Cross_Entropy_Contrastive_Loss':
        return Cross_Entropy_Contrastive_Loss(
            alpha=cfg[key]['params']['alpha']
        )
    elif cfg[key]['name'] == 'Cross_Entropy_Contrastive_Loss_moco_v2':
        return Cross_Entropy_Contrastive_Loss_moco_v2(
            alpha=cfg[key]['params']['alpha']
        )
    
    elif cfg[key]['name'] == 'CLIP_Score_Loss':
        return CLIP_Score_Loss(
            alpha=cfg[key]['params']['alpha'],
            beta=cfg[key]['params']['beta'],
            gamma=cfg[key]['params']['gamma']
        )
    elif cfg[key]['name'] == 'MSE_Loss':
        return MSE_Loss()
    elif cfg[key]['name'] == 'reg_L1_Loss':
        return reg_L1_Loss(cfg[key]['params']['alpha'])
    elif cfg[key]['name'] == 'reg_L1_Loss_Similarity_CE':
        return reg_L1_Loss_Similarity_CE(cfg[key]['params']['alpha'], cfg[key]['params']['beta'], cfg[key]['params']['gamma'], cfg[key]['params']['theta'])
    elif cfg[key]['name'] == 'reg_L2_Loss_shrinkage':
        return reg_L2_Loss_shrinkage(cfg[key]['params']['alpha'])
    elif cfg[key]['name'] == 'reg_L1_Loss_Subscore_Similarity':
        return reg_L1_Loss_Subscore_Similarity(cfg[key]['params']['alpha'], cfg[key]['params']['beta'], cfg[key]['params']['gamma'], cfg[key]['params']['theta'], cfg[key]['params']['delta'])
    elif cfg[key]['name'] == 'reg_LDS_L1_Loss_Subscore_Similarity':
        return reg_LDS_L1_Loss_Subscore_Similarity(cfg[key]['params']['alpha'], cfg[key]['params']['beta'], cfg[key]['params']['gamma'], cfg[key]['params']['theta'], cfg[key]['params']['delta'])
    elif cfg[key]['name'] == 'reg_L1_RnC_Loss_Subscore_Similarity':
        return reg_L1_RnC_Loss_Subscore_Similarity(cfg[key]['params']['alpha'], cfg[key]['params']['beta'], cfg[key]['params']['gamma'], cfg[key]['params']['theta'], 
                                                   cfg[key]['params']['delta'], cfg[key]['params']['eta'], cfg[key]['params']['temperature'])
    else:
        raise NotImplementedError(f"Loss {cfg[key]['name']} not implemented")