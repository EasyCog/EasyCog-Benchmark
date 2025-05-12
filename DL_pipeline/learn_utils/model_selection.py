from DL_pipeline.models.eye_tracking_model import *
from DL_pipeline.models.exg_backbone import *
from DL_pipeline.models.exg_multitask import *
from DL_pipeline.models.exg_multitask_yx import CBraMod_ExGMultiTask_FeatFusion
from DL_pipeline.models.exg_multitask_simclr import CBraMod_ExGMultiTaskSimCLR
from DL_pipeline.models.exg_base_model import CBraMod_BaseModel
from DL_pipeline.models.exg_clip import CBraMod_ExGCLIP, CBraMod_ExG_SharedLayer_CLIP
from DL_pipeline.models.reg_model import Regression_MLP, Regression_CNN, Regression_MLP_Split, Regression_Transformer, Regression_MLP_Subscore, Regression_Transformer_TaskScore, Regression_Transformer_4head
from DL_pipeline.models.EEGConformer import Conformer, Conformer_TaskCLS_CogScore, Conformer_GCN_TaskCLS_CogScore
from DL_pipeline.models.moco import MoCo
from DL_pipeline.models.joint_model import Joint_CLS_REG_Model
from DL_pipeline.models.joint_model_yx import Joint_CLS_REG_Model_yx
from DL_pipeline.models.moco_joint_model import Joint_CLS_REG_Model_moco
from DL_pipeline.models.moco_joint_model_v2 import Joint_CLS_REG_Model_moco_v2

from DL_pipeline.models.Baselines.adaptive_spatiotemporal_encoding_network import CombinedGCNCNN
import utils.mask_pretrain_models as mask_pretrain_models


__all__ = ['get_model']
    
def get_model(cfg):
    """
    Factory function to create model instance based on configuration.
    """
    freeze_cls_model = cfg['model']['freeze_cls_model'] if 'freeze_cls_model' in cfg['model'].keys() else False
    vlm_fusion_type = cfg['model']['vlm_fusion_type'] if 'vlm_fusion_type' in cfg['model'].keys() else None
    reg_return_features = cfg['model']['reg_return_features'] if 'reg_return_features' in cfg['model'].keys() else None
    if cfg['model']['name'] == 'EyetrackingModel':
        model = EyetrackingModel(
            input_channels=cfg['model']['input_channels'],
            nhead=cfg['model']['nhead'],
            num_classes=cfg['model']['num_classes']
        )
    
    elif cfg['model']['name'] == 'EyetrackingModel_MultiTask':
        model = EyetrackingModel_MultiTask(
            input_channels=cfg['model']['input_channels'],
            nhead=cfg['model']['nhead'],
            num_classes=cfg['model']['num_classes']
        )
    elif cfg['model']['name'] == 'CBraMod_ExGBackbone':
        model = CBraMod_ExGBackbone(
            input_channels=cfg['model']['input_channels'],
            num_classes=cfg['model']['num_classes'],
            n_layer=cfg['model']['n_layer'],
            load_pretrained_backbone=cfg['model']['load_pretrained_backbone'],
            freeze_backbone=cfg['model']['freeze_backbone'],
            pretrained_path=cfg['model']['backbone_weight_path']
        )
    elif cfg['model']['name'] == 'CBraMod_BaseModel':
        model = CBraMod_BaseModel(
            input_channels=cfg['model']['input_channels'],
            num_classes=cfg['model']['num_classes'],
            n_layer=cfg['model']['n_layer'],
            n_dim=cfg['model']['n_dim'],
            load_pretrained_backbone=cfg['model']['load_pretrained_backbone'],
            freeze_backbone=cfg['model']['freeze_backbone'],
            pretrained_path=cfg['model']['backbone_weight_path']
        )
    elif cfg['model']['name'] == 'CBraMod_ExGMultiTask':
        model = CBraMod_ExGMultiTask(
            input_channels=cfg['model']['input_channels'],
            num_classes=cfg['model']['num_classes'],
            n_layer=cfg['model']['n_layer'],
            load_pretrained_backbone=cfg['model']['load_pretrained_backbone'],
            freeze_backbone=cfg['model']['freeze_backbone'],
            pretrained_path=cfg['model']['backbone_weight_path']
        )
    elif cfg['model']['name'] == 'CBraMod_ExGBackbone_FeatFusion':
        model = CBraMod_ExGMultiTask_FeatFusion(
            input_channels=cfg['model']['input_channels'],
            num_classes=cfg['model']['num_classes'],
            n_layer=cfg['model']['n_layer'],
            load_pretrained_backbone=cfg['model']['load_pretrained_backbone'],
            freeze_backbone=cfg['model']['freeze_backbone'],
            if_STFT=cfg['model']['if_STFT'],
            if_PCA=cfg['model']['if_PCA'],
            if_RAW=cfg['model']['if_Raw_feat'],
            if_ET=cfg['model']['if_ET'],
            ET_model=cfg['model']['ET_model'],
            pe_type = cfg['model']['pe_type'],
            n_feat_len=cfg['model']['n_feat_len'],
            pretrained_path=cfg['model']['backbone_weight_path'] 
        )

    elif cfg['model']['name'] == 'CBraMod_ExGMultiTaskSimCLR':
        model = CBraMod_ExGMultiTaskSimCLR(
            input_channels=cfg['model']['input_channels'],
            num_classes=cfg['model']['num_classes'],
            n_layer=cfg['model']['n_layer'],
            n_dim=cfg['model']['n_dim'],
            load_pretrained_backbone=cfg['model']['load_pretrained_backbone'],
            freeze_backbone=cfg['model']['freeze_backbone'],
            pretrained_path=cfg['model']['backbone_weight_path']
        )
    elif cfg['model']['name'] == 'CBraMod_ExGCLIP':
        model = CBraMod_ExGCLIP(
            input_channels=cfg['model']['input_channels'],
            num_classes=cfg['model']['num_classes'],
            n_layer=cfg['model']['n_layer'],
            n_dim=cfg['model']['n_dim'],
            load_pretrained_backbone=cfg['model']['load_pretrained_backbone'],
            freeze_backbone=cfg['model']['freeze_backbone'],
            pretrained_path=cfg['model']['backbone_weight_path']
        )
    elif cfg['model']['name'] == 'CBraMod_ExG_SharedLayer_CLIP':
        model = CBraMod_ExG_SharedLayer_CLIP(
            input_channels=cfg['model']['input_channels'],
            num_classes=cfg['model']['num_classes'],
            n_layer=cfg['model']['n_layer'],
            n_dim=cfg['model']['n_dim'],
            load_pretrained_backbone=cfg['model']['load_pretrained_backbone'],
            freeze_backbone=cfg['model']['freeze_backbone'],
            pretrained_path=cfg['model']['backbone_weight_path']
        )
    elif cfg['model']['name'] == 'CBraMod':
        model = CBraMod(
            in_dim=cfg['model']['in_dim'],
            out_dim=cfg['model']['out_dim'],
            d_model=cfg['model']['d_model'],
            dim_feedforward=cfg['model']['dim_feedforward'],
            seq_len=cfg['model']['seq_len'],
            n_layer=cfg['model']['n_layer'],
            nhead=cfg['model']['nhead']
        )
    elif cfg['model']['name'] == 'CBraMod_TaskCLS_CogScoreReg':
        model = CBraMod_TaskCLS_CogScoreReg(
            input_channels=cfg['model']['input_channels'],
            CLS_classes=cfg['model']['num_classes'],
            REG_dim=cfg['model']['num_reg_dims'],
            n_layer=cfg['model']['n_layer'],
            n_dim=cfg['model']['n_dim'],
            load_pretrained_backbone=cfg['model']['load_pretrained_backbone'],
            freeze_backbone=cfg['model']['freeze_backbone'],
            pretrained_path=cfg['model']['backbone_weight_path']
        )

    elif cfg['model']['name'] == 'Conformer':
        model = Conformer(
            emb_size=cfg['model']['emb_size'],
            depth=cfg['model']['depth'],
            n_classes=cfg['model']['num_classes']
        )
    elif cfg['model']['name'] == 'Conformer_TaskCLS_CogScore':
        model = Conformer_TaskCLS_CogScore(
            emb_size=cfg['model']['emb_size'],
            depth=cfg['model']['depth'],
            n_classes=cfg['model']['num_classes'],
            REG_dim=cfg['model']['num_reg_dims'],
            patch_sizes=cfg['model']['patch_sizes']
        )
    elif cfg['model']['name'] == 'Conformer_GCN_TaskCLS_CogScore':
        model = Conformer_GCN_TaskCLS_CogScore(
            emb_size=cfg['model']['emb_size'],
            depth=cfg['model']['depth'],
            n_classes=cfg['model']['num_classes'],
            REG_dim=cfg['model']['num_reg_dims'],
            use_dtf_template=cfg['model']['use_dtf_template'],
            patch_sizes=cfg['model']['patch_sizes']
        )

    elif cfg['model']['name'] == 'Regression_MLP':
        model = Regression_MLP(
            input_dim = cfg['model']['input_dim'],
            hidden_dims = cfg['model']['hidden_dims'],
            output_dim = cfg['model']['output_dim'],
            activation= cfg['model']['activation'] if 'activation' in cfg['model'].keys() else 'sigmoid',
        )
    elif cfg['model']['name'] == 'Regression_CNN':
        model = Regression_CNN(
            input_dim = cfg['model']['input_dim'],
            feat_len = cfg['model']['feat_len'],
            output_dim = cfg['model']['output_dim'],
            activation= cfg['model']['activation'] if 'activation' in cfg['model'].keys() else 'sigmoid',
        )
    elif cfg['model']['name'] == 'Regression_MLP_Split':
        model = Regression_MLP_Split(
            input_dim = cfg['model']['input_dim'],
            feat_len = cfg['model']['feat_len'],
            output_dim = cfg['model']['output_dim'],
            activation= cfg['model']['activation'] if 'activation' in cfg['model'].keys() else 'sigmoid',
        )
    elif cfg['model']['name'] == 'Regression_Transformer':
        model = Regression_Transformer(
            input_dim = cfg['model']['input_dim'],
            output_dim = cfg['model']['output_dim'],
            activation= cfg['model']['activation'] if 'activation' in cfg['model'].keys() else 'sigmoid',
        )
    elif cfg['model']['name'] == 'Regression_Transformer_TaskScore':
        model = Regression_Transformer_TaskScore(
            input_dim = cfg['model']['input_dim'],
            output_dim = cfg['model']['output_dim'],
            n_layers = cfg['model']['n_layers'],
            activation= cfg['model']['activation'] if 'activation' in cfg['model'].keys() else 'sigmoid',
        )
    elif cfg['model']['name'] == 'Regression_MLP_Subscore':
        model = Regression_MLP_Subscore(
            input_dim = cfg['model']['input_dim'],
            output_dim = cfg['model']['output_dim'],
            moca_subscore_dim = cfg['model']['moca_subscore_dim'],
            mmse_subscore_dim = cfg['model']['mmse_subscore_dim'],
            activation= cfg['model']['activation'] if 'activation' in cfg['model'].keys() else 'sigmoid',
        )
    elif cfg['model']['name'] == 'MoCo':
        model = MoCo(
            base_model_name=cfg['model']['base_model_name'],
            input_channels=cfg['model']['input_channels'],
            num_classes=cfg['model']['num_classes'],
            n_layer=cfg['model']['depth'],
            REG_dim=cfg['model']['num_reg_dims']
        )

    #### Joint CLS and REG Model Training
    elif cfg['model']['name'] == 'Joint_CLS_REG_Model':
        # model = Joint_CLS_REG_Model(
        #     cls_model_name=cfg['model']['cls_model_name'],
        #     reg_model_name=cfg['model']['reg_model_name'],
        #     input_channels=cfg['model']['input_channels'],
        #     emb_size=cfg['model']['emb_size'],
        #     depth=cfg['model']['depth'],
        #     n_classes=cfg['model']['num_classes'],
        #     REG_dim=cfg['model']['num_reg_dims'],
        #     input_length=cfg['model']['input_length'],
        #     patch_sizes=cfg['model']['patch_sizes'],
        #     aug_methods=cfg['model']['aug_methods']
        # )
        if cfg['model']['load_pretrained_backbone'] is True:
            if cfg['model']['backbone_weight_path'] == 'from utils.mask_pretrain_models.py':
                if cfg['data_type'] == 'clean_0426':
                    backbone_weight = mask_pretrain_models.MASK_PRETRAIN_MODEL_PATH_DICT_CLEAN_DATA[cfg['test_user_list_idx']]
                elif cfg['data_type'] == 'all_0426':
                    backbone_weight = mask_pretrain_models.MASK_PRETRAIN_MODEL_PATH_DICT_ALL_DATA[cfg['test_user_list_idx']]
                else:
                    raise ValueError(f"Invalid data type: {cfg['data_type']}")
                cfg['model']['backbone_weight_path'] = backbone_weight
                
        model = Joint_CLS_REG_Model(
            cls_model_name=cfg['model']['cls_model_name'],
            reg_model_name=cfg['model']['reg_model_name'],
            input_channels=cfg['model']['input_channels'],
            cls_emb_size=cfg['model']['emb_size'],
            cls_n_dim=cfg['model']['cls_n_dim'],
            cls_depth=cfg['model']['cls_depth'],
            n_classes=cfg['model']['num_classes'],
            REG_dim=cfg['model']['num_reg_dims'],
            reg_hidden_dims=cfg['model']['reg_hidden_dims'],
            input_length=cfg['model']['input_length'],
            patch_sizes=cfg['model']['patch_sizes'],
            load_pretrained_backbone=cfg['model']['load_pretrained_backbone'],
            freeze_backbone=cfg['model']['freeze_backbone'],
            pretrained_path=cfg['model']['backbone_weight_path'],
            aug_methods=cfg['model']['aug_methods'],
            freeze_cls_model=freeze_cls_model,
            vlm_fusion_type=vlm_fusion_type
        )
    elif cfg['model']['name'] == 'Joint_CLS_REG_Model_yx':
        freeze_cls_model = cfg['model']['freeze_cls_model'] if 'freeze_cls_model' in cfg['model'].keys() else False
        model = Joint_CLS_REG_Model_yx(
            cls_model_name=cfg['model']['cls_model_name'],
            reg_model_name=cfg['model']['reg_model_name'],
            input_channels=cfg['model']['input_channels'],
            emb_size=cfg['model']['emb_size'],
            depth=cfg['model']['depth'],
            n_classes=cfg['model']['num_classes'],
            REG_dim=cfg['model']['num_reg_dims'],
            input_length=cfg['model']['input_length'],
            patch_sizes=cfg['model']['patch_sizes'],
            aug_methods=cfg['model']['aug_methods'],
            freeze_cls_model=freeze_cls_model
        )

    elif cfg['model']['name'] == 'Joint_CLS_REG_Model_moco':
        freeze_cls_model = cfg['model']['freeze_cls_model'] if 'freeze_cls_model' in cfg['model'].keys() else False
        if cfg['model']['load_pretrained_backbone'] is True:
            if cfg['model']['backbone_weight_path'] == 'from utils.mask_pretrain_models.py':
                if cfg['data_type'] == 'clean_0426':
                    backbone_weight = mask_pretrain_models.CLEAN_DATA_MASK_PRETRAIN_MODEL_PATH[cfg['test_user_list_idx']]
                elif cfg['data_type'] == 'all_0426':
                    backbone_weight = mask_pretrain_models.ALL_DATA_MASK_PRETRAIN_MODEL_PATH[cfg['test_user_list_idx']]
                else:
                    raise ValueError(f"Invalid data type: {cfg['data_type']}")
                cfg['model']['backbone_weight_path'] = backbone_weight
        model = Joint_CLS_REG_Model_moco(
            cls_model_name=cfg['model']['cls_model_name'],
            reg_model_name=cfg['model']['reg_model_name'],
            input_channels=cfg['model']['input_channels'],
            cls_emb_size=cfg['model']['cls_emb_size'],
            cls_n_dim=cfg['model']['cls_n_dim'],
            cls_depth=cfg['model']['cls_depth'],
            n_classes=cfg['model']['num_classes'],
            REG_dim=cfg['model']['num_reg_dims'],
            reg_hidden_dims=cfg['model']['reg_hidden_dims'],
            input_length=cfg['model']['input_length'],
            patch_sizes=cfg['model']['patch_sizes'],
            load_pretrained_backbone=cfg['model']['load_pretrained_backbone'],
            freeze_backbone=cfg['model']['freeze_backbone'],
            pretrained_path=cfg['model']['backbone_weight_path'],
            aug_methods=cfg['model']['aug_methods'],
            T=cfg['model']['T'],
            m=cfg['model']['m'],
            K=cfg['model']['K'],
            freeze_cls_model=freeze_cls_model,
        )
    elif cfg['model']['name'] == 'Joint_CLS_REG_Model_moco_v2':
        if cfg['model']['load_pretrained_backbone'] is True:
            if cfg['model']['backbone_weight_path'] == 'from utils.mask_pretrain_models.py':
                if cfg['data_type'] == 'clean_0426':
                    backbone_weight = mask_pretrain_models.MASK_PRETRAIN_MODEL_PATH_DICT_CLEAN_DATA[cfg['test_user_list_idx']]
                elif cfg['data_type'] == 'all_0426':
                    backbone_weight = mask_pretrain_models.MASK_PRETRAIN_MODEL_PATH_DICT_ALL_DATA[cfg['test_user_list_idx']]
                else:
                    raise ValueError(f"Invalid data type: {cfg['data_type']}")
                cfg['model']['backbone_weight_path'] = backbone_weight
        model = Joint_CLS_REG_Model_moco_v2(
            cls_model_name=cfg['model']['cls_model_name'],
            reg_model_name=cfg['model']['reg_model_name'],
            input_channels=cfg['model']['input_channels'],
            cls_emb_size=cfg['model']['cls_emb_size'],
            cls_n_dim=cfg['model']['cls_n_dim'],
            cls_depth=cfg['model']['cls_depth'],
            n_classes=cfg['model']['num_classes'],
            REG_dim=cfg['model']['num_reg_dims'],
            reg_hidden_dims=cfg['model']['reg_hidden_dims'],
            input_length=cfg['model']['input_length'],
            patch_sizes=cfg['model']['patch_sizes'],
            load_pretrained_backbone=cfg['model']['load_pretrained_backbone'],
            freeze_backbone=cfg['model']['freeze_backbone'],
            pretrained_path=cfg['model']['backbone_weight_path'],
            aug_methods=cfg['model']['aug_methods'],
            T=cfg['model']['T'],
            m=cfg['model']['m'],
            K=cfg['model']['K'],
            freeze_cls_model=freeze_cls_model,
            vlm_fusion_type=vlm_fusion_type,
            reg_return_features=cfg['model']['reg_return_features']
        )

    elif cfg['model']['name'] == 'CombinedGCNCNN':
        model = CombinedGCNCNN(
            output_dim=cfg['model']['output_dim'],
            CLS_or_Reg=cfg['model']['CLS_or_Reg']
        )
    else:
        raise NotImplementedError(f"Model {cfg['model']['name']} not implemented")
    
    return model

