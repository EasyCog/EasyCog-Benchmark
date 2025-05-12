import torch
import torch.nn as nn

from DL_pipeline.models.exg_base_model import CBraMod_BaseModel, CBraMod_BaseModel_FeatureExtractor
from DL_pipeline.models.basic_modules import MLP
from DL_pipeline.models.EEGConformer import Conformer, Conformer_feature_extractor, Conformer_TaskCLS_CogScore
from DL_pipeline.models.reg_model import Regression_MLP, Regression_Transformer, Regression_Transformer_TaskScore
from data_processing.analysis_utils import BIAS_AUG_WEIGHT, get_biased_aug_times
from DL_pipeline.models.basic_modules import TaskAttentionAggregator
from torch.nn.utils.rnn import pad_sequence 
from utils.data_preproc import task_preprocessing, remove_common_features

from utils.utils import (
    check_shapes,
    get_data_dict_list_idx,
    to_device,
    convert_data_dict_to_list,
)
from utils.data_preproc import task_preprocessing
from DL_pipeline.dataset.data_aug import data_augmentation, apply_mixup_augmentation, mixup_from_subject
from DL_pipeline.models.basic_modules import FusionModule


class Joint_CLS_REG_Model_moco_v2(nn.Module):
    def __init__(self, 
                 cls_model_name="Conformer_TaskCLS_CogScore",
                 reg_model_name="Regression_MLP",
                 input_channels=16,
                 cls_emb_size=40, ###for the convformer model
                 cls_n_dim=200, ### for Transformer dimension
                 cls_depth=6,
                 n_classes=10,
                 REG_dim=2,
                 reg_depth=2,
                 reg_hidden_dims=[1024, 512, 256, 32],
                 input_length=375,
                 patch_sizes=[25, 16],
                 load_pretrained_backbone=False,
                 freeze_backbone=False,
                 pretrained_path=None,
                 dim=128,
                 K=8192,
                 m=0.999,
                 T=0.07,
                 aug_methods=None,
                 freeze_cls_model=False,
                 vlm_fusion_type=None,
                 reg_return_features=None,
                 ):
        super(Joint_CLS_REG_Model_moco_v2, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.aug_methods = aug_methods
        self.is_free_cls_model = freeze_cls_model
        self.reg_return_features = reg_return_features
        # if cls_model_name == "Conformer_TaskCLS_CogScore":

        ### pic-level for supervised contrastive learning
        if cls_model_name == "CBraMod_BaseModel":
            self.encoder_q = CBraMod_BaseModel_FeatureExtractor(
                input_channels=input_channels,  
                num_classes=n_classes,
                n_layer=cls_depth,
                n_dim=cls_n_dim,
                load_pretrained_backbone=load_pretrained_backbone,
                freeze_backbone=freeze_backbone,
                pretrained_path=pretrained_path,
            )
            self.final_layer = MLP(input_dim=cls_n_dim, hidden_dim=cls_n_dim, output_dim=n_classes, use_batch_norm=False)
            self.encoder_k = CBraMod_BaseModel_FeatureExtractor(
                input_channels=input_channels,  
                num_classes=n_classes,
                n_layer=cls_depth,
                n_dim=cls_n_dim,
                load_pretrained_backbone=load_pretrained_backbone,
                freeze_backbone=freeze_backbone,
                pretrained_path=pretrained_path,
            )

            self.projector_q_contrast = MLP(input_dim=cls_n_dim, hidden_dim=cls_n_dim, output_dim=cls_n_dim, use_batch_norm=True)
            # self.projector_k_contrast = MLP(input_dim=cls_n_dim, hidden_dim=cls_n_dim, output_dim=cls_n_dim, use_batch_norm=True)
            self.input_dim = cls_n_dim  ### 256


        self.vlm_fusion_type = vlm_fusion_type
        if vlm_fusion_type is not None:
            self.vlm_projection_q = nn.Linear(4096, cls_n_dim)
            self.vlm_fusion_q = FusionModule(
                exg_emb_dim=cls_n_dim,
                vlm_projected_dim=cls_n_dim,
                fused_emb_dim=cls_n_dim,
                fusion_type=vlm_fusion_type
            )
            self.vlm_projection_k = nn.Linear(4096, cls_n_dim)
            self.vlm_fusion_k = FusionModule(
                exg_emb_dim=cls_n_dim,
                vlm_projected_dim=cls_n_dim,
                fused_emb_dim=cls_n_dim,
                fusion_type=vlm_fusion_type
            )
            
            # Create cls_model as a reference to the encoder_q and other modules
            # This ensures that when cls_model is updated, the original modules are also updated
            # since they are the same objects in memory
            
        else:
            self.vlm_projection_q = nn.Identity()
            self.vlm_fusion_q = nn.Identity()
            self.vlm_projection_k = nn.Identity()
            self.vlm_fusion_k = nn.Identity()

        self.cls_model = nn.ModuleDict({
                'encoder': self.encoder_q,
                'vlm_projection': self.vlm_projection_q,
                'vlm_fusion': self.vlm_fusion_q,
                'projector': self.projector_q_contrast,
                'final_layer': self.final_layer
            })

        ### task-level to aggregate features from picture-level features
        if 'features_preprocess' in self.aug_methods.keys() and self.aug_methods['features_preprocess']['method'] == 'intra_task_attention':
            self.normal_task_attention = TaskAttentionAggregator(embed_dim=cls_n_dim, nhead=4, dim_feedforward=4*cls_n_dim, dropout=0.3, max_images=10)
            self.contrast_task_attention = TaskAttentionAggregator(embed_dim=cls_n_dim, nhead=4, dim_feedforward=4*cls_n_dim, dropout=0.3, max_images=10)
        

        task_total_dim = int(n_classes * self.input_dim)

        ### subject-level to aggregate features from task-level features
        if reg_model_name == "Regression_MLP":
            self.reg_model = Regression_MLP(
                input_dim=task_total_dim,
                    hidden_dims=reg_hidden_dims,
                    output_dim=REG_dim,
                )
        elif reg_model_name == "Regression_Transformer":
            self.reg_model = Regression_Transformer(
                input_dim=self.input_dim, output_dim=REG_dim, num_layers=reg_depth
            )
        elif reg_model_name == "Regression_Transformer_TaskScore":
            self.reg_model = Regression_Transformer_TaskScore(
                input_dim=self.input_dim, output_dim=REG_dim, n_layers=reg_depth
            )



        for module_q, module_k in zip([self.encoder_q, self.vlm_projection_q, self.vlm_fusion_q], 
                                    [self.encoder_k, self.vlm_projection_k, self.vlm_fusion_k]):
            for param_q, param_k in zip(module_q.parameters(), module_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        self.register_buffer('queue', torch.randn(cls_n_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer('label_queue', torch.full((K,), -1, dtype=torch.long))

        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        print(f"Initialized MoCo queue (features: {self.queue.shape}, labels: {self.label_queue.shape})")

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for module_q, module_k in zip([self.encoder_q, self.vlm_projection_q, self.vlm_fusion_q], 
                                    [self.encoder_k, self.vlm_projection_k, self.vlm_fusion_k]):
            for param_q, param_k in zip(module_q.parameters(), module_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels=None):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        indices_to_replace = torch.arange(ptr, ptr + batch_size) % self.K # 使用取模运算处理环绕
        self.queue[:, indices_to_replace] = keys.T
        if labels is not None:
            self.label_queue[indices_to_replace] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def get_model_device(self):
        return next(self.parameters()).device
    
    def train_cls_model(self, inputs):
        if self.vlm_fusion_type is not None:
            ExG, vlm_embeds = inputs
            exg_out = self.encoder_q.get_features(ExG)
            vlm_out = self.vlm_projection_q(vlm_embeds)
            out = self.vlm_fusion_q(exg_out, vlm_out)
            out = nn.functional.normalize(out, dim=1)
            out = self.final_layer(out)
            out = (out, 0)
        else:
            inputs = inputs[0]
            inputs = torch.tensor(
                inputs, dtype=torch.float32
            ).to(self.get_model_device())
            out = self.encoder_q.get_features(inputs)
            out = nn.functional.normalize(out, dim=1)
            out = self.final_layer(out)
            out = (out, 0)
        return out
    
    def freeze_cls_model(self):
        """Freeze the classification model by setting requires_grad=False for all parameters"""
        for mod in [self.encoder_q, self.vlm_projection_q, self.vlm_fusion_q]:
            for param in mod.parameters():
                param.requires_grad = True
    
    def unfreeze_cls_model(self):
        """Unfreeze the classification model by setting requires_grad=True for all parameters"""
        for mod in [self.encoder_q, self.vlm_projection_q, self.vlm_fusion_q]:
            for param in mod.parameters():
                param.requires_grad = True

    def get_cls_model_features(self, inputs, q_or_k='q'):
        device = self.get_model_device()
        modules = {
            'q': [self.encoder_q, self.vlm_projection_q, self.vlm_fusion_q],
            'k': [self.encoder_k, self.vlm_projection_k, self.vlm_fusion_k]
        }
        if self.vlm_fusion_type is not None:
            ExG, vlm_embeds = inputs
            if isinstance(ExG, torch.Tensor) is False:
                ExG = torch.tensor(
                    ExG, dtype=torch.float32
                ).to(device)
                
            else:
                ExG = ExG.to(device)
            if isinstance(vlm_embeds, torch.Tensor) is False:
                vlm_embeds = torch.tensor(
                    vlm_embeds, dtype=torch.float32
                ).to(device)
            else:
                vlm_embeds = vlm_embeds.to(device)
            exg_out = modules[q_or_k][0].get_features(ExG)
            vlm_out = modules[q_or_k][1](vlm_embeds)
            out = modules[q_or_k][2](exg_out, vlm_out)
            out = nn.functional.normalize(out, dim=1)
        else:
            inputs = inputs[0]
            if isinstance(inputs, torch.Tensor) is False:
                inputs = torch.tensor(
                    inputs, dtype=torch.float32
                ).to(device)
            else:
                inputs = inputs.to(device)
            out = modules[q_or_k][0].get_features(inputs)
            out = nn.functional.normalize(out, dim=1)
        return out
        

    def intra_task_attention(self, features_list):
        ### features_list: [task1_feats_array, task2_feats_array, ...]
        ### shapes:
        '''
        torch.Size([10, 200])
        torch.Size([10, 200])
        torch.Size([5, 200])
        torch.Size([5, 200])
        torch.Size([10, 200])
        torch.Size([6, 200])
        torch.Size([5, 200])
        torch.Size([10, 200])
        torch.Size([10, 200])
        torch.Size([1, 200])
        '''
        ###: task 2 / 3 / 6 as self-contrast task, use the same attention weight.
        ###: task 0 / 1 / 4 / 7 / 8  as normal task, use the same attention weight.
        ###: task 5 as normal task but different size, use can use the same attention weight as above
        ###: task 9 as resting task but only one sample, do not use attention.
        current_max_len = max([i.shape[0] for i in features_list])
        original_lengths = [t.shape[0] for t in features_list]

        padded_embeddings = pad_sequence(features_list, batch_first=True, padding_value=0.0)
        padding_mask = torch.arange(current_max_len)[None, :] >= torch.tensor(original_lengths)[:, None]
        padding_mask = padding_mask.to(padded_embeddings.device)
        
        aggregated_features = []
        for i in range(len(features_list)):
            if i in [2, 3, 6]:
                task_embedding = self.contrast_task_attention(padded_embeddings[i], padding_mask[i])
            elif i in [0, 1, 4, 5, 7, 8]:
                task_embedding = self.normal_task_attention(padded_embeddings[i], padding_mask[i])
            elif i == 9:
                task_embedding = features_list[i]

            aggregated_features.append(task_embedding)

        return aggregated_features

    def aggregate_features_across_tasks(self, features_list, method="concat"):
        if not check_shapes(features_list):
            ### the tasks have different elements, so we cannot aggregate them
            return features_list

        if method == "concat":
            features = torch.cat(features_list, dim=0)
            # Reshape to (1, -1) to match the original behavior
            features = features.view(1, -1)
        elif method == "mean":
            features = torch.stack(features_list, dim=0)
            features = torch.mean(features, dim=0, keepdim=True)
        elif method == "none":
            features = features_list
        else:
            raise ValueError(f"Invalid aggregation method: {method}")
        return features

    def convert_task_modalities_seqs(self, task_modalities_seqs):
        ### task_modalities_seqs: [[B, [[task1_m1_seq numpy array], [task2_m1_seq numpy array], ...]], [B, [task1_m2_seq numpy array], [task2_m2_seq numpy array], ...]], multi-modalities, single-view
        ### task_modalities_seqs[i][j] is the ExG sequence of the j-th task of the i-th subject
        ### task_modalities_seqs[i][j] is a numpy array of shape (n_samples, 375)
        ### target: [B, [[task1_m1_seq numpy array, task1_m2_seq numpy array, ...], [task2_m1_seq numpy array, task2_m2_seq numpy array, ...], ...]]
        
        # Get dimensions
        num_modalities = len(task_modalities_seqs)
        num_subjects = len(task_modalities_seqs[0])
        
        # Initialize the result structure
        result = [[] for _ in range(num_subjects)]
        # For each subject
        for subject_idx in range(num_subjects):
            # Get number of tasks for this subject (assuming same across modalities)
            num_tasks = len(task_modalities_seqs[0][subject_idx])
            # Initialize task list for this subject
            subject_tasks = [[] for _ in range(num_tasks)]
            
            # For each modality
            for modality_idx in range(num_modalities):
                # For each task
                for task_idx in range(num_tasks):
                    # Add the modality data to the appropriate task
                    subject_tasks[task_idx].append(task_modalities_seqs[modality_idx][subject_idx][task_idx])
            # Add all tasks for this subject to the result
            result[subject_idx] = subject_tasks
            
        return result
    

    def get_all_cls_features(self, task_modalities_seqs):
        ### task_modalities_seqs: [[B, [[task1_ExG_seq numpy array], [task2_ExG_seq numpy array], ...]]], multi-modalities, single-view
        ### task_modalities_seqs[i][j] is the ExG sequence of the j-th task of the i-th subject
        ### task_modalities_seqs[i][j] is a numpy array of shape (n_samples, 375)
        ### feats: [B, [task1_feats_array, task2_feats_array, ...]], each feats_array is a numpy array of shape (n_samples, n_dim)
        device = self.get_model_device()
        overall_feats = []

        # for modality in range(len(task_seqs)):
        converted_seqs = self.convert_task_modalities_seqs(task_modalities_seqs)
        for subject in range(len(converted_seqs)):
            feats = []
            for task in range(len(converted_seqs[subject])):
                task_modalities_array = converted_seqs[subject][task]
                exg_feats = self.get_cls_model_features(
                    task_modalities_array, q_or_k='q'
                )  ### [n_samples, n_dim]
                feats.append(exg_feats)
            if 'features_preprocess' in self.aug_methods.keys() and self.aug_methods['features_preprocess']['method'] == 'task_preprocessing':
                feats = task_preprocessing(feats, contrast_task=self.aug_methods['features_preprocess']['contrast_task'])
            
            if 'features_preprocess' in self.aug_methods.keys() and self.aug_methods['features_preprocess']['method'] == 'intra_task_attention':
                feats = task_preprocessing(feats, contrast_task=self.aug_methods['features_preprocess']['contrast_task'])
                feats = self.intra_task_attention(feats)
            if 'common_remove' in self.aug_methods.keys() and self.aug_methods['common_remove'] == True:
                feats = remove_common_features(feats)
            overall_feats.append(feats)
        return overall_feats
    

    @torch.no_grad()
    def convert_gt_to_tensor_list(self, gt_list):
        ### gt_list: [[m1_sample1, ...], [m2_sample1, ...], ...]
        ### convert to [m1_stacked_tensor, m2_stacked_tensor, ...]
        batch = []
        for i in range(len(gt_list)):
            if isinstance(gt_list[i][0], torch.Tensor):
                batch.append(torch.stack(gt_list[i], dim=0))
            elif isinstance(gt_list[i][0], float):
                batch.append(
                    torch.tensor(
                        gt_list[i], dtype=torch.float32, device=self.get_model_device()
                    )
                )
        return batch
    
    @torch.no_grad()
    def convert_gt_list_to_batch(self, gt_list):
        ### gt_list: [m1_stacked_tensor, m2_stacked_tensor, ...], length is the number of subjects
        ### convert to: [batch_m1_stacked_tensor, batch_m2_stacked_tensor, ...], each first dimension is the number of subjects
        batch = []
        num_modalities = len(gt_list[0])
        num_subjects = len(gt_list)
        for j in range(num_modalities):
            batch.append(
                torch.stack([gt_list[i][j] for i in range(num_subjects)], dim=0)
            )
        return batch
    
    @torch.no_grad()
    def convert_imbalanced_feat_gt(self, feat_list, gt_list):
        ### feat_list: [m1_stacked_tensor, m2_stacked_tensor, ...], length is the number of subjects
        ### gt_list: [m1_stacked_tensor, m2_stacked_tensor, ...], length is the number of subjects
        subject_augtimes = [i.shape[0] for i in feat_list]
        max_id = subject_augtimes.index(max(subject_augtimes))
        sum_augtimes = sum(subject_augtimes)
        if sum_augtimes % len(feat_list) != 0:
            additional_times = len(feat_list) - sum_augtimes % len(feat_list)
        else:
            additional_times = 0

        subject_augtimes[max_id] += additional_times
        for i in range(additional_times):
            feat_list[max_id] = torch.cat([feat_list[max_id], feat_list[max_id][-1].unsqueeze(0)], dim=0)
            for modality in range(len(gt_list[max_id])):    
                gt_list[max_id][modality] = torch.cat([gt_list[max_id][modality], gt_list[max_id][modality][-1].unsqueeze(0)], dim=0)

        residual_times = [subject_augtimes[max_id]-i for i in subject_augtimes]
        for i in range(len(residual_times)):
            if residual_times[i] > 0:
                mixup_data, mixup_gt = mixup_from_subject(feat_list[max_id], gt_list[max_id], residual_times[i])
                feat_list[i] = torch.cat([feat_list[i], mixup_data], dim=0)
                for modality in range(len(gt_list[i])):
                    gt_list[i][modality] = torch.cat([gt_list[i][modality], mixup_gt[modality]], dim=0)

        return feat_list, gt_list


    # @torch.no_grad()
    # def form_subject_features(self, overall_feats):
    #     ### overall_feats: [[B, [task1_feats_array, task2_feats_array, ...]]], multi-views, each feats_array is a numpy array of shape (n_samples, n_dim)
        
    #     ### return: [B, N_tasks, 1, N_dim]
    #     subject_feats = []
    #     for subject in range(len(overall_feats)):
    #         subject_feats.append(overall_feats[subject])
    #     return subject_feats



    @torch.no_grad()
    def augment_cls_features(self, overall_feats, y, aug_methods=None, mode="train"):
        ### overall_feats: [[B, [task1_feats_array, task2_feats_array, ...]]], multi-views, each feats_array is a numpy array of shape (n_samples, n_dim)
        aug_feats = []
        aug_gts = []
     
        for subject in range(len(overall_feats)):
            subject_feats = overall_feats[subject]
            input_dict = {"features-value": subject_feats}
            gt_dict = get_data_dict_list_idx(y, subject)
            if 'bias_aug' in self.aug_methods.keys() and self.aug_methods['bias_aug'] == True and mode == 'train':
                aug_times_weight = get_biased_aug_times(gt_dict['MoCA-value'], n=8)
            else:
                aug_times_weight = 1

            aug_subject_feats, aug_y = data_augmentation(
                input_dict, gt_dict, aug_methods, mode, weight=aug_times_weight
            )
            aug_y = convert_data_dict_to_list(aug_y)
            aug_y = to_device(aug_y, self.get_model_device())
            aug_gts.append(self.convert_gt_to_tensor_list(aug_y))       # [n_subjects, n_labels, n_augs]
            aug_subject_feats = torch.stack(aug_subject_feats["features"], dim=0)       # [n_augs, n_task, n_dim]
            aug_feats.append(aug_subject_feats)
        subject_aug = [i.shape[0] for i in aug_feats]
        if max(subject_aug) != min(subject_aug):
            aug_feats, aug_gts = self.convert_imbalanced_feat_gt(aug_feats, aug_gts)
        aug_feats = torch.stack(aug_feats, dim=0)  ### [B, n_feats_augs, n_tasks, n_dim]
        aug_gts = self.convert_gt_list_to_batch(aug_gts)

        if 'mix_up' in aug_methods[mode]['features'].keys():
            if 'mode' in aug_methods[mode]['features']['mix_up'].keys():
                aug_feats, aug_gts = apply_mixup_augmentation(aug_feats, aug_gts, aug_methods[mode]['features']['mix_up']['aug_times'], mode=aug_methods[mode]['features']['mix_up']['mode'])
            else:
                aug_feats, aug_gts = apply_mixup_augmentation(aug_feats, aug_gts, aug_methods[mode]['features']['mix_up']['aug_times'], mode='uniform')
        return aug_feats, aug_gts

    def random_task_mask_transformer(self, overall_feats, mask_ratio=0.3):
        ### overall_feats: [B, n_feats_augs, n_tasks, n_dim]
        B, n_feats_augs, n_tasks, n_dim = overall_feats.shape
        mask = torch.rand(B*n_feats_augs,n_tasks, device=overall_feats.device) < mask_ratio
        mask = torch.cat((torch.zeros(B*n_feats_augs,1, dtype=torch.bool).to(overall_feats.device), mask), dim=1)
        return mask
    
    def predict(self, x, y, mode='test_cls'):
        if 'cls' in mode:
            return self.train_cls_model(x)
        elif 'reg' in mode:
            overall_feats = self.get_all_cls_features(x)
            aug_feats, aug_gts = self.augment_cls_features(
                overall_feats, y, self.aug_methods, "test"
            )
            return self.train_reg_model(aug_feats), aug_gts
        
    def train_reg_model(self, overall_feats):
        ### overall_feats: [B, n_feats_augs, n_tasks, n_dim]
        B, n_feats_augs, n_tasks, n_dim = overall_feats.shape

        # Reshape to merge batch and augmentation dimensions
        # From [B, n_feats_augs, n_tasks, n_dim] to [B*n_feats_augs, n_tasks, n_dim]
        merged_feats = overall_feats.view(B * n_feats_augs, n_tasks, n_dim)

        # Pass through the regression model
        outputs = self.reg_model(merged_feats)
        # Reshape back to separate batch and augmentation dimensions
        # From [B*n_feats_augs, output_dim] to [B, n_feats_augs, output_dim]
        outputs = [out.view(B, n_feats_augs, -1).squeeze(-1) for out in outputs]

        return outputs

    def convert_cls_labels(self, labels):
        if 'Subject_id' in labels.keys():
            subject_id = labels['Subject_id']
        else:
            subject_id = labels['Subject_Category']
        task_id = labels['Task_id']    
        return subject_id * 10 + task_id

    def contrast_features_labels(self, q, k, labels):
        
        labels = self.convert_cls_labels(labels).long().to(self.get_model_device())
        q = self.get_cls_model_features(q, q_or_k='q')
        p_q = self.projector_q_contrast(q)
        p_q = nn.functional.normalize(p_q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.get_cls_model_features(k, q_or_k='k')
            p_k = nn.functional.normalize(k, dim=1)

        queue_ = self.queue.clone().detach()  # DxK
        labels_ = self.label_queue.clone().detach()
    
        queue_enqueue = torch.cat([p_k.T, queue_], dim=1)  # Dx(B+K)
        labels_enqueue = torch.cat([labels, labels_], dim=0)  # B+K

        mask = torch.eq(labels[:, None], labels_enqueue[:, None].T)  # bx(B+K)

        logits = torch.einsum('nc,ck->nk', [p_q, queue_enqueue.clone().detach()]).div(self.T)

        loss_sup = (-torch.log_softmax(logits, dim=1) * mask).sum(dim=-1, keepdim=True).div(mask.sum(dim=-1, keepdim=True) + 1e-5)
        loss = loss_sup.mean()

        # dequeue and enqueue
        self._dequeue_and_enqueue(p_k, labels)
        
        return q, loss
    
    def contrast_features_sample(self, q, k):
        q = self.get_cls_model_features(q, q_or_k='q')
        p_q = self.projector_q_contrast(q)
        p_q = nn.functional.normalize(p_q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.get_cls_model_features(k, q_or_k='k')
            p_k = nn.functional.normalize(k, dim=1)

        l_batch_s = torch.mm(p_q, k.transpose(0,1))
        l_queue_s = torch.mm(p_q, self.queue.clone().detach())

        logits_s = torch.cat([l_batch_s, l_queue_s], dim=1)
        logits_s /= self.T

        labels_s = torch.arange(p_q.shape[0], dtype=torch.long).to(device = p_q.device)
        self._dequeue_and_enqueue(k) ### 之前忘了。。。。
        return q, logits_s, labels_s
    

    def contrast_features(self, q, k, labels):
        if labels is None:
            return self.contrast_features_sample(q, k)
        else:
            return self.contrast_features_labels(q, k, labels)



    def forward(self, qk, y=None, mode="train_cls"):
        if 'test' in mode:
            ### only q is input as qk
            return self.predict(qk, y, mode=mode)
        else:
            
            if mode == 'train_cls':
                if self.is_free_cls_model:
                    self.unfreeze_cls_model()
                q, k = qk
                if y is None:
                    q, logits_s, labels_s = self.contrast_features_sample(q, k)
                    cates = self.final_layer(q)
                    return cates, logits_s, labels_s
                else:
                    q, loss = self.contrast_features_labels(q, k, y)
                    cates = self.final_layer(q)
                    return cates, loss
            elif mode == 'train_reg':
                if self.vlm_fusion_type is None:
                    if len(qk) == 2:
                        q, k = qk
                    else:
                        q = qk
                else:
                    if len(qk) == 2:
                        q = qk ### qk is a list of [ExG, vlm_embeds]
                if self.is_free_cls_model:
                    self.freeze_cls_model()
                overall_feats = self.get_all_cls_features(q)
                aug_feats, aug_gts = self.augment_cls_features(
                    overall_feats, y, self.aug_methods, "train"
                )
                if self.reg_return_features is None:
                    return self.train_reg_model(aug_feats), aug_gts, None
                elif self.reg_return_features == 'aug_feats':
                    return self.train_reg_model(aug_feats), aug_gts, aug_feats
                elif self.reg_return_features == 'overall_feats':
                    return self.train_reg_model(aug_feats), aug_gts, overall_feats
