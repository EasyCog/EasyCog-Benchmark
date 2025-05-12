import torch
from torch import nn as nn
from torch.nn import functional as F
from DL_pipeline.losses.SupContrastLoss import SupConLoss
from DL_pipeline.losses.RankNContrastLoss import RnCLoss
from data_processing.analysis_utils import MOCA_TASK_SCORE_MAX, MMSE_TASK_SCORE_MAX
from DL_pipeline.learn_utils.LDS import calculate_eff_label_dist


IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
# if IS_HIGH_VERSION:
import torch.fft

def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()

    def get_num_loss_items(self):
        return 1

    def forward(self, inputs, labels, reduction='mean'):
        loss = F.mse_loss(inputs, labels, reduction=reduction)
        return  (loss, )


class Cross_Entropy(nn.Module):
    def __init__(self):
        super(Cross_Entropy, self).__init__()

    def get_num_loss_items(self):
        return 1
    
    def forward(self, logits, label, reduction='mean'):
        if isinstance(label, list):
            label = label[0]
        if isinstance(logits, tuple):
            logits = logits[0]
        ### unify the format as tuple.
        loss = F.cross_entropy(input=logits, target=label.long(), reduction=reduction)
        return (loss, )
    
class Cross_Entropy_L1(nn.Module):
    def __init__(self, alpha=0.5):
        super(Cross_Entropy_L1, self).__init__()
        self.alpha = alpha

    def get_num_loss_items(self):
        return 3

    def forward(self, inputs, labels, reduction='mean'):
        if isinstance(labels, list):
            label = labels[0]
            reg_label = labels[1]
        if isinstance(inputs, tuple):
            logits = inputs[0]
            reg_logits = inputs[1]

        loss1 = F.cross_entropy(input=logits, target=label.long(), reduction=reduction)
        loss2 = F.l1_loss(reg_logits, reg_label, reduction=reduction)
        loss = loss1 + self.alpha * loss2
        return loss, loss1, loss2
    
class Cross_Entropy_L1_MoCA_MMSE(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(Cross_Entropy_L1_MoCA_MMSE, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def get_num_loss_items(self):
        return 4

    def forward(self, inputs, labels, reduction='mean'):
        if isinstance(labels, list) or isinstance(labels, tuple):
            label = labels[0]
            MoCA_label = labels[1]
            MMSE_label = labels[2]
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            logits = inputs[0]
            MoCA_logits = inputs[1]
            MMSE_logits = inputs[2]

        loss1 = F.cross_entropy(input=logits, target=label.long(), reduction=reduction)
        loss2 = F.l1_loss(MoCA_logits, MoCA_label, reduction=reduction)
        loss3 = F.l1_loss(MMSE_logits, MMSE_label, reduction=reduction)
        loss = loss1 + self.alpha * loss2 + self.beta * loss3
        return loss, loss1, loss2, loss3

class Cross_Entropy_L1_Task_score(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(Cross_Entropy_L1_Task_score, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def get_num_loss_items(self):
        return 4
    
    def forward(self, inputs, labels, reduction='mean'):
        if isinstance(labels, list):
            label = labels[0]
            reg_label = labels[1]
            task_score_label = labels[4]
        if isinstance(inputs, tuple):
            logits = inputs[0]
            reg_logits = inputs[1]
            task_score_logits = inputs[2]

        loss1 = F.cross_entropy(input=logits, target=label.long(), reduction=reduction)
        loss2 = F.l1_loss(reg_logits, reg_label, reduction=reduction)
        loss3 = F.l1_loss(task_score_logits, task_score_label, reduction=reduction)
        loss = loss1 + self.alpha * loss2 + self.beta * loss3
        return loss, loss1, loss2, loss3

class Cross_Entropy_L1_Task_score_EyeTracking(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5):
        super(Cross_Entropy_L1_Task_score_EyeTracking, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_num_loss_items(self):
        return 5
    
    def forward(self, inputs, labels, reduction='mean'):
        if isinstance(labels, list):
            label = labels[0]
            reg_label = labels[1]
            task_score_label = labels[2]
            gaze_posi_label = labels[4]
        if isinstance(inputs, tuple):
            logits = inputs[0]
            reg_logits = inputs[1]
            task_score_logits = inputs[2]
            gaze_posi_logits = inputs[3]

        loss1 = F.cross_entropy(input=logits, target=label.long(), reduction=reduction)
        loss2 = F.l1_loss(reg_logits, reg_label, reduction=reduction)
        loss3 = F.l1_loss(task_score_logits, task_score_label, reduction=reduction)
        loss4 = F.l1_loss(gaze_posi_logits, gaze_posi_label, reduction=reduction)   
        if torch.sum(gaze_posi_label) == 0:
            loss4=0
        loss = loss1 + self.alpha * loss2 + self.beta * loss3 + self.gamma * loss4
        return loss, loss1, loss2, loss3, loss4

class Cross_Entropy_smooth_L1(nn.Module):
    def __init__(self, alpha=0.5):
        super(Cross_Entropy_smooth_L1, self).__init__()
        self.alpha = alpha

    def get_num_loss_items(self):
        return 3

    def forward(self, logits, labels, reduction='mean'):
        if isinstance(labels, list):
            label = labels[0]
            et_labels = labels[1]
        if isinstance(logits, tuple):
            logit = logits[0]
            et_logit = logits[2]

        loss1 = F.cross_entropy(input=logit, target=label.long(), reduction=reduction)
        loss2 = F.smooth_l1_loss(et_logit, et_labels, reduction=reduction)
        loss = loss1 + self.alpha * loss2
        return loss, loss1, loss2

class reg_L1_Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super(reg_L1_Loss, self).__init__()
        self.alpha = alpha

    def get_num_loss_items(self):
        return 3
    
    def forward(self, inputs, labels, reduction='mean'):
        moca_pred, mmse_pred = inputs
        moca_gt, mmse_gt = labels[0], labels[1]    
        loss1 = F.l1_loss(moca_pred.squeeze(-1), moca_gt.squeeze(-1), reduction=reduction)
        loss2 = F.l1_loss(mmse_pred.squeeze(-1), mmse_gt.squeeze(-1), reduction=reduction)
        loss = loss1 + self.alpha * loss2
        return loss, loss1, loss2
    
class reg_L1_Loss_Subscore_Similarity(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5, theta=0.5, delta=0.5):
        super(reg_L1_Loss_Subscore_Similarity, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.delta = delta

    def get_num_loss_items(self):
        return 7
    
    def forward(self, inputs, labels, reduction='mean'):
        moca_pred, mmse_pred, moca_subscore_pred, mmse_subscore_pred = inputs
        moca_gt, mmse_gt, moca_subscore_gt, mmse_subscore_gt = labels[0], labels[1], labels[2], labels[3].squeeze()
        loss1 = F.l1_loss(moca_pred, moca_gt, reduction=reduction)
        loss2 = F.l1_loss(mmse_pred, mmse_gt, reduction=reduction)
        # loss3 = F.cosine_similarity(torch.matmul(moca_subscore_pred, torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)),
        #                              torch.matmul(moca_subscore_gt, torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)), dim=-1).mean()
        # loss4 = F.cosine_similarity(torch.matmul(mmse_subscore_pred, torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)),
        #                              torch.matmul(mmse_subscore_gt, torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)), dim=-1).mean()
        if moca_subscore_gt.max() <= 1:
            moca_subscore_gt = moca_subscore_gt * torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)
        if mmse_subscore_gt.max() <= 1:
            mmse_subscore_gt = mmse_subscore_gt * torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)

        loss3 = 1 - F.cosine_similarity(moca_subscore_pred*(torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)),
                                     moca_subscore_gt).mean()
        loss4 = 1 - F.cosine_similarity(mmse_subscore_pred*(torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)),
                                     mmse_subscore_gt).mean()
        loss5 = F.l1_loss(torch.sum(moca_subscore_pred*(torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)), dim=-1)/30,
                           moca_gt, reduction=reduction)
        loss6 = F.l1_loss(torch.sum(mmse_subscore_pred*(torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)), dim=-1)/30,
                           mmse_gt, reduction=reduction)
        # loss5 = F.l1_loss(torch.sum(moca_subscore_pred*(torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)), dim=-1)/30,
        #                    moca_pred, reduction=reduction)
        # loss6 = F.l1_loss(torch.sum(mmse_subscore_pred*(torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)), dim=-1)/30,
        #                    mmse_pred, reduction=reduction)
        loss = loss1 + self.alpha * loss2 + self.beta * loss3 + self.gamma * loss4 + self.theta * loss5 + self.delta * loss6
        return loss, loss1, loss2, loss3, loss4, loss5, loss6

class reg_LDS_L1_Loss_Subscore_Similarity(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5, theta=0.5, delta=0.5):
        super(reg_LDS_L1_Loss_Subscore_Similarity, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.delta = delta

    def get_num_loss_items(self):
        return 7
    
    def forward(self, inputs, labels, reduction='mean'):
        moca_pred, mmse_pred, moca_subscore_pred, mmse_subscore_pred = inputs
        moca_gt, mmse_gt, moca_subscore_gt, mmse_subscore_gt = labels[0], labels[1], labels[2], labels[3].squeeze()

        label_shape = moca_gt.shape
        weights = calculate_eff_label_dist(moca_gt.view(-1)*30, ks=3, sigma=2)
        weights = torch.tensor(weights, device=moca_pred.device).view(label_shape)
        # loss1 = F.l1_loss(moca_pred, moca_gt, reduction=reduction)
        # loss2 = F.l1_loss(mmse_pred, mmse_gt, reduction=reduction)
        loss1 = weighted_l1_loss(moca_pred, moca_gt, weights=weights) * 100
        loss2 = weighted_l1_loss(mmse_pred, mmse_gt, weights=weights) * 100

        if moca_subscore_gt.max() <= 1:
            moca_subscore_gt = moca_subscore_gt * torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)
        if mmse_subscore_gt.max() <= 1:
            mmse_subscore_gt = mmse_subscore_gt * torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)

        loss3 = 1 - F.cosine_similarity(moca_subscore_pred*(torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)),
                                     moca_subscore_gt).mean()
        loss4 = 1 - F.cosine_similarity(mmse_subscore_pred*(torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)),
                                     mmse_subscore_gt).mean()
        loss5 = F.l1_loss(torch.sum(moca_subscore_pred*(torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)), dim=-1)/30,
                           moca_gt, reduction=reduction)
        loss6 = F.l1_loss(torch.sum(mmse_subscore_pred*(torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)), dim=-1)/30,
                           mmse_gt, reduction=reduction)
        # loss5 = F.l1_loss(torch.sum(moca_subscore_pred*(torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)), dim=-1)/30,
        #                    moca_pred, reduction=reduction)
        # loss6 = F.l1_loss(torch.sum(mmse_subscore_pred*(torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)), dim=-1)/30,
        #                    mmse_pred, reduction=reduction)
        loss = loss1 + self.alpha * loss2 + self.beta * loss3 + self.gamma * loss4 + self.theta * loss5 + self.delta * loss6
        return loss, loss1, loss2, loss3, loss4, loss5, loss6


class reg_L1_Loss_Similarity_CE(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5, theta=0.5):
        super(reg_L1_Loss_Similarity_CE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta

    def get_num_loss_items(self):
        return 6
    
    def forward(self, inputs, labels, reduction='mean'):
        moca_pred, mmse_pred = inputs
        moca_gt, mmse_gt = labels[0], labels[1]
        loss1 = F.l1_loss(moca_pred, moca_gt, reduction=reduction)
        loss2 = F.l1_loss(mmse_pred, mmse_gt, reduction=reduction)
        loss3 = F.cosine_similarity(moca_pred, moca_gt, dim=-1)
        loss4 = F.cosine_similarity(mmse_pred, mmse_gt, dim=-1)

        moca_pred_cls = torch.where(moca_pred >= 26/30, 0,
                                  torch.where((moca_pred >= 18/30) & (moca_pred < 26/30), 1, 2))
        moca_gt_cls = torch.where(moca_gt >= 26/30, 0,
                                  torch.where((moca_gt >= 18/30) & (moca_gt < 26/30), 1, 2))
        loss5 = F.cross_entropy(moca_pred_cls.to(torch.float), moca_gt_cls.to(torch.float), reduction=reduction)

        loss = loss1 + self.alpha * loss2 + self.beta * loss3 + self.gamma * loss4 + self.theta * loss5
        return loss, loss1, loss2, loss3, loss4, loss5

def shrinkage(l, a=10, c=0.04):
    f = 1 / (1 + torch.exp(a * (c-l)))
    return f*l*l

class reg_L2_Loss_shrinkage(nn.Module):
    def __init__(self, alpha=0.5):
        super(reg_L2_Loss_shrinkage, self).__init__()
        self.alpha = alpha

    def get_num_loss_items(self):
        return 3
    
    def forward(self, inputs, labels, reduction='mean'):
        moca_pred, mmse_pred = inputs
        moca_gt, mmse_gt = labels   
        loss1 = F.l1_loss(moca_pred, moca_gt, reduction=reduction)
        loss1 = shrinkage(loss1)
        loss2 = F.l1_loss(mmse_pred, mmse_gt, reduction=reduction)
        loss2 = shrinkage(loss2)
        loss = loss1 + self.alpha * loss2
        return loss, loss1, loss2

class ContrastiveLoss(nn.Module):
    def __init__(self, alpha, beta, gamma, theta, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.temperature = temperature
        self.supcon_loss = SupConLoss(temperature=temperature)
        self.rnc_loss = RnCLoss(temperature=temperature)

    def get_num_loss_items(self):
        return 5


    def forward(self, inputs, labels, reduction='mean'):
                
        # logits: [B, num_classes]
        # regressions: [B, 4096]
        # cls_feats: [B, 2, n_dim]
        # reg_feats: [B, 2, 4096]
        # task_ids: [B, 1]
        # task_embeds: [B, n_dim]
        # cog_scores: [B, 1]

        logits, regressions, cls_feats, reg_feats = inputs
        task_ids, task_embeds, cog_scores = labels

        loss1 = F.cross_entropy(input=logits, target=task_ids.long(), reduction=reduction)
        loss3 = self.supcon_loss(cls_feats, task_ids)


        loss2 = F.l1_loss(regressions[:, 0, :], task_embeds[:, 0, :], reduction=reduction)
        loss4 = self.rnc_loss(reg_feats, cog_scores)

        ### use embed_dis to replace reg_feats
        # reg_feats_dis = F.l1_loss(regressions, task_embeds.detach(), reduction='none') # [B, 2, 4096]
        # loss2 = reg_feats_dis[:, 0, :].mean()
        # loss4 = self.rnc_loss(reg_feats_dis, cog_scores)

        loss = self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3 + self.theta * loss4
        return loss, loss1, loss2, loss3, loss4
    

class CLIP_Loss(nn.Module):
    def __init__(self):
        super(CLIP_Loss, self).__init__()

    def get_num_loss_items(self):
        return 3
    
    def forward(self, inputs, labels=None, reduction='mean'):
        exg_aligned_feats, task_aligned_feats, logit_scale = inputs
        # task_ids = labels
        
        logits = task_aligned_feats @ exg_aligned_feats.T / logit_scale
        if labels is None:
            exg_similarity = exg_aligned_feats @ exg_aligned_feats.T / logit_scale
            task_similarity = task_aligned_feats @ task_aligned_feats.T / logit_scale
            targets = F.softmax((exg_similarity + task_similarity) / 2 * logit_scale, dim=-1)
        else:
            targets = labels[0]
        
        exg_loss = F.cross_entropy(logits, targets, reduction=reduction)
        task_loss = F.cross_entropy(logits.T, targets.T, reduction=reduction)

        loss = (exg_loss + task_loss) / 2
        return loss, exg_loss, task_loss
        
class CLIP_Score_Loss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5):
        super(CLIP_Score_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.clip_loss = CLIP_Loss()
       

    def get_num_loss_items(self):
        return 6
    
    def forward(self, inputs, labels, reduction='mean'):
        exg_cls_out, exg_aligned_feats, task_aligned_feats, logit_scale = inputs
        task_ids, cog_scores = labels

        # logit_scale = 1
        
        logits = task_aligned_feats @ exg_aligned_feats.T / logit_scale
        alignment_score = torch.diagonal(logits)
        
        loss1 = F.cross_entropy(exg_cls_out, task_ids, reduction=reduction)
        # loss2, exg_loss, task_loss = self.clip_loss(inputs=(exg_aligned_feats, task_aligned_feats, logit_scale), labels=(task_ids, cog_scores))
        loss2, exg_loss, task_loss = self.clip_loss(inputs=(exg_aligned_feats, task_aligned_feats, logit_scale), labels=None)
        
        loss3 = 1 - F.cosine_similarity(cog_scores-cog_scores.mean(), alignment_score-alignment_score.mean(), dim=-1)

        loss = self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3
        return loss, loss1, loss2, loss3, exg_loss, task_loss


class Cross_Entropy_L1_Contrastive_Loss(nn.Module):
    def __init__(self, alpha, beta, gamma, Reg_dim=2):
        super().__init__()
        self.criterion_CL = nn.CrossEntropyLoss()
        self.gamma = gamma
        self.Reg_dim = Reg_dim
        if Reg_dim == 2:
            self.semantic_loss = Cross_Entropy_L1_MoCA_MMSE(alpha, beta)
        elif Reg_dim == 1:
            self.semantic_loss = Cross_Entropy_L1(alpha)

    
    def get_num_loss_items(self):
        if self.Reg_dim == 2:
            return 6
        elif self.Reg_dim == 1:
            return 5
    
    def forward(self, X, Y, reduction='mean'):
        logits_s, labels_s = None, None
        if len(X) == 4:
            cates, reg_logits, logits_s, labels_s = X
        elif len(X) == 5:
            cates, MoCA_logits, MMSE_logits, logits_s, labels_s = X
        elif len(X) == 3:
            cates, MoCA_logits, MMSE_logits = X
        
        if len(Y) == 2:
            cates_label, reg_labels = Y
        elif len(Y) == 3:
            cates_label, MoCA_label, MMSE_label = Y
        
        if len(X) == 4 and len(Y) == 2:
            loss1, loss1_1, loss1_2 = self.semantic_loss(inputs=(cates, reg_logits), labels=(cates_label, reg_labels), reduction=reduction)
        elif (len(X) == 5 or len(X) == 3) and len(Y) == 3:
            loss1, loss1_1, loss1_2, loss1_3 = self.semantic_loss(inputs=(cates, MoCA_logits, MMSE_logits), labels=(cates_label, MoCA_label, MMSE_label), reduction=reduction)
           # contrastive loss
        if logits_s is not None and labels_s is not None:
            loss2 = self.criterion_CL(logits_s, labels_s)

        else:
            loss2 = torch.tensor(0.0, requires_grad=False)
  
        
        # loss1 = loss1_1 + alpha * loss1_2 + beta * loss1_3
        loss = loss1 + self.gamma * loss2

        if len(X) == 4 and len(Y) == 2:
            return loss, loss1, loss1_1, loss1_2, loss2
        elif (len(X) == 5 or len(X) == 3) and len(Y) == 3:
            return loss, loss1, loss1_1, loss1_2, loss1_3, loss2
        
class Cross_Entropy_Contrastive_Loss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.criterion_CL = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.semantic_loss = Cross_Entropy()

    
    def get_num_loss_items(self):
        return 3
    
    def forward(self, X, Y, reduction='mean'):
        cates, logits_s, labels_s = X
        cates_label = Y
        
        loss1 = self.semantic_loss(cates, cates_label)[0]
        loss2 = self.criterion_CL(logits_s, labels_s)
        
        loss = loss1 + self.alpha * loss2

        return loss, loss1, loss2
    

class Cross_Entropy_Contrastive_Loss_moco_v2(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.semantic_loss = Cross_Entropy()

    
    def get_num_loss_items(self):
        return 3
    
    def forward(self, X, Y, reduction='mean'):
        cates, supcon_loss = X
        cates_label = Y
        
        loss1 = self.semantic_loss(cates, cates_label, reduction=reduction)[0]
        
        loss = loss1 + self.alpha * supcon_loss

        return loss, loss1, supcon_loss



class reg_L1_RnC_Loss_Subscore_Similarity(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5, theta=0.5, delta=0.5, eta=0.5, temperature=0.07):
        super(reg_L1_RnC_Loss_Subscore_Similarity, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.delta = delta
        self.eta = eta
        self.rnc_loss = RnCLoss(temperature=temperature)

    def get_num_loss_items(self):
        return 7
    
    def forward(self, inputs, labels, features, reduction='mean'):
        moca_pred, mmse_pred, moca_subscore_pred, mmse_subscore_pred = inputs
        moca_gt, mmse_gt, moca_subscore_gt, mmse_subscore_gt = labels[0], labels[1], labels[2], labels[3].squeeze()
        loss1 = F.l1_loss(moca_pred, moca_gt, reduction=reduction)
        loss2 = F.l1_loss(mmse_pred, mmse_gt, reduction=reduction)
        # loss3 = F.cosine_similarity(torch.matmul(moca_subscore_pred, torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)),
        #                              torch.matmul(moca_subscore_gt, torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)), dim=-1).mean()
        # loss4 = F.cosine_similarity(torch.matmul(mmse_subscore_pred, torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)),
        #                              torch.matmul(mmse_subscore_gt, torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)), dim=-1).mean()
        if moca_subscore_gt.max() <= 1:
            moca_subscore_gt = moca_subscore_gt * torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)
        if mmse_subscore_gt.max() <= 1:
            mmse_subscore_gt = mmse_subscore_gt * torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)

        loss3 = 1 - F.cosine_similarity(moca_subscore_pred*(torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)),
                                     moca_subscore_gt).mean()
        loss4 = 1 - F.cosine_similarity(mmse_subscore_pred*(torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)),
                                     mmse_subscore_gt).mean()
        loss5 = F.l1_loss(torch.sum(moca_subscore_pred*(torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)), dim=-1)/30,
                           moca_gt, reduction=reduction)
        loss6 = F.l1_loss(torch.sum(mmse_subscore_pred*(torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)), dim=-1)/30,
                           mmse_gt, reduction=reduction)
        # loss5 = F.l1_loss(torch.sum(moca_subscore_pred*(torch.tensor(MOCA_TASK_SCORE_MAX).to(moca_subscore_pred.device).to(torch.float32)), dim=-1)/30,
        #                    moca_pred, reduction=reduction)
        # loss6 = F.l1_loss(torch.sum(mmse_subscore_pred*(torch.tensor(MMSE_TASK_SCORE_MAX).to(mmse_subscore_pred.device).to(torch.float32)), dim=-1)/30,
        #                    mmse_pred, reduction=reduction)
        # features original shape: torch.Size([4, 201, 10, 200])
        # Reshape features: merge the first two dimensions and the last two dimensions
        # Target shape: [4 * 201, 10 * 200] = [804, 2000]
        original_shape = features.shape
        if len(original_shape) == 4:
            features = features.reshape(original_shape[0] * original_shape[1], original_shape[2] * original_shape[3])
            moca_gt = moca_gt.reshape(original_shape[0] * original_shape[1])
            mmse_gt = mmse_gt.reshape(original_shape[0] * original_shape[1])
        loss7 = self.rnc_loss(features, moca_gt+mmse_gt) / features.shape[0]

        loss = loss1 + self.alpha * loss2 + self.beta * loss3 + self.gamma * loss4 + self.theta * loss5 + self.delta * loss6 + self.eta * loss7
        return loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7