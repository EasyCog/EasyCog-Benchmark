import torch
import torch.nn as nn
import torcheval.metrics.functional as MetricsFunc
__all__ = ['AverageMeter', 'evaluate']


class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"

class CosineSimilarity(nn.Module):
    def forward(self, tensor1, tensor2):
        norm_tensor_1 = tensor1 / tensor1.norm(dim=-1, keep_dim=True)
        norm_tensor_2 = tensor2 / tensor2.norm(dim=-1, keep_dim=True)
        return (norm_tensor_1*norm_tensor_2).sum(dim=-1)


def evaluate(preds, gt):
    with torch.no_grad():
        diff = preds - gt
        mae_ratio = (diff.abs() / (gt.abs()+1e-4)).mean()
        mae = diff.abs().mean()
        mean_error = diff.mean()
        var_error = torch.var(diff, unbiased=False)
        rho = torch.cosine_similarity(preds-preds.mean(dim=-1, keepdim=True), gt-gt.mean(dim=-1, keepdim=True), dim=-1).mean()
        return mae, mae_ratio, mean_error, var_error, rho
    
def evaluate_c(preds, gt, average='macro', num_classes=5):
    """
    Comprehensive evaluation for multiple classification tasks.
    
    Args:
        preds: [B, Num_classes] logits or predictions
        gt: [B, 1] ground truth labels
        average: Averaging method ('macro', 'micro', 'weighted')
        num_classes: Number of classes
    
    Returns:
        tuple: (accuracy, f1_score, recall, precision)
    """
    with torch.no_grad():
        preds = preds.clone().detach()
        gt = gt.long().clone().detach().view(-1)
        
        # For logits input, convert to class predictions
        if preds.dim() > 1 and preds.size(1) > 1:  # if logits
            preds = preds.argmax(dim=1)
        preds = preds.view(-1)
        
        if num_classes == 2:
            accuracy = MetricsFunc.binary_accuracy(preds, gt)
            f1_score = MetricsFunc.binary_f1_score(preds, gt)
            recall = MetricsFunc.binary_recall(preds, gt)
            precision = MetricsFunc.binary_precision(preds, gt)
        else:
            accuracy = MetricsFunc.multiclass_accuracy(preds, gt, average=average, num_classes=num_classes)
            f1_score = MetricsFunc.multiclass_f1_score(preds, gt, average=average, num_classes=num_classes)
            recall = MetricsFunc.multiclass_recall(preds, gt, average=average, num_classes=num_classes)
            # recall = 0
            precision = MetricsFunc.multiclass_precision(preds, gt, average=average, num_classes=num_classes)
        
        # Convert to numpy for consistency
        metrics = [accuracy, f1_score, recall, precision]
        metrics = [m.cpu().numpy() if torch.is_tensor(m) else m for m in metrics]
        
    return tuple(metrics)


def cal_var(preds_list, gt_list):
    with torch.no_grad():
        diff = preds_list - gt_list
        var = torch.var(diff, unbiased=False)
    return var