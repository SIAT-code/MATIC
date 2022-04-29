import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.
    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

class BinaryFocalLoss(nn.Module):

    def __init__(self, args):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.smooth = 0

    def forward(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        return loss