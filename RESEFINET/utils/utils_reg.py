"""
训练脚本依赖：FocalLoss、da_loss，以及 torchvision 等导出供 norm_resnet50_SAR / efficient_SAR 使用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms, models

# 供脚本中 "from utils.utils_reg import *" 使用（norm_resnet50_SAR 用 torchvision.datasets.ImageFolder）
__all__ = ['FocalLoss', 'da_loss', 'transforms', 'models', 'torchvision', 'F', 'np', 'torch']


class FocalLoss(nn.Module):
    """多分类 Focal Loss，alpha_t 为各类别权重，gamma 为聚焦参数。"""
    def __init__(self, alpha_t, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha_t  # [num_classes]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,) long
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        alpha_t = self.alpha.to(logits.device)
        if alpha_t.dim() == 1:
            alpha_t = alpha_t[targets]
        focal = alpha_t * (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return focal.mean()
        if self.reduction == 'sum':
            return focal.sum()
        return focal


class da_loss:
    """域对齐损失：对两路特征做 MMD，用于 EO/SAR 特征对齐。"""
    def __call__(self, feat_a, feat_b, sigma=1.0):
        # feat_a, feat_b: (B, C, H, W) -> 展平为 (B, C*H*W) 或 (B, C) 后算 MMD
        if feat_a.dim() == 4:
            feat_a = feat_a.view(feat_a.size(0), -1)
        if feat_b.dim() == 4:
            feat_b = feat_b.view(feat_b.size(0), -1)
        xx = torch.cdist(feat_a, feat_a, p=2)
        yy = torch.cdist(feat_b, feat_b, p=2)
        xy = torch.cdist(feat_a, feat_b, p=2)
        loss = (torch.mean(torch.exp(-xx / (2 * sigma ** 2))) +
                torch.mean(torch.exp(-yy / (2 * sigma ** 2))) -
                2 * torch.mean(torch.exp(-xy / (2 * sigma ** 2))))
        return loss
