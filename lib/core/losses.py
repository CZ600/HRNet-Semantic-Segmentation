import torch
import torch.nn as nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w),
                                  mode='bilinear', align_corners=True)

        log_pt = F.log_softmax(score, dim=1)
        pt = torch.exp(log_pt)
        log_pt = log_pt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)

        mask = (target != self.ignore_index).float()
        log_pt = log_pt * mask
        pt = pt * mask

        at = torch.ones_like(target, dtype=torch.float)
        at[target == 1] = self.alpha
        at[target == 0] = 1.0 - self.alpha
        at = at * mask

        loss = -at * torch.pow(1.0 - pt, self.gamma) * log_pt

        if self.reduction == 'mean':
            valid = mask.sum()
            if valid > 0:
                return loss.sum() / valid
            return loss.sum() * 0
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w),
                                  mode='bilinear', align_corners=True)

        prob = F.softmax(score, dim=1)
        prob_fg = prob[:, 1, :, :]

        mask = (target != self.ignore_index).float()
        target_fg = (target == 1).float()

        prob_fg = prob_fg * mask
        target_fg = target_fg * mask

        intersection = (prob_fg * target_fg).sum()
        union = prob_fg.sum() + target_fg.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=1.0, dice_weight=1.0, alpha=0.25, gamma=2.0,
                 ignore_index=255):
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, score, target):
        if isinstance(score, (list, tuple)):
            loss = sum(
                self.focal_weight * self.focal(s, target) +
                self.dice_weight * self.dice(s, target)
                for s in score
            ) / len(score)
        else:
            loss = self.focal_weight * self.focal(score, target) + \
                   self.dice_weight * self.dice(score, target)
        return loss
