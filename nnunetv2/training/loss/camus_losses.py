import torch
from torch import nn
import torch.nn.functional as F

import torch.nn.functional as F

class SoftTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5, do_bg=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.do_bg = do_bg

    def forward(self, net_output, target):
        probs = F.softmax(net_output, dim=1)

        if target.ndim != probs.ndim:
            target = target.view((target.shape[0], 1, *target.shape[1:]))

        target_onehot = torch.zeros_like(probs)
        target_onehot.scatter_(1, target.long(), 1)

        if not self.do_bg:
            probs = probs[:, 1:]
            target_onehot = target_onehot[:, 1:]

        dims = tuple(range(2, probs.ndim))

        tp = (probs * target_onehot).sum(dims)
        fp = (probs * (1 - target_onehot)).sum(dims)
        fn = ((1 - probs) * target_onehot).sum(dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )

        return 1 - tversky.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=None):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, net_output, target):
        target = target[:, 0].long()

        ce = F.cross_entropy(
            net_output,
            target,
            reduction="none",
            ignore_index=self.ignore_index if self.ignore_index is not None else -100,
        )

        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            focal = focal[mask]

        return focal.mean()


class TverskyFocalLoss(nn.Module):
    def __init__(
        self,
        alpha=0.7,
        beta=0.3,
        gamma=2.0,
        weight_tversky=0.7,
        weight_focal=0.3,
        ignore_label=None,
    ):
        super().__init__()
        self.tversky = SoftTverskyLoss(alpha=alpha, beta=beta, do_bg=False)
        self.focal = FocalLoss(gamma=gamma, ignore_index=ignore_label)

        self.weight_tversky = weight_tversky
        self.weight_focal = weight_focal

    def forward(self, net_output, target):
        return (
            self.weight_tversky * self.tversky(net_output, target)
            + self.weight_focal * self.focal(net_output, target)
        )