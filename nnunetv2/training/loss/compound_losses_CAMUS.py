import torch
from torch import nn

from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.camus_losses_v2 import MemoryEfficientSoftTverskyLoss


class DC_and_CE_and_Tversky_loss(nn.Module):
    """
    Dice + CrossEntropy + Tversky

    Final loss:

        weight_dice * Dice
      + weight_ce * CrossEntropy
      + weight_tversky * Tversky
    """

    def __init__(
        self,
        soft_dice_kwargs,
        ce_kwargs,
        weight_ce=0.4,
        weight_dice=0.4,
        weight_tversky=0.2,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss,
        tversky_class=MemoryEfficientSoftTverskyLoss,
        tversky_kwargs=None,
    ):
        super().__init__()

        if ignore_label is not None:
            ce_kwargs["ignore_index"] = ignore_label

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_tversky = weight_tversky
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.dc = dice_class(
            apply_nonlin=softmax_helper_dim1,
            **soft_dice_kwargs,
        )

        if tversky_kwargs is None:
            tversky_kwargs = dict(
                alpha=0.7,
                beta=0.3,
            )

        self.tv = tversky_class(
            apply_nonlin=softmax_helper_dim1,
            **soft_dice_kwargs,
            **tversky_kwargs,
        )

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
    ):

        if self.ignore_label is not None:

            assert target.shape[1] == 1

            mask = target != self.ignore_label

            target_dice = torch.where(
                mask,
                target,
                torch.zeros_like(target),
            )

            num_fg = mask.sum()

        else:

            target_dice = target
            mask = None

        dice_loss = (
            self.dc(
                net_output,
                target_dice,
                loss_mask=mask,
            )
            if self.weight_dice != 0
            else 0
        )

        tversky_loss = (
            self.tv(
                net_output,
                target_dice,
                loss_mask=mask,
            )
            if self.weight_tversky != 0
            else 0
        )

        target_ce = target[:, 0].long()

        ce_loss = (
            self.ce(
                net_output,
                target_ce,
            )
            if (
                self.weight_ce != 0
                and (
                    self.ignore_label is None
                    or num_fg > 0
                )
            )
            else 0
        )

        return (
            self.weight_dice * dice_loss
            + self.weight_ce * ce_loss
            + self.weight_tversky * tversky_loss
        )