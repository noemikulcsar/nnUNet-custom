import torch
from torch import nn
from typing import Callable

from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class MemoryEfficientSoftTverskyLoss(nn.Module):
    """
    Memory-efficient implementation of Soft Tversky Loss.

    Supports:
    - batch Dice
    - DDP
    - ignore mask
    - optional background removal
    """

    def __init__(
        self,
        apply_nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = True,
        alpha: float = 0.4,
        beta: float = 0.6,
    ):
        super().__init__()

        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp

        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():

            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y.float()
            else:
                y_onehot = torch.zeros(
                    x.shape,
                    device=x.device,
                    dtype=torch.float32,
                )

                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        if not self.do_bg:
            x = x[:, 1:].contiguous()

        if loss_mask is None:

            tp = (x * y_onehot).sum(
                axes,
                dtype=torch.float32,
            )

            fp = (x * (1 - y_onehot)).sum(
                axes,
                dtype=torch.float32,
            )

            fn = ((1 - x) * y_onehot).sum(
                axes,
                dtype=torch.float32,
            )

        else:

            tp = (
                x * y_onehot * loss_mask
            ).sum(
                axes,
                dtype=torch.float32,
            )

            fp = (
                x * (1 - y_onehot) * loss_mask
            ).sum(
                axes,
                dtype=torch.float32,
            )

            fn = (
                (1 - x) * y_onehot * loss_mask
            ).sum(
                axes,
                dtype=torch.float32,
            )

        if self.batch_dice:

            if self.ddp:

                tp = AllGatherGrad.apply(tp).sum(
                    0,
                    dtype=torch.float32,
                )

                fp = AllGatherGrad.apply(fp).sum(
                    0,
                    dtype=torch.float32,
                )

                fn = AllGatherGrad.apply(fn).sum(
                    0,
                    dtype=torch.float32,
                )

            tp = tp.sum(0, dtype=torch.float32)
            fp = fp.sum(0, dtype=torch.float32)
            fn = fn.sum(0, dtype=torch.float32)

        tversky = (
            tp + self.smooth
        ) / (
            tp
            + self.alpha * fn
            + self.beta * fp
            + self.smooth
        ).clamp_min(1e-8)

        return -tversky.mean()