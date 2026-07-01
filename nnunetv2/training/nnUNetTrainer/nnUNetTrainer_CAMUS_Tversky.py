from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CAMUS_CBAM_v2 import (
    nnUNetTrainer_CAMUS_CBAM_v2,
)

from nnunetv2.training.loss.compound_losses_CAMUS import (
    DC_and_CE_and_Tversky_loss,
)

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
import torch
import numpy as np

class nnUNetTrainer_CAMUS_Tversky(nnUNetTrainer_CAMUS_CBAM_v2):

    def _build_loss(self):

        if self.label_manager.has_regions:

            raise NotImplementedError(
                "CAMUS trainer currently supports only label-based segmentation."
            )

        loss = DC_and_CE_and_Tversky_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            ce_kwargs={},
            weight_ce=0.4,
            weight_dice=0.4,
            weight_tversky=0.2,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
            tversky_kwargs={
                "alpha": 0.7,
                "beta": 0.3,
            },
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)
            loss.tv = torch.compile(loss.tv)

        if self.enable_deep_supervision:

            deep_supervision_scales = self._get_deep_supervision_scales()

            weights = np.array(
                [1 / (2 ** i) for i in range(len(deep_supervision_scales))]
            )

            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()

            from nnunetv2.training.loss.deep_supervision import (
                DeepSupervisionWrapper,
            )

            loss = DeepSupervisionWrapper(loss, weights)

        return loss