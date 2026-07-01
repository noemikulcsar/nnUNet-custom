import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CAMUS_CBAM import (
    ChannelAttention,
    SpatialAttention,
    nnUNetTrainer_CAMUS_CBAM,
)

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)


class ProgressiveResidualCBAM(nn.Module):
    """
    Residual CBAM block.

    Output:
        y = x + alpha * CBAM(x)

    alpha este un parametru invatabil (LayerScale) care controleaza
    contributia mecanismului de atentie.
    """

    def __init__(self,
                 channels: int,
                 reduction: int = 16):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels,
            reduction
        )

        self.spatial_attention = SpatialAttention()

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):

        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        scale = torch.sigmoid(self.alpha)

        return x + scale * (out - x)


class ProgressiveEncoderWithCBAM(nn.Module):
    """
    Encoder care aplica CBAM doar pe nivelurile specificate.

    Implicit:
        Stage 0 -> fara CBAM
        Stage 1 -> fara CBAM
        Stage 2 -> CBAM
        Stage 3 -> CBAM
        Stage 4 -> CBAM
    """

    def __init__(
        self,
        encoder: nn.Module,
        features_per_stage,
        cbam_stages=(2, 3, 4),
    ):
        super().__init__()

        self.encoder = encoder

        self.cbam_stages = set(cbam_stages)

        self.attention_blocks = nn.ModuleDict()

        for stage in self.cbam_stages:
            self.attention_blocks[str(stage)] = ProgressiveResidualCBAM(
                features_per_stage[stage]
            )

    def forward(self, x):

        skips = self.encoder(x)

        outputs = []

        for stage, skip in enumerate(skips):

            if stage in self.cbam_stages:
                outputs.append(
                    self.attention_blocks[str(stage)](skip)
                )
            else:
                outputs.append(skip)

        return outputs

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.encoder, name)
        
class nnUNetTrainer_CAMUS_CBAM_v2(nnUNetTrainer_CAMUS_CBAM):
    """
    Improved CBAM trainer.

    Improvements:
    - Progressive CBAM (only on deeper encoder stages)
    - Residual CBAM with learnable LayerScale
    """
    CBAM_STAGES = (2,3,4)

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )

        features_per_stage = configuration_manager.network_arch_init_kwargs[
            "features_per_stage"
        ]

        network.encoder = ProgressiveEncoderWithCBAM(
            encoder=network.encoder,
            features_per_stage=features_per_stage,
            cbam_stages=nnUNetTrainer_CAMUS_CBAM_v2.CBAM_STAGES, 
        )

        return network