import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_CAMUS_v1 import nnUNetTrainer_CAMUS_v1
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        reduced_channels = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))

        return x * attention


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class EncoderWithCBAM(nn.Module):
    def __init__(self, encoder: nn.Module, features_per_stage):
        super().__init__()

        self.encoder = encoder

        self.attention_blocks = nn.ModuleList([
            CBAMBlock(channels)
            for channels in features_per_stage
        ])

    def forward(self, x):
        skips = self.encoder(x)

        skips = [
            attention(skip)
            for attention, skip in zip(self.attention_blocks, skips)
        ]

        return skips

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.encoder, name)


class nnUNetTrainer_CAMUS_CBAM(nnUNetTrainer_CAMUS_v1):

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

        network.encoder = EncoderWithCBAM(
            network.encoder,
            features_per_stage,
        )

        return network