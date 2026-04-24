from typing import List, Tuple, Union

import numpy as np
import torch

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTransform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_CAMUS_v1(nnUNetTrainer):
    """
    Trainer custom pentru CAMUS.
    Modificări conservative:
    - learning rate mai stabil
    - foreground oversampling mai mare
    - rotații mai moderate pentru ecografie cardiacă
    - augmentări intensity adaptate ultrasound
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.initial_lr = 3e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.50
        self.num_epochs = 50

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )

        # CAMUS este ecografie cardiacă 2D. Rotațiile de 180° pot fi prea agresive.
        if len(self.configuration_manager.patch_size) == 2:
            rotation_for_DA = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
            mirror_axes = (0, 1)

        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:

        transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.25,
                rotation=rotation_for_DA,
                p_scaling=0.25,
                scaling=(0.80, 1.25),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                border_mode_seg="constant",
                padding_value_seg=-1,
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.06),
                    p_per_channel=1,
                    synchronize_channels=True,
                ),
                apply_probability=0.20,
            )
        )

        transforms.append(
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.4, 0.9),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5,
                    benchmark=True,
                ),
                apply_probability=0.15,
            )
        )

        transforms.append(
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.70, 1.30)),
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.25,
            )
        )

        transforms.append(
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.65, 1.35)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.25,
            )
        )

        transforms.append(
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.65, 1.0),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5,
                ),
                apply_probability=0.15,
            )
        )

        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.75, 1.35)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.35,
            )
        )

        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(
                MaskImageTransform(
                    apply_to_channels=[
                        i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]
                    ],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        transforms.append(RemoveLabelTransform(-1, 0))

        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label]
                    if ignore_label is not None
                    else regions,
                    channel_in_seg=0,
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)