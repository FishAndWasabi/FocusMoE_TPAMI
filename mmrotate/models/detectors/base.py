# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import BaseDetector
from ..builder import ROTATED_DETECTORS


@ROTATED_DETECTORS.register_module()
class RotatedBaseDetector(BaseDetector):
    """Base detector for FP32 single-image inference."""

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.fp16_enabled = False
