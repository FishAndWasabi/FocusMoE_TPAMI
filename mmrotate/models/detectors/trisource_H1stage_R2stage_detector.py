# Copyright (c) OpenMMLab. All rights reserved.
import inspect

from mmdet.core import bbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector


@ROTATED_DETECTORS.register_module()
class TriSourceDetector(RotatedBaseDetector):
    """Inference wrapper for the SAR, RGB and IR detector heads."""

    def __init__(self, backbone, neck=None, rgb_rpn_head=None, rgb_roi_head=None,
                 rgb_test_cfg=None, ifr_rpn_head=None, ifr_roi_head=None,
                 ifr_test_cfg=None, sar_bbox_head=None, sar_test_cfg=None,
                 train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        super(TriSourceDetector, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self._backbone_dataset_arg = self._infer_backbone_dataset_arg()
        if neck is not None:
            self.neck = build_neck(neck)
        # Keep the original head names and order for checkpoint compatibility.
        for name, rpn, roi, cfg in (
                ('rgb', rgb_rpn_head, rgb_roi_head, rgb_test_cfg),
                ('ifr', ifr_rpn_head, ifr_roi_head, ifr_test_cfg)):
            if rpn is not None:
                options = rpn.copy()
                options.update(train_cfg=None, test_cfg=cfg.rpn)
                setattr(self, name + '_rpn_head', build_head(options))
            if roi is not None:
                options = roi.copy()
                options.update(train_cfg=None, test_cfg=cfg.rcnn, pretrained=None)
                setattr(self, name + '_roi_head', build_head(options))
            setattr(self, name + '_test_cfg', cfg)
        options = sar_bbox_head.copy()
        options.update(train_cfg=None, test_cfg=sar_test_cfg)
        self.sar_bbox_head = build_head(options)
        self.sar_test_cfg = sar_test_cfg


    def _infer_backbone_dataset_arg(self):
        """Infer backbone forward argument name for modality tags."""
        forward = getattr(self.backbone, 'forward', None)
        if forward is None:
            return None
        try:
            sig = inspect.signature(forward)
        except (TypeError, ValueError):
            return None
        if 'dataset_names' in sig.parameters:
            return 'dataset_names'
        if 'datasets' in sig.parameters:
            return 'datasets'
        return None

    def _forward_backbone(self, batch_inputs, datasets):
        """Call backbone forward with compatible dataset argument name."""
        if self._backbone_dataset_arg == 'dataset_names':
            return self.backbone(batch_inputs, dataset_names=datasets)
        if self._backbone_dataset_arg == 'datasets':
            return self.backbone(batch_inputs, datasets=datasets)

        # Fallback for wrappers or unusual signatures.
        try:
            return self.backbone(batch_inputs, dataset_names=datasets)
        except TypeError as e:
            msg = str(e)
            if 'unexpected keyword argument' not in msg or 'dataset_names' not in msg:
                raise
        try:
            return self.backbone(batch_inputs, datasets=datasets)
        except TypeError as e:
            msg = str(e)
            if 'unexpected keyword argument' not in msg or 'datasets' not in msg:
                raise
        return self.backbone(batch_inputs)


    @property
    def with_rgb_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rgb_rpn_head') and self.rgb_rpn_head is not None


    @property
    def with_rgb_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'rgb_roi_head') and self.rgb_roi_head is not None

    @property
    def with_ifr_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'ifr_rpn_head') and self.ifr_rpn_head is not None


    @property
    def with_ifr_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'ifr_roi_head') and self.ifr_roi_head is not None

    def extract_feat(self, batch_inputs, datasets):
        """Extract one modality's features with the original neck settings."""
        if len(datasets) != 1:
            raise ValueError('Evaluate one modality per batch.')
        x = self._forward_backbone(batch_inputs, datasets)
        experts_id = None
        if isinstance(x, tuple) and len(x) == 2:
            x, _ = x
        elif isinstance(x, tuple) and len(x) == 3:
            x, _, experts_id = x
        if self.with_neck:
            if datasets[0] == 'sar':
                x = self.neck(x, start_level=1, add_extra_convs='on_output')
            elif datasets[0] in ('rgb', 'ifr'):
                x = self.neck(x)
            else:
                raise ValueError('Unknown modality: ' + str(datasets[0]))
        return x, experts_id


    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        # Handle input shape: get_model_complexity_info may pass (C, H, W) instead of (B, C, H, W)
        if img.dim() == 3:
            img = img.unsqueeze(0)  # Add batch dimension

        # For FLOPs calculation, we only need to compute one path
        # Use SAR path as it's the main detection path
        x, _ = self.extract_feat(img, ['sar'])
        results = self.sar_bbox_head.forward(x)
        return (results,)

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError('This release supports inference only.')

    def simple_test(self, img, img_metas, subdataset, proposals=None, rescale=False):
        """Test without augmentation."""
        assert isinstance(subdataset[0],list) and len(subdataset)==1 # subdataset: [['sar']]
        assert all(x == subdataset[0][0] for x in subdataset[0]), "Not all elements in subdataset are the same: " + str(subdataset)
        subdataset = subdataset[0][0]

        x = self.extract_feat(img, [subdataset])

        if isinstance(x,tuple):
            x,experts_id=x
        else:
            experts_id=None
        if subdataset == 'sar':

            results_list = self.sar_bbox_head.simple_test(
                x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.sar_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results
        elif subdataset == 'rgb':
            if proposals is None:
                 proposal_list = self.rgb_rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            return self.rgb_roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale)


        elif subdataset == 'ifr':
            if proposals is None:
                proposal_list = self.ifr_rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            return self.ifr_roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas,subdataset, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        assert isinstance(subdataset[0],list) and len(subdataset)==1
        assert all(x == subdataset[0][0] for x in subdataset[0]), "Not all elements in subdataset are the same: " + str(subdataset)
        subdataset = subdataset[0][0]
        x = self.extract_feat(imgs, [subdataset])
        if subdataset == 'sar':
            results_list = self.sar_bbox_head.aug_test(
            x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.sar_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results
        elif subdataset == 'rgb':
            proposal_list = self.rgb_rpn_head.aug_test_rpn(x, img_metas)
            return self.rgb_roi_head.aug_test(
                x, proposal_list, img_metas, rescale=rescale)

        elif subdataset == 'ifr':
            proposal_list = self.ifr_rpn_head.aug_test_rpn(x, img_metas)
            return self.ifr_roi_head.aug_test(
                x, proposal_list, img_metas, rescale=rescale)
