# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

import torch
from mmcv.runner import BaseModule
from mmdet.core import bbox2roi

from mmrotate.core import (aug_multiclass_nms_rotated, build_assigner,
                           build_sampler, obb2xyxy, rbbox2result)
from ..builder import (ROTATED_HEADS, build_head, build_roi_extractor,
                       build_shared_head)


@ROTATED_HEADS.register_module()
class RotatedStandardRoIHead(BaseModule, metaclass=ABCMeta):
    """Simplest base rotated roi head including one bbox head.

    Args:
        bbox_roi_extractor (dict, optional): Config of ``bbox_roi_extractor``.
        bbox_head (dict, optional): Config of ``bbox_head``.
        shared_head (dict, optional): Config of ``shared_head``.
        train_cfg (dict, optional): Config of train.
        test_cfg (dict, optional): Config of test.
        pretrained (str, optional): Path of pretrained weight.
        init_cfg (dict, optional): Config of initialization.
        version (str, optional): Angle representations. Defaults to 'oc'.
    """

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 version='oc'):

        super(RotatedStandardRoIHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.version = version

        if shared_head is not None:
            shared_head.pretrained = pretrained
            self.shared_head = build_shared_head(shared_head)

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        self.init_assigner_sampler()

        self.with_bbox = True if bbox_head is not None else False
        self.with_shared_head = True if shared_head is not None else False

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``.

        Args:
            bbox_roi_extractor (dict): Config of ``bbox_roi_extractor``.
            bbox_head (dict): Config of ``bbox_head``.
        """
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs


    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing.

        Args:
            x (list[Tensor]): list of multi-level img features.
            rois (list[Tensors]): list of region of interests.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results


    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                rescale=False):
        """Async test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_list (list[Tensors]): list of region proposals.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = rbbox2result(det_bboxes, det_labels,
                                    self.bbox_head.num_classes)
        return bbox_results

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation.

        Args:
            x (list[Tensor]): list of multi-level img features.
            proposal_list (list[Tensors]): list of region proposals.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                         self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.
        
        Args:
            x (list[list[Tensor]]): Features from multiple augmentations.
                The outer list indicates augmentations, and the inner list
                indicates images in a batch.
            proposal_list (list[list[Tensor]]): Region proposals from multiple
                augmentations. The outer list indicates augmentations, and the
                inner list indicates images in a batch.
            img_metas (list[list[dict]]): Meta information from multiple
                augmentations. The outer list indicates augmentations, and the
                inner list indicates images in a batch.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
                
        Returns:
            list[ndarray]: Detection results of each class for each image.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        
        # Assuming batch size is 1 for test time augmentation
        # Collect detection results from all augmentations
        aug_bboxes = []
        aug_labels = []
        
        for x_single, proposals_single, img_meta_single in zip(
                x, proposal_list, img_metas):
            # Get detection results for single augmentation
            # Note: simple_test_bboxes returns lists of tensors
            det_bboxes, det_labels = self.simple_test_bboxes(
                x_single, img_meta_single, proposals_single, 
                self.test_cfg, rescale=False)
            
            # For each image in the batch (typically just 1)
            for bbox, label in zip(det_bboxes, det_labels):
                aug_bboxes.append(bbox)  # Shape: [N, 6] (cx, cy, w, h, a, score)
                aug_labels.append(label)  # Shape: [N]
        
        # Merge results from all augmentations
        # Process each image separately (typically just one image)
        num_imgs = len(img_metas[0])
        bbox_results = []
        
        for img_id in range(num_imgs):
            # Collect all augmented results for this image
            img_aug_bboxes = []
            img_aug_labels = []
            
            for aug_id in range(len(x)):
                idx = aug_id * num_imgs + img_id
                if idx < len(aug_bboxes) and aug_bboxes[idx].numel() > 0:
                    img_aug_bboxes.append(aug_bboxes[idx])
                    img_aug_labels.append(aug_labels[idx])
            
            # Merge all augmented detections
            if len(img_aug_bboxes) > 0:
                # Concatenate all bboxes and labels
                merged_bboxes = torch.cat(img_aug_bboxes, dim=0)
                merged_labels = torch.cat(img_aug_labels, dim=0)
                
                # Apply NMS across all augmentations
                det_bboxes, det_labels = aug_multiclass_nms_rotated(
                    merged_bboxes, merged_labels,
                    self.test_cfg.score_thr, self.test_cfg.nms,
                    self.test_cfg.max_per_img, self.bbox_head.num_classes)
                
                if rescale and det_bboxes.shape[0] > 0:
                    # Rescale to original image size
                    # Note: angle (index 4) and score (index 5) should not be rescaled
                    # For rotated boxes: (cx, cy, w, h, angle, score)
                    scale_factor = det_bboxes.new_tensor(
                        img_metas[0][img_id]['scale_factor'])
                    det_bboxes[:, :4] = det_bboxes[:, :4] / scale_factor
            else:
                # No detections for this image
                # Use the device from img_metas or proposals instead of aug_bboxes
                device = x[0][0].device if len(x) > 0 and len(x[0]) > 0 else torch.device('cpu')
                det_bboxes = torch.zeros((0, 6), device=device)
                det_labels = torch.zeros((0,), dtype=torch.long, device=device)
            
            # Convert to result format
            bbox_result = rbbox2result(
                det_bboxes, det_labels, self.bbox_head.num_classes)
            bbox_results.append(bbox_result)
        
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains \
                the boxes of the corresponding image in a batch, each \
                tensor has the shape (num_boxes, 5) and last dimension \
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor \
                in the second list is the labels with shape (num_boxes, ). \
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
