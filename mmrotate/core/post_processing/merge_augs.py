# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings

import torch
from mmcv import ConfigDict
from mmcv.ops import nms_rotated

from mmrotate.core.bbox import bbox_mapping_back


def merge_aug_proposals(aug_proposals, img_metas, cfg):
    """Merge augmented proposals for rotated boxes (multiscale, flip, etc.)

    Args:
        aug_proposals (list[Tensor]): proposals from different testing
            schemes, shape (n, 6) for rotated boxes (cx, cy, w, h, a, score).
            Note that they are not rescaled to the original image size.

        img_metas (list[dict]): list of image info dict where each dict has:
            'img_shape', 'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `mmdet/datasets/pipelines/formatting.py:Collect`.

        cfg (dict): rpn test config.

    Returns:
        Tensor: shape (n, 6), proposals corresponding to original image scale.
            Format: (cx, cy, w, h, angle, score)
    """

    cfg = copy.deepcopy(cfg)

    # deprecate arguments warning
    if 'nms' not in cfg or 'max_num' in cfg or 'nms_thr' in cfg:
        warnings.warn(
            'In rpn_proposal or test_cfg, '
            'nms_thr has been moved to a dict named nms as '
            'iou_threshold, max_num has been renamed as max_per_img, '
            'name of original arguments and the way to specify '
            'iou_threshold of NMS will be deprecated.')
    if 'nms' not in cfg:
        cfg.nms = ConfigDict(dict(type='nms', iou_threshold=cfg.nms_thr))
    if 'max_num' in cfg:
        if 'max_per_img' in cfg:
            assert cfg.max_num == cfg.max_per_img, f'You set max_num and ' \
                f'max_per_img at the same time, but get {cfg.max_num} ' \
                f'and {cfg.max_per_img} respectively' \
                f'Please delete max_num which will be deprecated.'
        else:
            cfg.max_per_img = cfg.max_num
    if 'nms_thr' in cfg:
        assert cfg.nms.iou_threshold == cfg.nms_thr, f'You set ' \
            f'iou_threshold in nms and ' \
            f'nms_thr at the same time, but get ' \
            f'{cfg.nms.iou_threshold} and {cfg.nms_thr}' \
            f' respectively. Please delete the nms_thr ' \
            f'which will be deprecated.'

    recovered_proposals = []
    for proposals, img_info in zip(aug_proposals, img_metas):
        img_shape = img_info['img_shape']
        scale_factor = img_info['scale_factor']
        flip = img_info['flip']
        flip_direction = img_info['flip_direction']
        _proposals = proposals.clone()
        
        # For rotated boxes: (cx, cy, w, h, angle, score)
        # bbox_mapping_back handles both horizontal (4-dim) and rotated (5-dim) boxes
        _proposals = bbox_mapping_back(
            _proposals, img_shape, scale_factor, flip, flip_direction)
        
        recovered_proposals.append(_proposals)
    
    aug_proposals = torch.cat(recovered_proposals, dim=0)
    
    # Apply NMS for rotated boxes
    if aug_proposals.size(-1) >= 6:  # Has score
        # Use rotated NMS
        merged_proposals, _ = nms_rotated(
            aug_proposals[:, :5].contiguous(),  # (cx, cy, w, h, angle)
            aug_proposals[:, -1].contiguous(),  # scores
            cfg.nms.iou_threshold)
    else:
        # No score, just return all proposals
        merged_proposals = aug_proposals
    
    # Sort by score and keep top-k
    if merged_proposals.size(-1) >= 6:
        scores = merged_proposals[:, -1]
        _, order = scores.sort(0, descending=True)
        num = min(cfg.max_per_img, merged_proposals.shape[0])
        order = order[:num]
        merged_proposals = merged_proposals[order, :]
    
    return merged_proposals

