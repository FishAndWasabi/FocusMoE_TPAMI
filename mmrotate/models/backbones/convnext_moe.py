# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import chain
from typing import Sequence, Tuple, Optional
from functools import partial
from collections import OrderedDict

from typing import List

import os

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

import numpy as np


from mmengine.model import ModuleList, Sequential
from mmengine.logging import MMLogger 
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.model import BaseModule

import torch
import torch.nn as nn
from ..builder import ROTATED_BACKBONES

from mmcv.runner import BaseModule
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from mmcv.cnn import build_activation_layer

from .gates import (FocusGate, SM3DetGate, DeepSeekShareGate,
                    ControlledDeepSeekGate, ControlledMoCEGate)
from .utils import build_LayerNorm2d_layer
import os

class GRN(nn.Module):
    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor, data_format='channel_first'):
        if data_format == 'channel_last':
            gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
            x = self.gamma * (x * nx) + self.beta + x
        elif data_format == 'channel_first':
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
            x = self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(
                1, -1, 1, 1) + x
        return x
        
class FFN(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 pw_conv,
                 act_cfg=dict(type='GELU'),
                 use_grn=False):
        super().__init__()
        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = build_activation_layer(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        if use_grn:
            self.grn = GRN(mid_channels)
        else:
            self.grn = None
    def forward(self, x):
        x = self.pointwise_conv1(x)
        x = self.act(x)
        if self.grn is not None:
            if len(x.shape) == 2:  # [B, C]
                x = x.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, C]
            x = self.grn(x, data_format='channel_last')
            if len(x.shape) == 4:  # [B, 1, 1, C]
                x = x.squeeze(1).squeeze(1)  # [B, C]
        x = self.pointwise_conv2(x) 
        return x

GATES = dict(focusgate=FocusGate)


class MoE(nn.Module):
    """Expert routing."""
    def __init__(self, in_channels, mid_channels, moe_args, pw_conv=nn.Conv2d,
                 act_cfg=dict(type='GELU'), use_grn=False,
                 enable_offloading=False, lut_dir=None, cache_size=10,
                 prune_modality=None):
        super().__init__()
        if enable_offloading or prune_modality is not None:
            raise ValueError('This package contains full detector checkpoints.')
        self.moe_args = moe_args
        self.enable_offloading = False
        self.prune_modality = None
        for i, args in enumerate(moe_args):
            gate_class = {'focusgate': FocusGate, 'sm3detgate': SM3DetGate,
                          'deepseeksharegate': DeepSeekShareGate,
                          'controlled_deepseek': ControlledDeepSeekGate,
                          'controlled_moce': ControlledMoCEGate}[args.type]
            setattr(self, f'gate_{i}', gate_class(in_channels, args))
            setattr(self, f'expert_{i}', nn.ModuleList([
                FFN(in_channels, int(mid_channels * args.mid_scale),
                    pw_conv, act_cfg, use_grn)
                for _ in range(args.n_routed_experts)]))

    def _recalculate_batch_indices(self, features: torch.Tensor, original_batch_indices: dict) -> dict:
        """Recalculate batch_indices based on current feature tensor batch size."""
        if original_batch_indices is None:
            return None
        
        batch_size = features.shape[0]
        
        new_batch_indices = {}
        current_idx = 0
        
        for modal_name, original_indices in original_batch_indices.items():
            modal_batch_size = len(original_indices)
            
            if current_idx >= batch_size:
                break
                
            end_idx = min(current_idx + modal_batch_size, batch_size)
            new_batch_indices[modal_name] = list(range(current_idx, end_idx))
            current_idx = end_idx
            
            if current_idx >= batch_size:
                break
        
        total_indices = sum(len(indices) for indices in new_batch_indices.values())
        if total_indices != batch_size:
            logger = MMLogger.get_current_instance()
            if logger is not None:
                logger.warning(
                    f"Batch size mismatch in MoE: expected {batch_size}, got {total_indices}. "
                    f"This may indicate an issue with batch_indices handling in cascaded MoE."
                )
        
        return new_batch_indices
    
    def forward(self, x: torch.Tensor, batch_indices=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the MoE module."""
        total_gate_loss = 0
        gate_log_vars = {}
        y = x
        current_batch_indices = batch_indices
        
        for i in range(len(self.moe_args)):
            gate = getattr(self, f'gate_{i}')
            experts = getattr(self, f'expert_{i}')
            
            if current_batch_indices is not None and i > 0:
                current_batch_indices = self._recalculate_batch_indices(y, batch_indices)
            
            gate_loss, y = gate(y, experts, current_batch_indices)
            total_gate_loss = total_gate_loss + gate_loss
            for key, value in getattr(gate, 'latest_log_vars', {}).items():
                gate_log_vars.setdefault(key, []).append(value.detach())
        avg_gate_loss = total_gate_loss / torch.tensor(max(len(self.moe_args), 1), device=total_gate_loss.device, dtype=total_gate_loss.dtype)
        self.latest_gate_log_vars = {
            key: sum(values) / len(values)
            for key, values in gate_log_vars.items()
        }
        return y, avg_gate_loss
    


class ConvNeXtBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, 
                                  padding=3),
                 norm_cfg=dict(type='LN2d', 
                               eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 moe_args:List[dict]=[],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False,
                 enable_offloading: bool = False,
                 lut_dir: Optional[str] = None,
                 cache_size: int = 10,
                 prune_modality: Optional[str] = None):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg)
        self.linear_pw_conv = linear_pw_conv
        self.norm = build_LayerNorm2d_layer(norm_cfg, in_channels)
        mid_channels = int(mlp_ratio * in_channels) 
        if self.linear_pw_conv:
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)
        self.ffn = FFN(in_channels,mid_channels,pw_conv,act_cfg,use_grn)

        self.moe_args = moe_args
        if len(moe_args) != 0:
            
            self.ffn = MoE(in_channels,
                           mid_channels,
                           moe_args,
                           pw_conv=pw_conv,
                           use_grn=use_grn,
                           act_cfg=act_cfg,
                           enable_offloading=enable_offloading,
                           lut_dir=lut_dir,
                           cache_size=cache_size,
                           prune_modality=prune_modality)
        else:
            self.ffn = FFN(in_channels,mid_channels,pw_conv,act_cfg,use_grn)


        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        
        
        
    def forward(self, x, batch_indices=None):
        def _inner_forward(x, batch_indices):
            shortcut = x
            gate_loss = x.sum().detach() * 0

            x = self.depthwise_conv(x)
            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                if len(self.moe_args) != 0:
                    x, gate_loss = self.ffn(x, batch_indices)
                else:
                    x = self.ffn(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                if len(self.moe_args) != 0:
                    x, gate_loss = self.ffn(x, batch_indices)
                else:
                    x = self.ffn(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x, gate_loss

        if self.with_cp and x.requires_grad:
            x, gate_loss = cp.checkpoint(_inner_forward, x, batch_indices)
        else:
            x, gate_loss = _inner_forward(x, batch_indices)
        return x, gate_loss


@ROTATED_BACKBONES.register_module()
class ConvNeXtMoE(BaseModule):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'swin_large': {
            'depths': [2,  2, 18,  2],
            'channels': [192, 384, 768, 1536]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 moe_indices = [[],[],[],[]],
                 moe_args:List[dict] = [],
                 frozen_stages=0,
                 gap_before_final_norm=False,
                 with_cp=False,
                 return_gate_loss=True,
                 freeze_non_moe=False,
                 enable_offloading: bool = False,
                 lut_dir: Optional[str] = None,
                 cache_size: int = 10,
                 prune_modality: Optional[str] = None,
                 expert_init_noise_scale: float = 0.0,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)
        self.expert_init_noise_scale = float(expert_init_noise_scale)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices
        self.moe_indices = moe_indices
        self.moe_args = moe_args
        self.enable_offloading = enable_offloading
        self.lut_dir = lut_dir
        self.cache_size = cache_size
        self.prune_modality = prune_modality
        
        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm
        self.freeze_non_moe = freeze_non_moe

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_LayerNorm2d_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_LayerNorm2d_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)
            moe_id = [list(range(depth))[q] for q in self.moe_indices[i] if q < depth]
            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    moe_args = moe_args if j in moe_id else [],
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp,
                    enable_offloading=self.enable_offloading if j in moe_id else False,
                    lut_dir=self.lut_dir,
                    cache_size=self.cache_size,
                    prune_modality=self.prune_modality if j in moe_id else None) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_LayerNorm2d_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()
        
        self.downsample_layers[0] = nn.Sequential(build_LayerNorm2d_layer(norm_cfg, self.channels[0]))
        
        self.dataset_stems = nn.Conv2d(in_channels,
                                       self.channels[0],
                                       kernel_size=stem_patch_size,
                                       stride=stem_patch_size)
        self.return_gate_loss = return_gate_loss
        
    def forward(self, inputs, dataset_names=['single']):
        
        if len(dataset_names) == 1:
            inputs = [inputs] 
        
        batch_indices = {
            dataset_name: list(range(final_index, final_index + inputs[i].shape[0]))
            for i, dataset_name in enumerate(dataset_names)
            for final_index in [sum(inputs[j].shape[0] for j in range(i))]
        }
        
        inputs = torch.cat(inputs, dim=0)
        features = self.dataset_stems(inputs)
        outs = []
        gate_losses = []
        gate_log_vars = {}
        for i, stage in enumerate(self.stages):
            features = self.downsample_layers[i](features)
            for each_layer in stage.children():
                features, gate_loss = each_layer(features, batch_indices)
                if len(getattr(each_layer, 'moe_args', [])) > 0:
                    gate_losses.append(gate_loss)
                    for key, value in getattr(
                            each_layer.ffn, 'latest_gate_log_vars', {}).items():
                        gate_log_vars.setdefault(key, []).append(value.detach())
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = features.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(features))
        gate_losses = sum(gate_losses) / len(gate_losses) if len(gate_losses) > 0 else features.sum().detach() * 0
        self.latest_gate_log_vars = {
            key: sum(values) / len(values)
            for key, values in gate_log_vars.items()
        }
        if self.return_gate_loss:
            return tuple(outs), gate_losses
        else:
            return tuple(outs)
    
    def forward_dummy(self, inputs):
        """Used for computing network FLOPs."""
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)
        
        dataset_names = ['sar']
        inputs_list = [inputs]
        
        batch_indices = {
            dataset_name: list(range(final_index, final_index + inputs_list[i].shape[0]))
            for i, dataset_name in enumerate(dataset_names)
            for final_index in [sum(inputs_list[j].shape[0] for j in range(i))]
        }
        
        inputs_concat = torch.cat(inputs_list, dim=0)
        features = self.dataset_stems(inputs_concat)
        outs = []
        for i, stage in enumerate(self.stages):
            features = self.downsample_layers[i](features)
            for each_layer in stage.children():
                features, _ = each_layer(features, batch_indices)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = features.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(features))
        
        return tuple(outs)
        
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def _freeze_non_moe(self):
        """Freeze all parameters except MoE-related parameters."""
        if not self.freeze_non_moe:
            return
            
        for name, param in self.named_parameters():
            is_moe_param = False
            
            if 'ffn.gate_' in name or 'ffn.expert_' in name:
                is_moe_param = True
            
            if not is_moe_param:
                param.requires_grad = False

    def prune_for_modality(self, modality_name: str, include_share_experts: bool = True):
        """Prune MoE layers for single modality inference."""
        if self.training:
            raise RuntimeError("Pruning should only be done in eval mode")
        
        all_pruning_stats = {
            'original_experts': [],
            'pruned_experts': [],
            'reduction_ratio': []
        }
        
        for stage_idx, stage in enumerate(self.stages):
            for block_idx, block in enumerate(stage):
                if not hasattr(block, 'ffn'):
                    continue
                
                ffn = block.ffn
                
                if not hasattr(ffn, 'moe_args') or len(ffn.moe_args) == 0:
                    continue
                
                if not hasattr(ffn, 'prune_for_modality'):
                    continue
                
                try:
                    layer_stats = ffn.prune_for_modality(modality_name, include_share_experts)
                    if layer_stats:
                        all_pruning_stats['original_experts'].extend(layer_stats.get('original_experts', []))
                        all_pruning_stats['pruned_experts'].extend(layer_stats.get('pruned_experts', []))
                        all_pruning_stats['reduction_ratio'].extend(layer_stats.get('reduction_ratio', []))
                except Exception as e:
                    print(f"Warning: Failed to prune MoE layer at stage {stage_idx}, block {block_idx}: {e}")
                    continue
        
        return all_pruning_stats

    def train(self, mode=True):
        super(ConvNeXtMoE, self).train(mode)
        self._freeze_stages()
        self._freeze_non_moe()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter."""

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2

    def init_weights(self): 
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None or not os.path.exists(self.init_cfg.checkpoint):
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training starts from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.)
        else:
            
            assert 'checkpoint' in self.init_cfg, f'Only support specifying `Pretrained` in `init_cfg` for {self.__class__.__name__}'
            
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            _state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
            
            if self.init_cfg.reformat:
                model_state_dict = self.state_dict()
                state_dict = OrderedDict()

                for k, v in _state_dict.items():
                    if k.startswith('backbone.'):
                        k = k[9:]
                        if 'downsample_layers.0.0' in k:
                            state_dict[k.replace('downsample_layers.0.0', 'dataset_stems')] = v
                        elif 'downsample_layers.0.1' in k:
                            state_dict[k.replace('downsample_layers.0.1', 'downsample_layers.0.0')] = v
                        elif 'pointwise_conv' in k or 'grn' in k:
                            stage_splits = k.split('.')
                            stage_ind = int(stage_splits[1])
                            block_ind = int(stage_splits[2])
                            if block_ind in self.moe_indices[stage_ind]:
                                for expert_idx, moe_arg in enumerate(self.moe_args):
                                    for i in range(moe_arg.n_routed_experts):
                                        new_k = k.replace('pointwise_conv', f'ffn.expert_{expert_idx}.{i}.pointwise_conv') \
                                                if 'pointwise_conv' in k else \
                                                k.replace('grn', f'ffn.expert_{expert_idx}.{i}.grn')
                                        if "grn.gamma" in new_k or "grn.beta" in new_k:
                                            v = v.reshape(-1)
                                        if new_k in model_state_dict and model_state_dict[new_k].shape == v.shape:
                                            state_dict[new_k] = v
                            else:
                                new_k = k.replace('pointwise_conv', 'ffn.pointwise_conv') if 'pointwise_conv' in k else k.replace('grn', 'ffn.grn')
                                if "grn.gamma" in new_k or "grn.beta" in new_k:
                                    v = v.reshape(-1)
                                state_dict[new_k] = v
                        else:
                            state_dict[k] = v

                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
            else:
                model_state_dict = self.state_dict()
                state_dict = OrderedDict()
                
                for k, v in _state_dict.items():
                    if not k.startswith('backbone.'):
                        continue                    
                    k = k[9:]  # Remove 'backbone.' prefix
                    state_dict[k] = v
                
                if state_dict and list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
            warnings = self.load_state_dict(state_dict, strict=False)
            if warnings.missing_keys or warnings.unexpected_keys:
                logger.warn(f'Warnings when loading state_dict: {warnings}')

        if self.expert_init_noise_scale > 0:
            self._perturb_experts()

    def _perturb_experts(self):
        """Break expert clone-init by adding per-expert Gaussian noise."""
        scale = float(self.expert_init_noise_scale)
        with torch.no_grad():
            for name, p in self.named_parameters():
                if '.expert_' not in name or p.dim() < 2:
                    continue
                w_std = p.detach().std().item()
                if w_std <= 0:
                    continue
                p.add_(torch.randn_like(p) * w_std * scale)
