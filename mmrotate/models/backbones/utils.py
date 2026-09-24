from typing import List, Literal, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer,
                      constant_init, trunc_normal_init)
@dataclass
class MoEArgs:
    type: str = "refinedgatev1"
    mid_scale: float = 1.0
    n_routed_experts: int = 64
    n_activated_experts: int = 6
    route_scale: float = 1.0
    gate_func_cfg: dict = field(default_factory=dict)
    act_func: Literal["softmax", "sigmoid"] = "softmax"
    loss_coef_cfg: dict = field(default_factory=dict)
    noisy_gating: bool = False
    noise_epsilon: float = 1e-2
    use_dispatch: bool = False
    dataset_names: List[str] = field(default_factory=list)

class GRN(nn.Module):
    """Global Response Normalization Module.

    Come from `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
    Autoencoders <http://arxiv.org/abs/2301.00808>`_

    Args:
        in_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-6.
    """

    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        gx = torch.norm(x, p=2, dim=0, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        x = self.gamma * (x * nx) + self.beta + x
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
        # print(x)
        x = self.act(x)
        if self.grn is not None:
            x = self.grn(x)
         
        x = self.pointwise_conv2(x) 
        return x

def cv_squared(x, eps=1e-10):
    """
    计算变异系数的平方 (coefficient of variation squared)
    CV^2 = var(x) / mean(x)^2
    数值稳定版本
    """
    # if only num_experts = 1
    if x.shape[0] == 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    if len(x.shape) == 2:
        x = x.sum(dim=0)
    
    x_float = x.float()
    mean_x = x_float.mean()
    var_x = x_float.var()
    
    # 避免除以接近零的平方均值
    return var_x / (mean_x ** 2 + eps)

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]
    def forward(self, x, data_format='channel_first'):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            # If the output is discontiguous, it may cause some unexpected
            # problem in the downstream tasks
            x = x.permute(0, 3, 1, 2).contiguous()
        return x
def build_LayerNorm2d_layer(cfg: dict, num_features: int) -> nn.Module:
 
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    norm_layer = LayerNorm2d
    # if norm_layer is None:
    #     raise KeyError(f'Cannot find {layer_type} in registry under scope '
    #                    f'name {MODELS.scope}')
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    layer = norm_layer(num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return layer


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        
        # If all gates are zero, select the first expert for all inputs
        if torch.sum(gates) == 0:
            self._gates = torch.zeros_like(gates)
            self._gates[:, 0] = 1.0
            
        # sort experts
        nonzero_gates = torch.nonzero(self._gates)
        sorted_experts, index_sorted_experts = nonzero_gates.sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = nonzero_gates[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (self._gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = self._gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        if torch.is_grad_enabled():
            zeros = torch.zeros(
                self._gates.size(0), expert_out[-1].size(1),
                requires_grad=True, device=stitched.device)
            stitched = stitched.float()
        else:
            zeros = stitched.new_zeros(
                self._gates.size(0), expert_out[-1].size(1))
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
