"""Multi-Branch and Multi-Stage routing."""
from typing import Tuple, Dict
import torch
from torch import nn
from ...builder import GATE_FUNCS


def _as_dict(args):
    return dict(args)


class _ControlledGate(nn.Module):
    """Content router and sparse expert execution used by the table-3 controls."""
    def __init__(self, in_channels, args):
        super().__init__()
        self.in_channels = int(in_channels)
        self.n = int(args['n_routed_experts'])
        self.topk = int(args['n_activated_experts'])
        self.route_scale = float(args.get('route_scale', 1.0))
        cfg = dict(args['gate_func_cfg'], in_channels=self.in_channels,
                   n_routed_experts=self.n)
        self.gate_logit = GATE_FUNCS.build(cfg)

    @staticmethod
    def _flatten(features: torch.Tensor):
        if features.dim() == 4:
            batch, height, width, channels = features.shape
            return features.reshape(-1, channels), (batch, height, width, channels)
        if features.dim() == 3:
            batch, tokens, channels = features.shape
            return features.reshape(-1, channels), (batch, tokens, channels)
        raise ValueError(f"Expected a 3D or 4D token tensor, got {features.dim()}D")

    @staticmethod
    def _restore(tokens: torch.Tensor, shape: Tuple[int, ...]):
        return tokens.reshape(shape)

    @staticmethod
    def _run_sparse(expert_input, experts, expert_ids, expert_weights, n_experts):
        output = torch.zeros_like(expert_input)
        for expert_id in range(n_experts):
            token_ids, slots = torch.where(expert_ids == expert_id)
            if token_ids.numel() == 0:
                continue
            value = experts[expert_id](expert_input[token_ids])
            output[token_ids] += value * expert_weights[token_ids, slots, None]
        return output


class ControlledDeepSeekGate(_ControlledGate):
    def __init__(self, in_channels: int, args: Dict, eps: float = 1e-6):
        del eps
        super().__init__(in_channels, args)
        args = _as_dict(args)
        self.shared = tuple(int(v) for v in args["share_expert_indices"])
        shared_set = set(self.shared)
        self.routed = tuple(v for v in range(self.n) if v not in shared_set)
        if len(self.shared) not in (1, 2) or self.topk != len(self.shared) + 1:
            raise ValueError(
                "DeepSeek-style control requires one routed expert in addition to the shared experts")
        self.register_buffer("_routed", torch.tensor(self.routed, dtype=torch.long))

    def forward(self, features, experts, batch_indices=None):
        tokens, shape = self._flatten(features)
        logits = self.gate_logit(tokens)
        routed_logits = logits.index_select(-1, self._routed)
        routed_prob = routed_logits.softmax(dim=-1)
        routed_pos = routed_prob.argmax(dim=-1, keepdim=True)
        routed_ids = self._routed[routed_pos]

        shared_ids = torch.tensor(self.shared, device=tokens.device).view(
            1, len(self.shared))
        shared_ids = shared_ids.expand(tokens.shape[0], -1)
        active_ids = torch.cat([shared_ids, routed_ids], dim=-1)
        probabilities = logits.softmax(dim=-1)
        active_weights = probabilities.gather(1, active_ids) * self.route_scale

        output = self._run_sparse(tokens, experts, active_ids, active_weights, self.n)
        return output.new_zeros(()), self._restore(output, shape)


class ControlledMoCEGate(_ControlledGate):
    _GROUP_KEY = '__moce_group_ids__'

    def __init__(self, in_channels, args):
        super().__init__(in_channels, args)
        self.register_buffer('expert_groups', torch.tensor(args['expert_groups'], dtype=torch.long))
        self.register_buffer('cluster_centers', torch.zeros(2, args['cluster_center_dim']))

    def _select_groups(self, features, batch_indices):
        pooled = features.reshape(features.shape[0], -1, features.shape[-1]).mean(dim=1)
        if pooled.shape[-1] != self.cluster_centers.shape[-1]:
            raise ValueError(
                "The first MoE input does not match the fitted MoCE center dimension: "
                f"{pooled.shape[-1]} vs {self.cluster_centers.shape[-1]}")
        distance = torch.cdist(pooled.float(), self.cluster_centers.float())
        groups = distance.argmin(dim=-1)
        batch_indices[self._GROUP_KEY] = groups.detach()
        return groups

    def forward(self, features, experts, batch_indices=None):
        if batch_indices is None:
            raise ValueError("Multi-Stage Routing requires a persistent batch-index dictionary")
        group_ids = batch_indices.get(self._GROUP_KEY)
        if group_ids is None:
            group_ids = self._select_groups(features, batch_indices)
        else:
            group_ids = torch.as_tensor(group_ids, device=features.device, dtype=torch.long)

        image_tokens = features.reshape(features.shape[0], -1, features.shape[-1])
        output = torch.zeros_like(image_tokens)
        for group_id in range(2):
            image_ids = torch.where(group_ids == group_id)[0]
            if image_ids.numel() == 0:
                continue
            tokens = image_tokens.index_select(0, image_ids).reshape(-1, features.shape[-1])
            group = self.expert_groups[group_id]
            group_logits = self.gate_logit(tokens).index_select(-1, group)
            group_prob = group_logits.softmax(dim=-1)
            weights, positions = group_prob.topk(self.topk, dim=-1)
            active_ids = group[positions]
            weights = weights * self.route_scale
            group_output = self._run_sparse(tokens, experts, active_ids, weights, self.n)
            output[image_ids] = group_output.reshape(
                image_ids.numel(), -1, features.shape[-1])

        return output.new_zeros(()), output.reshape_as(features)
