from typing import Tuple, List, Dict

import torch
from torch import nn

from ...builder import GATE_FUNCS


class DeepSeekShareGate(nn.Module):
    """Multi-Branch routing."""

    def __init__(self, in_channels: List[int], args: Dict, eps: float = 1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.n = int(args['n_routed_experts'])
        self.topk = int(args['n_activated_experts'])
        self.share_idx = list(args.get('share_expert_indices', []))
        share_set = set(self.share_idx)
        self.spec_idx = [e for e in range(self.n) if e not in share_set]
        self.n_share = len(self.share_idx)
        # routed slots among specific experts (keep total active == topk)
        self.topk_spec = max(1, self.topk - self.n_share)
        self.route_scale = float(args.get('route_scale', 1.0))
        self.act_func = args.get('act_func', 'softmax')
        self.eps = eps

        gate_cfg = dict(args['gate_func_cfg'])
        gate_cfg['in_channels'] = self.in_channels
        gate_cfg['n_routed_experts'] = self.n
        self.gate_logit = GATE_FUNCS.build(gate_cfg)

        self.register_buffer('_spec_idx_t', torch.tensor(self.spec_idx, dtype=torch.long))

    def forward(self, expert_input: torch.Tensor, experts: nn.ModuleList,
                batch_indices=None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, C = expert_input.shape
        x = expert_input.reshape(-1, C)

        # (1) always-on dense shared experts
        share_out = torch.zeros_like(x)
        for e in self.share_idx:
            share_out = share_out + experts[e](x)
        if self.n_share > 0:
            share_out = share_out / self.n_share

        # (2) route among specific experts (top-k')
        logits = self.gate_logit(x)                      # [N, n]
        spec_logits = logits[:, self._spec_idx_t]        # [N, n_spec]
        k = min(self.topk_spec, spec_logits.shape[-1])
        top_w, top_pos = spec_logits.topk(k, dim=-1)     # [N, k]
        if self.act_func == 'softmax':
            top_w = top_w.softmax(dim=-1, dtype=torch.float32).clamp(min=1e-9)
        elif self.act_func == 'sigmoid':
            top_w = top_w.sigmoid()
        else:
            raise ValueError(f'Unknown act_func: {self.act_func}')
        top_w = top_w * self.route_scale
        top_g = self._spec_idx_t[top_pos]                # [N, k] global expert ids

        # (3) sparse combine over selected specific experts
        spec_out = torch.zeros_like(x)
        for e in self.spec_idx:
            idx, slot = torch.where(top_g == e)
            if idx.numel() > 0:
                spec_out[idx] += experts[e](x[idx]) * top_w[idx, slot, None]

        y = (share_out + spec_out).reshape(B, H, W, C)

        return y.new_zeros(()), y
