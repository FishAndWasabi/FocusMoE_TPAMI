"""SM3Det top-k routing."""
import torch
from torch import nn
from ..utils import SparseDispatcher
from ...builder import GATE_FUNCS


class SM3DetGate(nn.Module):
    def __init__(self, in_channels, args):
        super().__init__()
        self.n = args['n_routed_experts']
        self.topk = args['n_activated_experts']
        self.min_gate_weight = args.get('min_gate_weight', 1e-9)
        self.route_scale = args.get('route_scale', 1.0)
        self.act_func = args.get('act_func', 'softmax')
        if not args.get('use_dispatch', True):
            raise ValueError('The released SM3Det models use sparse dispatch.')
        config = dict(args['gate_func_cfg'], in_channels=in_channels,
                      n_routed_experts=self.n)
        self.gate_logit = GATE_FUNCS.build(config)
        if args.get('noisy_gating', False):
            self.w_noise = nn.Parameter(torch.zeros(in_channels, self.n))
        self.register_buffer('mean', torch.tensor([0.0]))
        self.register_buffer('std', torch.tensor([1.0]))

    def forward(self, features, experts, batch_indices=None):
        shape = features.shape
        channels = shape[-1]
        flat = features.reshape(-1, channels)
        logits = self.gate_logit(flat)
        values, indices = logits.topk(min(self.topk + 1, self.n), dim=-1)
        values, indices = values[..., :self.topk], indices[..., :self.topk]
        if self.act_func == 'softmax':
            weights = values.softmax(dim=-1, dtype=torch.float32).clamp(min=self.min_gate_weight)
        elif self.act_func == 'sigmoid':
            weights = values.sigmoid()
        else:
            raise ValueError('Unknown gate activation: ' + self.act_func)
        gates = torch.zeros_like(logits).scatter(-1, indices, weights)
        dispatcher = SparseDispatcher(self.n, gates * self.route_scale)
        inputs = dispatcher.dispatch(flat)
        outputs = [expert(x).reshape(-1, channels)
                   for expert, x in zip(experts, inputs)]
        result = dispatcher.combine(outputs).reshape(shape)
        return features.new_zeros(()), result
