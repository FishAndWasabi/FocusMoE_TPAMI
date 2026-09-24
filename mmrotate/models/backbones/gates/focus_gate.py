"""FocusMoE inference: content scores plus learned Focus Affinity."""
import torch
from torch import nn
from ..utils import SparseDispatcher
from ...builder import GATE_FUNCS


class FocusGate(nn.Module):
    def __init__(self, in_channels, args):
        super().__init__()
        self.n = args['n_routed_experts']
        self.topk = args['n_activated_experts']
        self.route_scale = args.get('route_scale', 1.0)
        self.modal_names = list(args['specific_expert_indices'])
        self.act_func = args.get('act_func', 'softmax')
        self.register_buffer('expert_bias', torch.zeros(self.n))
        if args.get('noisy_gating', False):
            self.w_noise = nn.Parameter(torch.zeros(in_channels, self.n))
        self.register_buffer('mean', torch.tensor([0.0]))
        self.register_buffer('std', torch.tensor([1.0]))
        config = dict(args['gate_func_cfg'], in_channels=in_channels,
                      n_routed_experts=self.n, modal_names=self.modal_names)
        self.gate_func = GATE_FUNCS.build(config)
        shape = (len(self.modal_names), self.n)
        self.register_buffer('preference_mask', torch.zeros(shape))
        for name in ('specific_group_mask', 'share_group_mask', 'other_group_mask'):
            self.register_buffer(name, torch.zeros(shape, dtype=torch.bool))
        # Initialize Focus Affinity from the role prior.
        for i, modality in enumerate(self.modal_names):
            specific = args['specific_expert_indices'][modality]
            shared = args['share_expert_indices']
            self.preference_mask[i, specific] = args.get('specific_preference_value', 1.0)
            self.preference_mask[i, shared] = args.get('shared_preference_value', 0.5)
            self.specific_group_mask[i, specific] = True
            self.share_group_mask[i, shared] = True
            self.other_group_mask[i] = ~(self.specific_group_mask[i] | self.share_group_mask[i])
        self.gate_func.set_modality_prior(self.preference_mask)

    def forward(self, features, experts, batch_indices):
        shape = features.shape
        channels = shape[-1]
        flat = features.reshape(shape[0], -1, channels)
        results = []
        for modality, indices in batch_indices.items():
            if not len(indices):
                continue
            selected = flat[indices]
            logits = self.gate_func(selected, modality).reshape(-1, self.n)
            logits = logits + torch.zeros_like(self.preference_mask[self.modal_names.index(modality)])
            if self.act_func == 'softmax':
                probabilities = logits.softmax(dim=-1)
            elif self.act_func == 'sigmoid':
                probabilities = logits.sigmoid()
            else:
                raise ValueError('Unknown gate activation: ' + self.act_func)
            indices_topk = logits.topk(min(self.topk + 1, self.n), dim=-1).indices[..., :self.topk]
            weights = probabilities.gather(-1, indices_topk)
            gates = torch.zeros_like(logits)
            gates.scatter_(-1, indices_topk, weights)
            dispatcher = SparseDispatcher(self.n, gates * self.route_scale)
            inputs = dispatcher.dispatch(selected.reshape(-1, channels))
            outputs = [expert(x).reshape(-1, channels) if x.shape[0]
                       else x.new_empty((0, channels))
                       for expert, x in zip(experts, inputs)]
            combined = dispatcher.combine(outputs)
            results.append(combined.reshape((selected.shape[0],) + shape[1:]))
        return features.new_zeros(()), torch.cat(results, dim=0)
