import torch


from torch import nn


from torch.nn import functional as F


from typing import Tuple


from mmrotate.models.builder import GATE_FUNCS


@GATE_FUNCS.register_module()
class ModalityAwareLogit(nn.Module):
    def __init__(self, 
                 in_channels, 
                 modal_names,
                 n_routed_experts):
        super().__init__()
        self.modal_names = modal_names
        self.modality_embeddings = nn.Embedding(len(modal_names), in_channels)
        self.shared_fc = nn.Linear(in_channels, n_routed_experts)

    def forward(self, x: torch.Tensor, dataset_name: str):
        modality_ids = self.modal_names.index(dataset_name)
        modality_ids = torch.tensor(modality_ids, dtype=torch.long, device=x.device)
        mod_embed = self.modality_embeddings(modality_ids)  # [B, D]
        gated_input = x + mod_embed
        return self.shared_fc(gated_input)


class ModalityPriorMixin:
    """Focus Affinity initialization."""

    applies_modality_prior_affinity = True

    def _init_modality_prior(self,
                             modal_names,
                             n_routed_experts,
                             prior_strength=0.3,
                             prior_floor=0.1,
                             modality_wise_prior_strength=False,
                             prior_strength_max=None,
                             trainable_prior_affinity=True):
        self.modal_names = modal_names
        if isinstance(prior_floor, dict):
            floor_list = [float(prior_floor.get(m, 0.0)) for m in modal_names]
        elif isinstance(prior_floor, (list, tuple)):
            floor_list = [float(x) for x in prior_floor]
            if len(floor_list) != len(modal_names):
                raise ValueError(
                    'prior_floor list length must match #modalities.')
        else:
            floor_list = [float(prior_floor)] * len(modal_names)
        if any(f < 0.0 for f in floor_list):
            raise ValueError('prior_floor must be non-negative.')
        self.prior_floor = torch.tensor(floor_list, dtype=torch.float32)
        self.prior_strength = float(prior_strength)
        self.modality_wise_prior_strength = bool(
            modality_wise_prior_strength)
        self.prior_strength_max = (
            None if prior_strength_max is None else float(prior_strength_max))
        self.trainable_prior_affinity = bool(trainable_prior_affinity)
        self.center_prior = True
        prior_affinity = torch.zeros(len(modal_names), n_routed_experts)
        if self.trainable_prior_affinity:
            self.prior_affinity = nn.Parameter(
                prior_affinity, requires_grad=True)
        else:
            self.register_buffer('prior_affinity', prior_affinity)
        self.register_buffer(
            'prior_affinity_init',
            torch.zeros(len(modal_names), n_routed_experts))
        self.register_buffer(
            'prior_log_template',
            torch.zeros(len(modal_names), n_routed_experts))
        self.register_buffer(
            'prior_probs',
            torch.zeros(len(modal_names), n_routed_experts))
        if self.modality_wise_prior_strength:
            if self.prior_strength <= 0.0:
                raise ValueError(
                    'prior_strength must be positive when '
                    'modality_wise_prior_strength=True.')
            if self.prior_strength_max is None:
                self.register_buffer(
                    'prior_strength_center',
                    torch.log(torch.expm1(torch.ones(1))).squeeze(0))
                init_raw = torch.tensor(0.0)
            elif self.prior_strength < self.prior_strength_max:
                init_prob = torch.tensor(
                    self.prior_strength / self.prior_strength_max)
                init_raw = torch.logit(init_prob.clamp(1e-6, 1 - 1e-6))
            else:
                raise ValueError(
                    'prior_strength must be in (0, prior_strength_max) when '
                    'modality_wise_prior_strength=True.')
            self.prior_strength_raw = nn.Parameter(
                torch.full((len(modal_names),), float(init_raw)))

    def _prior_strength_vector(self) -> torch.Tensor:
        if self.modality_wise_prior_strength:
            if self.prior_strength_max is None:
                scale = F.softplus(
                    self.prior_strength_raw + self.prior_strength_center)
                return self.prior_strength * scale
            return self.prior_strength_max * torch.sigmoid(
                self.prior_strength_raw)
        return self.prior_affinity.new_full(
            (len(self.modal_names),), self.prior_strength)

    def _current_prior_affinity(self) -> torch.Tensor:
        if not self.modality_wise_prior_strength:
            return self.prior_affinity

        base_affinity = (
            self.prior_log_template *
            self._prior_strength_vector().unsqueeze(-1))
        if self.trainable_prior_affinity:
            residual = self.prior_affinity - self.prior_affinity_init
            return base_affinity + residual
        return base_affinity

    def set_modality_prior(self, prior_weights: torch.Tensor):
        weights = prior_weights.detach().to(
            device=self.prior_affinity.device,
            dtype=self.prior_affinity.dtype)
        weights = weights.clamp_min(0.0) + self.prior_floor.view(-1, 1).to(
            weights)
        prior_probs = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            1e-12)

        log_prior = prior_probs.clamp_min(1e-12).log()
        if self.center_prior:
            log_prior = log_prior - log_prior.mean(dim=-1, keepdim=True)
        if self.modality_wise_prior_strength:
            prior_affinity = (
                log_prior *
                self._prior_strength_vector().detach().unsqueeze(-1))
        else:
            prior_affinity = log_prior * self.prior_strength

        with torch.no_grad():
            self.prior_log_template.copy_(log_prior)
            self.prior_affinity.copy_(prior_affinity)
            self.prior_affinity_init.copy_(prior_affinity)
            self.prior_probs.copy_(prior_probs)

    def _prior_affinity_for(self, dataset_name: str) -> torch.Tensor:
        modality_id = self.modal_names.index(dataset_name)
        return self._current_prior_affinity()[modality_id]


@GATE_FUNCS.register_module()
class CosineLogit(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 n_routed_experts, 
                 init_t = 0.5,
                 **kwargs):
        super(CosineLogit, self).__init__()
        proj_dim = min(in_channels//2, 256)
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = torch.nn.Linear(in_channels, proj_dim)
        self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, n_routed_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)

    def _normalize_feature(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1)

    def _cosine_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(
            self._normalize_feature(self.cosine_projector(x)),
            F.normalize(self.sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        return logits * logit_scale

    def forward(self, x):
        return self._cosine_logits(x)


@GATE_FUNCS.register_module()
class ModalityPriorCosineLogit(ModalityPriorMixin, CosineLogit):
    """Cosine scores with modality-dependent affinity bias."""

    requires_modal_name = True
    def __init__(self,
                 in_channels,
                 n_routed_experts,
                 modal_names,
                 init_t=0.5,
                 norm_dim='legacy',
                 prior_strength=0.3,
                 prior_floor=0.1,
                 modality_wise_prior_strength=False,
                 prior_strength_max=None,
                 trainable_prior_affinity=True,
                 use_mmtoken_prior=False,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            n_routed_experts=n_routed_experts,
            init_t=init_t,
            **kwargs)
        self.norm_dim = norm_dim
        self.use_mmtoken_prior = bool(use_mmtoken_prior)
        self._init_modality_prior(
            modal_names=modal_names,
            n_routed_experts=n_routed_experts,
            prior_strength=prior_strength,
            prior_floor=prior_floor,
            modality_wise_prior_strength=modality_wise_prior_strength,
            prior_strength_max=prior_strength_max,
            trainable_prior_affinity=trainable_prior_affinity)
        if self.use_mmtoken_prior:
            hidden_dim = max(8, in_channels // 4)
            self.mmtokens = nn.Embedding(len(modal_names), in_channels)
            self.mmtoken_norm = nn.LayerNorm(in_channels)
            self.mmtoken_delta = nn.Sequential(
                nn.Linear(in_channels, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, 1, bias=True))
            self.register_buffer(
                'mmtoken_scale_center',
                torch.log(torch.expm1(torch.ones(1))).squeeze(0))
            nn.init.normal_(self.mmtokens.weight, std=0.02)
            nn.init.zeros_(self.mmtoken_delta[-1].weight)
            nn.init.zeros_(self.mmtoken_delta[-1].bias)
            self.latest_mmtoken_prior_scales = {}

    def _normalize_feature(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_dim == 'channel':
            dim = -1
        elif self.norm_dim == 'legacy':
            dim = 1 if x.dim() > 1 else 0
        else:
            raise ValueError(f'Unknown norm_dim: {self.norm_dim}')
        return F.normalize(x, dim=dim)

    def _mmtoken_prior_scale(self, x: torch.Tensor,
                             modality_id: int) -> torch.Tensor:
        if x.dim() == 2:
            x_for_pool = x.unsqueeze(0)
            squeeze_batch = True
        elif x.dim() == 3:
            x_for_pool = x
            squeeze_batch = False
        else:
            raise ValueError(f'MMToken prior expects 2D/3D input, got {x.dim()}D')

        query = self.mmtokens.weight[modality_id]
        key = self.mmtoken_norm(x_for_pool)
        score = (key * query).sum(dim=-1) / (x_for_pool.shape[-1] ** 0.5)
        attn = score.softmax(dim=1)
        context = (attn.unsqueeze(-1) * x_for_pool).sum(dim=1)
        delta = self.mmtoken_delta(context).squeeze(-1)
        scale = F.softplus(delta + self.mmtoken_scale_center)
        if squeeze_batch:
            return scale.squeeze(0)
        return scale

    def _prior_affinity_for(self, dataset_name: str,
                            x: torch.Tensor = None) -> torch.Tensor:
        modality_id = self.modal_names.index(dataset_name)
        if not self.use_mmtoken_prior or x is None:
            return super()._prior_affinity_for(dataset_name)

        strength = self._prior_strength_vector()[modality_id]
        base_affinity = self.prior_log_template[modality_id] * strength
        scale = self._mmtoken_prior_scale(x, modality_id)
        self.latest_mmtoken_prior_scales[dataset_name] = scale.detach()

        if x.dim() == 3:
            affinity = base_affinity.view(1, 1, -1) * scale.view(-1, 1, 1)
            if self.trainable_prior_affinity:
                residual = (
                    self.prior_affinity[modality_id] -
                    self.prior_affinity_init[modality_id])
                affinity = affinity + residual.view(1, 1, -1)
            return affinity

        affinity = base_affinity * scale
        if self.trainable_prior_affinity:
            affinity = affinity + (
                self.prior_affinity[modality_id] -
                self.prior_affinity_init[modality_id])
        return affinity

    def content_logits(self, x: torch.Tensor,
                       dataset_name: str = None) -> torch.Tensor:
        del dataset_name
        return self._cosine_logits(x)

    def forward(self, x: torch.Tensor, dataset_name: str):
        # Add the modality-dependent affinity bias to the content scores.
        return self.content_logits(x, dataset_name) + self._prior_affinity_for(
            dataset_name, x)

