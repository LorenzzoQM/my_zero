import torch
import torch.nn as nn
from typing import Union
from torch.distributions import Categorical, Normal


def scale_value_function(x, eps=0.001):
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def inverse_scale_value_function(y, eps=0.001):
    sign = torch.sign(y)
    abs_y = torch.abs(y)
    return sign * (
        ((torch.sqrt(1 + 4 * eps * (abs_y + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1
    )


def scalar_to_support(x, support):
    """
    x: (B,)
    returns: (B, SUPPORT_SIZE)
    """
    x = x.clamp(support[0], support[-1])

    low = torch.floor(x)
    high = torch.ceil(x)

    p_high = x - low
    p_low = 1.0 - p_high

    low_idx = (low - support[0]).long()
    high_idx = (high - support[0]).long()

    B = x.shape[0]
    out = torch.zeros(B, support.numel(), device=x.device)

    out.scatter_(1, low_idx.unsqueeze(1), p_low.unsqueeze(1))
    out.scatter_(1, high_idx.unsqueeze(1), p_high.unsqueeze(1))

    return out


def support_to_scalar(probs, support):
    return torch.sum(probs * support, dim=-1)


class BodyMLP(nn.Module):

    def __init__(
        self,
        width: int,
        block_depth: int,
        blocks: int,
        input_dim: int,
        output_dim: int,
        skip_connection: bool = False,
    ):
        super().__init__()
        self.width = width
        self.block_depth = block_depth
        self.blocks = blocks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.skip_connection = skip_connection
        self._create_network()

    def _create_network(self):

        in_dim = self.input_dim
        blocks = []
        for _ in range(self.blocks):
            layers = []
            for j in range(self.block_depth):
                layers.append(nn.Linear(in_dim, self.width))
                if j < self.block_depth - 1:
                    layers.append(nn.ReLU())
                in_dim = self.width
            block = nn.Sequential(*layers)
            blocks.append(block)

        self.network = nn.ModuleList(blocks)
        self.out = nn.Linear(self.width, self.output_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:

        x = observation

        for i, block in enumerate(self.network):
            if self.skip_connection and i > 0:
                x = x + block(x)
            else:
                x = block(x)

            x = torch.relu(x)

        return self.out(x)


class EncoderMLP(nn.Module):

    def __init__(self, body: nn.Module, normalize: str | None = "l2"):

        super().__init__()
        self.body = body
        self.normalize = normalize

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        s = self.body(obs)

        if self.normalize is None:
            return s

        if self.normalize == "l2":
            return s / (s.norm(dim=-1, keepdim=True) + 1e-8)

        raise ValueError(f"Unknown normalize={self.normalize}")


class DynamicsMLP(nn.Module):

    def __init__(
        self,
        body: nn.Module,
        num_actions: int,
        action_embed_dim: int,
        reward_head: nn.Module | None = None,
        normalize_latent: str | None = "l2",
        output_probabilities: bool = False,
        support: torch.Tensor = None,
        scale_value: bool = False,
    ):

        super().__init__()
        self.latent_dim = body.output_dim
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim
        self.body = body
        self.reward_head = reward_head or nn.Linear(
            self.latent_dim, len(support) if output_probabilities else 1
        )
        self.normalize_latent = normalize_latent
        if action_embed_dim > 0:
            self.action_embed = nn.Embedding(num_actions, action_embed_dim)
        else:
            self.action_embed = self.action_embed = lambda x: x.view(x.size(0), -1)
        self.output_probabilities = output_probabilities
        self.support = support
        self.scale_value = (lambda x: x) if not scale_value else scale_value_function
        self.inverse_scale_value = (
            (lambda y: y) if not scale_value else inverse_scale_value_function
        )

        if self.output_probabilities and self.support is None:
            raise ValueError("support must be provided if output_probabilities is True")

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, return_logits: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:

        action_embed = self.action_embed(action)
        x = torch.cat([state, action_embed], dim=-1)
        next_state = self.body(x)
        if self.normalize_latent:
            next_state = next_state / (next_state.norm(dim=-1, keepdim=True) + 1e-8)

        reward_logits = self.reward_head(next_state)  # .squeeze(-1)

        if return_logits:
            return next_state, reward_logits
        elif not self.output_probabilities:
            return next_state, reward_logits.squeeze(-1)
        else:
            probs = nn.functional.softmax(reward_logits, dim=-1)
            return next_state, self.inverse_scale_value(
                support_to_scalar(probs, self.support)
            )


class PredictorMLP(nn.Module):

    def __init__(
        self,
        body: nn.Module,
        num_actions: int,
        output_probabilities: bool = False,
        support: torch.Tensor = None,
        scale_value: bool = False,
    ):

        super().__init__()
        self.body = body
        self.num_actions = num_actions
        self.policy_head = nn.Linear(self.body.output_dim, num_actions)
        self.value_head = nn.Linear(
            self.body.output_dim, len(support) if output_probabilities else 1
        )
        self.output_probabilities = output_probabilities
        self.support = support
        self.scale_value = (lambda x: x) if not scale_value else scale_value_function
        self.inverse_scale_value = (
            (lambda y: y) if not scale_value else inverse_scale_value_function
        )

        if self.output_probabilities and self.support is None:
            raise ValueError("support must be provided if output_probabilities is True")

    def forward(
        self, state: torch.Tensor, return_logits: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:

        x = self.body(state)
        logits = self.policy_head(x)
        value_logits = self.value_head(x)

        if return_logits:
            return logits, value_logits
        elif not self.output_probabilities:
            return logits, value_logits.squeeze(-1)
        else:
            probs = nn.functional.softmax(value_logits, dim=-1)
            return logits, self.inverse_scale_value(
                support_to_scalar(probs, self.support)
            )

    def sample_action(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        return action

    def logp(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        logp = nn.functional.log_softmax(logits, dim=-1)
        return logp

    def cross_entropy_loss(
        self,
        logits: torch.Tensor,
        target_probs: torch.Tensor,
    ) -> torch.Tensor:
        logp = self.logp(logits)
        loss = -torch.sum(target_probs * logp, dim=-1)
        return loss


class PredictorMLPCont(PredictorMLP):

    def __init__(
        self,
        body: nn.Module,
        num_actions: int,
        output_probabilities: bool = False,
        support: torch.Tensor = None,
        scale_value: bool = False,
        lims: tuple[float, float] = (-1.0, 1.0),
    ):

        super().__init__(
            body=body,
            num_actions=num_actions,
            output_probabilities=output_probabilities,
            support=support,
            scale_value=scale_value,
        )
        self.policy_head = nn.Linear(self.body.output_dim, num_actions * 2)
        self.lims = lims

    def sample_action(self, logits: torch.Tensor, return_prob=True) -> torch.Tensor:
        mu, log_sigma = torch.chunk(
            torch.tensor(logits, dtype=torch.float32), 2, dim=-1
        )

        sigma = torch.exp(log_sigma).clamp(min=1e-3, max=1.0)

        dist = Normal(mu, sigma)

        # 4. Sample using the reparameterization trick
        z = dist.rsample()
        # action = torch.clamp(action, self.lims[0], self.lims[1])
        action = torch.tanh(z)
        a_scaled = self.lims[0] + (action + 1) * 0.5 * (self.lims[1] - self.lims[0])

        logp = dist.log_prob(z).sum(-1) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)

        if return_prob:
            return a_scaled, torch.exp(logp)
        else:
            return a_scaled

    def logp(self, logits: torch.Tensor, action: torch.Tensor):
        mu, log_std = torch.chunk(logits, 2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        std = log_std.exp()

        # 1. unscale action back to [-1, 1]
        low, high = self.lims
        a = 2 * (action - low) / (high - low) - 1
        a = a.clamp(-1 + 1e-6, 1 - 1e-6)

        # 2. inverse tanh
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh

        # 3. base log-prob under Normal
        dist = Normal(mu, std)
        logp = dist.log_prob(z).sum(dim=-1)

        # 4. tanh correction
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)

        # 5. scale correction
        logp -= torch.log((self.lims[1] - self.lims[0]) / 2).sum()

        return logp

    def cross_entropy_loss(self, logits, actions, pi_target):
        # logits: (B, 2*act_dim)
        B, M, act_dim = actions.shape

        # Expand logits to match actions
        logits_exp = logits.unsqueeze(1).expand(
            B, M, logits.shape[-1]
        )  # (B, M, 2*act_dim)

        # logp for each sampled action
        logp = self.logp(logits_exp, actions)  # (B, M)

        # Cross-entropy with soft targets
        loss = -(pi_target * logp).sum(dim=1)  # (B,)
        return loss


class MuZeroNet(nn.Module):
    def __init__(self, h=None, g=None, f=None, net_config=None):
        super().__init__()
        h_config = net_config.get("h", None) if net_config else None
        g_config = net_config.get("g", None) if net_config else None
        f_config = net_config.get("f", None) if net_config else None
        assert (h is not None) or (
            h_config is not None
        ), "Either h or h_config must be provided"
        assert (g is not None) or (
            g_config is not None
        ), "Either g or g_config must be provided"
        assert (f is not None) or (
            f_config is not None
        ), "Either f or f_config must be provided"

        self.continuous_actions = (
            net_config.get("continuous_actions", False) if net_config else False
        )

        self.h = h if h is not None else self._create_encoder(h_config)
        self.g = g if g is not None else self._create_dynamics(g_config)
        self.f = f if f is not None else self._create_predictor(f_config)
        self.config = net_config

    def _create_encoder(self, config: dict) -> nn.Module:
        body = BodyMLP(**config["body"])
        encoder = EncoderMLP(body=body, **config.get("h", {}))
        return encoder

    def _create_dynamics(self, config: dict) -> nn.Module:
        body = BodyMLP(**config["body"])
        dynamics = DynamicsMLP(body=body, **config.get("g", {}))
        return dynamics

    def _create_predictor(self, config: dict) -> nn.Module:
        body = BodyMLP(**config["body"])
        if self.continuous_actions:
            predictor = PredictorMLPCont(body=body, **config.get("f", {}))
        else:
            predictor = PredictorMLP(body=body, **config.get("f", {}))
        return predictor
