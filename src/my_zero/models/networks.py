import torch
import torch.nn as nn
from typing import Union

def scale_value(x, eps=0.001):
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x

def inverse_scale_value(y, eps=0.001):
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

    def __init__(self, width: int, block_depth: int, blocks: int, input_dim: int, output_dim: int, skip_connection: bool=False):
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

    def __init__(self, body: nn.Module, num_actions: int, action_embed_dim: int, reward_head: nn.Module | None = None, normalize_latent: str | None = "l2", output_probabilities: bool = False, support: torch.Tensor = None):

        super().__init__()
        self.latent_dim = body.output_dim
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim
        self.body = body
        self.reward_head = reward_head or nn.Linear(self.latent_dim, len(support) if output_probabilities else 1)
        self.normalize_latent = normalize_latent 
        self.action_embed = nn.Embedding(num_actions, action_embed_dim)
        self.output_probabilities = output_probabilities
        self.support = support

        if self.output_probabilities and self.support is None:
            raise ValueError("support must be provided if output_probabilities is True")

    def forward(self, state: torch.Tensor, action: torch.Tensor, return_logits: bool = False) -> tuple[torch.Tensor, torch.Tensor]:

        action_embed = self.action_embed(action) 
        x = torch.cat([state, action_embed], dim=-1)
        next_state = self.body(x)
        if self.normalize_latent:
            next_state = next_state / (next_state.norm(dim=-1, keepdim=True) + 1e-8)
        
        reward_logits = self.reward_head(next_state) #.squeeze(-1)


        if return_logits:
            return next_state, reward_logits
        elif not self.output_probabilities:
            return next_state, reward_logits.squeeze(-1)
        else:
            probs = nn.functional.softmax(reward_logits, dim=-1)
            return next_state, inverse_scale_value(support_to_scalar(probs, self.support))
    

class PredictorMLP(nn.Module):

    def __init__(self, body: nn.Module, num_actions: int, output_probabilities: bool = False, support: torch.Tensor = None):

        super().__init__()
        self.body = body
        self.num_actions = num_actions
        self.policy_head = nn.Linear(self.body.output_dim, num_actions)
        self.value_head = nn.Linear(self.body.output_dim, len(support) if output_probabilities else 1)
        self.output_probabilities = output_probabilities
        self.support = support

        if self.output_probabilities and self.support is None:
            raise ValueError("support must be provided if output_probabilities is True")

    def forward(self, state: torch.Tensor, return_logits: bool = False) -> tuple[torch.Tensor, torch.Tensor]:

        x = self.body(state)
        logits = self.policy_head(x)
        value_logits = self.value_head(x)

        if return_logits:
            return logits, value_logits
        elif not self.output_probabilities:
            return logits, value_logits.squeeze(-1)
        else:
            probs = nn.functional.softmax(value_logits, dim=-1)
            return logits, inverse_scale_value(support_to_scalar(probs, self.support))
    

class MuZeroNet(nn.Module):
    def __init__(self, h=None, g=None, f=None, net_config=None):
        super().__init__()
        h_config = net_config.get("h", None) if net_config else None
        g_config = net_config.get("g", None) if net_config else None
        f_config = net_config.get("f", None) if net_config else None
        assert (h is not None) or (h_config is not None), "Either h or h_config must be provided"
        assert (g is not None) or (g_config is not None), "Either g or g_config must be provided"
        assert (f is not None) or (f_config is not None), "Either f or f_config must be provided"
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
        dynamics = DynamicsMLP(
            body=body, **config.get("g", {})
        )
        return dynamics
    
    def _create_predictor(self, config: dict) -> nn.Module:
        body = BodyMLP(**config["body"])
        predictor = PredictorMLP(
            body=body, **config.get("f", {})
        )
        return predictor