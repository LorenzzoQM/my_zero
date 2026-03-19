from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import math
import numpy as np
import torch
from typing import Any


@dataclass
class Node:
    prior: float
    action: Any
    sigmas: tuple = field(default_factory=tuple)
    reward: float = 0.0  # reward from parent -> this node
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)
    latent: Optional[torch.Tensor] = None  # (latent_dim,) tensor for this node
    action_values: Dict[int, float] = field(
        default_factory=dict
    )  # Q values for each action
    action_counts: Dict[int, int] = field(
        default_factory=dict
    )  # N values for each action

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def puct_score(parent: Node, child: Node, c_puct: float) -> float:
    pb_c = (
        c_puct
        * child.prior
        * math.sqrt(parent.visit_count + 1e-8)
        / (1 + child.visit_count)
    )
    return child.value() + pb_c


def puct_mu_zero(
    Q_sa: float,
    P_sa: float,
    N_s: float,
    N_sa: float,
    Q_min: float,
    Q_max: float,
    c1: float = 1.25,
    c2: float = 19652,
) -> float:

    q_range = Q_max - Q_min
    if math.isfinite(q_range) and math.isfinite(Q_min) and q_range > 1e-6:
        Q_sa = (Q_sa - Q_min) / (Q_max - Q_min + 1e-8)
    else:
        Q_sa = 0.0

    val = Q_sa + P_sa * (math.sqrt(N_s) / (1 + N_sa)) * (
        c1 + math.log((N_s + c2 + 1) / c2)
    )
    return val


def puct_mu_zero_batch(
    Q_sa: np.ndarray,
    P_sa: np.ndarray,
    N_s: float,
    N_sa: np.ndarray,
    Q_min: float,
    Q_max: float,
    c1: float = 1.25,
    c2: float = 19652,
) -> float:

    q_range = Q_max - Q_min
    if math.isfinite(q_range) and math.isfinite(Q_min) and q_range > 1e-6:
        Q_sa = (Q_sa - Q_min) / (q_range + 1e-8)
    else:
        Q_sa = np.zeros_like(Q_sa)

    val = Q_sa + P_sa * (math.sqrt(N_s) / (1 + N_sa)) * (
        c1 + math.log((N_s + c2 + 1) / c2)
    )
    return val


class MuZeroSampledMCTS:
    def __init__(
        self,
        prediction_net,  # f
        dynamics_net,  # g
        num_actions: int,
        num_sampled_actions: int,
        discount: float = 0.997,
        c_puct: float = 1.5,
        beta_temp: float = 1.5,
        device: str = "cpu",
        puct_opt: str = "puct",
        num_sampled_actions_root: int | None = None,
        sample_from_uniform: int = 4,
    ):
        self.f = prediction_net
        self.g = dynamics_net
        self.num_actions = num_actions
        self.num_sampled_actions = num_sampled_actions
        self.discount = discount
        self.c_puct = c_puct
        self.device = device
        self.beta_temp = beta_temp
        self.puct_opt = puct_opt
        self.sample_from_uniform = sample_from_uniform
        self.batched_search = True
        assert (
            self.sample_from_uniform < self.num_sampled_actions
        ), "sample_from_uniform must be less than num_sampled_actions"
        assert self.sample_from_uniform >= 0, "sample_from_uniform must be non-negative"

        self.q_min = np.inf
        self.q_max = -np.inf

        if num_sampled_actions_root is None:
            self.num_sampled_actions_root = num_sampled_actions * 2
        else:
            self.num_sampled_actions_root = num_sampled_actions_root

    @torch.no_grad()
    def run(
        self,
        root_latent: torch.Tensor,  # (latent_dim,)
        legal_actions: Optional[np.ndarray] = None,  # bool mask shape (num_actions,)
        num_simulations: int = 50,
        add_root_noise: bool = True,
    ):
        # Root node
        root = Node(
            prior=1.0,
            reward=0.0,
            latent=root_latent.detach().to(self.device),
            action=None,
        )

        # Expand root once to initialize priors and root value
        self.q_min = np.inf
        self.q_max = -np.inf
        root_value = self._expand(
            root,
            legal_actions=legal_actions,
            add_noise=add_root_noise,
            beta_temp=self.beta_temp,
            num_sampled_actions_node=self.num_sampled_actions_root,
        )

        for _ in range(num_simulations):
            node = root
            search_path = [node]
            actions_taken = []

            # 1) Selection: descend by PUCT until reaching an unexpanded node
            while node.expanded():
                parent = node
                if self.batched_search:
                    action, node = self._select_child_batch(search_path[-1])
                else:
                    action, node = self._select_child(search_path[-1])

                # Lazily materialize child latent via dynamics the first time we traverse it
                if node.latent is None:
                    s_parent = parent.latent.unsqueeze(0)  # (1, latent_dim)
                    a_t = torch.tensor(
                        [node.action], device=self.device, dtype=torch.float32
                    )
                    s_next, r_pred = self.g(
                        s_parent, a_t
                    )  # s_next: (1, latent_dim), r_pred: (1,)
                    node.latent = s_next.squeeze(0).detach()
                    node.reward = float(r_pred.squeeze().item())

                search_path.append(node)
                actions_taken.append(action)

            # 2) Expansion + evaluation at leaf
            # actions_taken.append(None)
            leaf_value = self._expand(
                node, legal_actions=None, add_noise=False, beta_temp=self.beta_temp
            )

            # 3) Backpropagate
            self._backpropagate(search_path, leaf_value, actions_taken)

        # Return visit counts (policy target) and root value estimate
        visit_counts = np.zeros(self.num_sampled_actions_root, dtype=np.int32)
        for a, child in root.children.items():
            visit_counts[a] = child.visit_count

        return (
            {
                "visit_counts": visit_counts,
                "actions": [
                    child.action.squeeze(0).cpu().numpy()
                    for child in root.children.values()
                ],
            },
            root.value(),
            root_value,
            {
                "sigmas": [root.sigmas[0] for child in root.children.values()],
                "log_sigmas": [root.sigmas[1] for child in root.children.values()],
            },
        )

    # Slow for larger action spaces
    def _select_child(self, parent: Node):
        best_action, best_child, best_score = None, None, -1e18
        for action, child in parent.children.items():
            if self.puct_opt == "puct":
                score = puct_score(parent, child, self.c_puct)
            else:
                score = puct_mu_zero(
                    parent.action_values.get(action, 0.0),
                    child.prior,
                    parent.visit_count,
                    parent.action_counts.get(action, 0),
                    self.q_min,
                    self.q_max,
                )

            if score > best_score:
                best_score = score
                best_action, best_child = action, child
        return best_action, best_child

    def _select_child_batch(self, parent: Node):

        children = parent.children
        actions = list(children)

        av = parent.action_values
        ac = parent.action_counts

        Q_sa = np.fromiter(
            (av.get(a, 0.0) for a in actions), dtype=np.float32, count=len(actions)
        )
        N_sa = np.fromiter(
            (ac.get(a, 0) for a in actions), dtype=np.float32, count=len(actions)
        )
        P_sa = np.fromiter(
            (children[a].prior for a in actions), dtype=np.float32, count=len(actions)
        )

        scores = puct_mu_zero_batch(
            Q_sa,
            P_sa,
            parent.visit_count,
            N_sa,
            self.q_min,
            self.q_max,
        )

        best_idx = int(np.argmax(scores))
        best_action = actions[best_idx]
        return best_action, children[best_action]

    @torch.no_grad()
    def _expand(
        self,
        node: Node,
        legal_actions: Optional[np.ndarray],
        add_noise: bool,
        beta_temp: float,
        num_sampled_actions_node: Optional[int] = None,
    ):
        """
        Expands node using f(node.latent) to create priors, and sets children.
        If node is not root, its latent is already set by dynamics from parent.
        """
        assert node.latent is not None, "Node latent must be set before expansion."

        s = node.latent.unsqueeze(0)  # (1, latent_dim)

        policy_logits, value = self.f(
            s.to(self.device)
        )  # logits: (1, A), value: (1,) or (1,)

        value = float(value.squeeze().item())

        # Create children
        if num_sampled_actions_node is not None:
            num_sampled_actions = num_sampled_actions_node
        else:
            num_sampled_actions = self.num_sampled_actions

        n_from_policy = num_sampled_actions - self.sample_from_uniform
        n_from_uniform = num_sampled_actions - n_from_policy

        actions_list, priors_beta, sigmas = self.f.sample_action(
            policy_logits, n_samples=n_from_policy, return_prob=True
        )

        if n_from_uniform > 0:
            lims = getattr(self.f, "lims", (-1.0, 1.0))
            low = torch.as_tensor(lims[0], device=self.device, dtype=torch.float32)
            high = torch.as_tensor(lims[1], device=self.device, dtype=torch.float32)
            if low.dim() == 0:
                low = low.expand(self.num_actions)
            if high.dim() == 0:
                high = high.expand(self.num_actions)
            uniform_dist = torch.distributions.Uniform(low=low, high=high)
            uniform_actions = uniform_dist.sample((n_from_uniform, 1))
            uniform_priors = (
                torch.ones((n_from_uniform, 1), device=self.device) / n_from_uniform
            )
            actions_list = torch.cat([actions_list, uniform_actions], dim=0)
            priors_beta = torch.cat([priors_beta, uniform_priors], dim=0)

        node.sigmas = sigmas

        node.children = {}
        priors_beta = np.array(priors_beta)
        priors_beta = priors_beta ** (1.0 / beta_temp)
        priors_beta = priors_beta / (priors_beta.sum() + 1e-8)

        for i in range(num_sampled_actions):
            node.children[i] = Node(prior=float(priors_beta[i]), action=actions_list[i])
        return value

    def _backpropagate(self, search_path, leaf_value: float, actions_taken):
        """
        search_path: [root, n1, n2, ..., leaf]
        actions_taken: [a0, a1, ..., a_{L-1}] where a_i is action from search_path[i] -> search_path[i+1]
        """
        v = leaf_value

        # Update leaf node stats first, then walk upward updating parent edge stats
        for i in reversed(range(len(search_path))):
            node = search_path[i]
            node.value_sum += v
            node.visit_count += 1

            if i == 0:
                break  # reached root

            parent = search_path[i - 1]
            action = actions_taken[i - 1]  # action taken at parent to reach node

            q = (
                node.reward + self.discount * v
            )  # reward is stored on child node (edge parent->child)

            # # Update edge stats on the parent
            if action not in parent.action_values:
                parent.action_values[action] = 0.0
                parent.action_counts[action] = 0

            n = parent.action_counts.get(action, 0)
            parent.action_values[action] += (q - parent.action_values[action]) / (n + 1)
            parent.action_counts[action] = n + 1

            new_q = parent.action_values[action]
            self.q_min = min(self.q_min, new_q)
            self.q_max = max(self.q_max, new_q)

            v = q

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + 1e-8)

    @staticmethod
    def select_action_from_visits(
        visit_counts_dict: dict, temperature: float, return_pi: bool
    ) -> Union[int, tuple[int, np.ndarray]]:

        visit_counts = visit_counts_dict["visit_counts"]
        actions = visit_counts_dict["actions"]

        if temperature <= 1e-8:
            action = int(np.argmax(visit_counts))
            pi_target = np.zeros_like(visit_counts, dtype=np.float32)
            pi_target[action] = 1.0
        else:
            pi_target = visit_counts ** (1.0 / temperature)
            pi_target = pi_target / (pi_target.sum() + 1e-12)
            action = int(np.random.choice(len(pi_target), p=pi_target))

        if return_pi:
            return actions[action], pi_target
        return actions[action]

    def _compute_entropy(self, visit_counts_dict: dict) -> float:
        visit_counts = visit_counts_dict["visit_counts"]
        total_visits = np.sum(visit_counts)
        if total_visits == 0:
            return 0.0
        probs = visit_counts / total_visits
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return entropy
