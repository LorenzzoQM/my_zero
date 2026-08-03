from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import math
import numpy as np
import torch


@dataclass
class Node:
    prior: float
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


def puct_score(
    q_value: float,
    prior: float,
    parent_visit_count: int,
    child_visit_count: int,
    c_puct: float,
) -> float:
    pb_c = (
        c_puct
        * prior
        * math.sqrt(parent_visit_count + 1e-8)
        / (1 + child_visit_count)
    )
    return q_value + pb_c


def puct_score_batch(
    q_values: np.ndarray,
    child_priors: np.ndarray,
    parent_visit_count: int,
    child_visit_counts: np.ndarray,
    c_puct: float,
) -> np.ndarray:
    pb_c = (
        c_puct
        * child_priors
        * math.sqrt(parent_visit_count + 1e-8)
        / (1 + child_visit_counts)
    )
    return q_values + pb_c


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


class MuZeroMCTS:
    def __init__(
        self,
        prediction_net,  # f
        dynamics_net,  # g
        num_actions: int,
        gamma: float = 0.997,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: str = "cpu",
        puct_opt: str = "puct",
    ):
        self.f = prediction_net
        self.g = dynamics_net
        self.num_actions = num_actions
        self.gamma = gamma
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device
        if puct_opt not in {"puct", "muzero"}:
            raise ValueError(f"Unknown puct_opt={puct_opt!r}")
        self.puct_opt = puct_opt

        self.q_min = np.inf
        self.q_max = -np.inf

    @torch.no_grad()
    def run(
        self,
        root_latent: torch.Tensor,  # (latent_dim,)
        legal_actions: Optional[np.ndarray] = None,  # bool mask shape (num_actions,)
        num_simulations: int = 50,
        add_root_noise: bool = True,
    ):
        # Root node
        root = Node(prior=1.0, reward=0.0, latent=root_latent.detach().to(self.device))

        # Expand root once to initialize priors and root value
        self.q_min = np.inf
        self.q_max = -np.inf
        root_value = self._expand(
            root, legal_actions=legal_actions, add_noise=add_root_noise
        )

        for _ in range(num_simulations):
            node = root
            search_path = [node]
            actions_taken = []

            # 1) Selection: descend by PUCT until reaching an unexpanded node
            while node.expanded():
                parent = node
                if self.num_actions <= 10:
                    action, node = self._select_child(search_path[-1])
                else:
                    action, node = self._select_child_batch(search_path[-1])

                # Lazily materialize child latent via dynamics the first time we traverse it
                if node.latent is None:
                    s_parent = parent.latent.unsqueeze(0)  # (1, latent_dim)
                    a_t = torch.tensor([action], device=self.device, dtype=torch.long)
                    s_next, r_pred = self.g(
                        s_parent, a_t
                    )  # s_next: (1, latent_dim), r_pred: (1,)
                    node.latent = s_next.squeeze(0).detach()
                    node.reward = float(r_pred.squeeze().item())

                search_path.append(node)
                actions_taken.append(action)

            # 2) Expansion + evaluation at leaf
            # actions_taken.append(None)
            leaf_value = self._expand(node, legal_actions=None, add_noise=False)

            # 3) Backpropagate
            self._backpropagate(search_path, leaf_value, actions_taken)

        # Return visit counts (policy target) and root value estimate
        visit_counts = np.zeros(self.num_actions, dtype=np.int32)
        for a, child in root.children.items():
            visit_counts[a] = child.visit_count

        return visit_counts, root.value(), root_value, {}

    def _select_child(self, parent: Node):
        best_action, best_child, best_score = None, None, -1e18
        for action, child in parent.children.items():
            if self.puct_opt == "puct":
                score = puct_score(
                    parent.action_values.get(action, 0.0),
                    child.prior,
                    parent.visit_count,
                    child.visit_count,
                    self.c_puct,
                )
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
        parent_action_values = []
        parent_visit_count = parent.visit_count
        parent_action_counts = []
        child_visit_counts = []
        child_priors = []
        actions = []
        child_list = []
        for action, child in children.items():
            parent_action_values.append(parent.action_values.get(action, 0.0))
            parent_action_counts.append(parent.action_counts.get(action, 0))
            child_visit_counts.append(child.visit_count)
            child_priors.append(child.prior)
            actions.append(action)
            child_list.append(child)

        if self.puct_opt == "puct":
            scores = puct_score_batch(
                np.array(parent_action_values),
                np.array(child_priors),
                parent_visit_count,
                np.array(child_visit_counts),
                self.c_puct,
            )
        else:
            scores = puct_mu_zero_batch(
                np.array(parent_action_values),
                np.array(child_priors),
                parent_visit_count,
                np.array(parent_action_counts),
                self.q_min,
                self.q_max,
            )

        best_index = np.argmax(scores)
        return actions[best_index], child_list[best_index]

    @torch.no_grad()
    def _expand(self, node: Node, legal_actions: Optional[np.ndarray], add_noise: bool):
        """
        Expands node using f(node.latent) to create priors, and sets children.
        If node is not root, its latent is already set by dynamics from parent.
        """
        assert node.latent is not None, "Node latent must be set before expansion."

        s = node.latent.unsqueeze(0)  # (1, latent_dim)
        policy_logits, value = self.f(
            s.to(self.device)
        )  # logits: (1, A), value: (1,) or (1,)

        logits = policy_logits.squeeze(0).float().cpu().numpy()  # (A,)
        value = float(value.squeeze().item())

        # Mask illegal actions at root (common). If you have legality at all nodes, pass it in too.
        if legal_actions is not None:
            illegal = ~legal_actions.astype(bool)
            logits[illegal] = -1e9

        priors = self._softmax(logits)

        # Dirichlet noise only at root
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * self.num_actions)
            priors = (1 - self.dirichlet_eps) * priors + self.dirichlet_eps * noise

        # Create children
        node.children = {}
        for a in range(self.num_actions):
            if legal_actions is not None and not bool(legal_actions[a]):
                continue
            node.children[a] = Node(prior=float(priors[a]))

        # IMPORTANT: in MuZero, children latents are created *lazily* when traversed (via dynamics).
        # We'll do that in backprop step by setting leaf latents when selected.
        # But we still need dynamics latents during selection → simplest is: compute when stepping down.
        # We'll handle that by computing latent when a child is first visited during selection.
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
                node.reward + self.gamma * v
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
        visit_counts: np.ndarray, temperature: float, return_pi: bool
    ) -> Union[int, tuple[int, np.ndarray]]:
        if temperature <= 1e-8:
            action = int(np.argmax(visit_counts))
            pi_target = np.zeros_like(visit_counts, dtype=np.float32)
            pi_target[action] = 1.0
        else:
            pi_target = visit_counts ** (1.0 / temperature)
            pi_target = pi_target / (pi_target.sum() + 1e-12)
            action = int(np.random.choice(len(pi_target), p=pi_target))

        if return_pi:
            return action, pi_target
        return action

    def _compute_entropy(self, visit_counts: np.ndarray) -> float:
        total_visits = np.sum(visit_counts)
        if total_visits == 0:
            return 0.0
        probs = visit_counts / total_visits
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return entropy
