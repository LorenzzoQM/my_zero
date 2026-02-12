import numpy as np
import torch
import random
from collections import deque
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Union


@dataclass
class Episode:
    obs: Union[np.ndarray, List[Any], dict[str, Union[np.ndarray, List[Any]]]]
    actions: Union[np.ndarray, List[Any]]
    actions_sampled: Optional[
        Union[np.ndarray, List[Any], dict[str, Union[np.ndarray, List[Any]]]]
    ] = None
    rewards: Union[np.ndarray, List[Any], dict[str, Union[np.ndarray, List[Any]]]]
    terminated: Union[np.ndarray, List[Any]]
    truncated: Union[np.ndarray, List[Any]]
    pis: Union[np.ndarray, List[Any]]
    root_v_est: Union[np.ndarray, List[Any]]
    legal_masks: Union[np.ndarray, List[Any]]

    priority: float = 1.0

    def __getitem__(self, key):
        return getattr(self, key)


def self_play_episode(
    env,
    env_callback,
    net,
    mcts,
    temperature: float,
    device="cpu",
    max_steps=10_000,
    mcts_num_simulations=50,
    seed=None,
    agents_embedding: Union[None, dict[str, np.ndarray]] = None,
) -> Tuple[Episode, Optional[Any]]:
    """
    net should expose:
      net.h(obs_tensor)-> latent
      net.f(latent_batch)-> (policy_logits, value)
      net.g(latent_batch, action_batch)-> (latent_next, r_pred)
    mcts.run(root_latent, legal_mask, num_simulations) -> visit_counts, root_value, root_value_raw
    """
    obs, _ = env.reset(seed=seed)
    if agents_embedding is not None:
        _multi_agent = True
    else:
        _multi_agent = False
    done = False

    if not _multi_agent:
        episode = Episode(
            obs=[],
            actions=[],
            actions_sampled=[],
            rewards=[],
            terminated=[],
            truncated=[],
            pis=[],
            legal_masks=[],
            root_v_est=[],
            priority=1.0,  # for prioritized replay
        )

    else:
        agents = list(agents_embedding.keys())
        episode_dict = {
            agent: Episode(
                obs=[],
                actions=[],
                actions_sampled=[],
                rewards=[],
                terminated=[],
                truncated=[],
                pis=[],
                legal_masks=[],
                root_v_est=[],
                priority=1.0,  # for prioritized replay
            )
            for agent in agents
        }

    t = 0
    while not done and t < max_steps:
        # legal actions mask (Gym usually has all legal; keep hook for later)
        num_actions = mcts.num_actions
        legal_mask = np.ones(num_actions, dtype=bool)
        # legal_mask = None

        # Encode observation -> latent
        if not _multi_agent:
            obs_t = (
                torch.as_tensor(obs, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .to(device)
            )  # (1, obs_dim)
            actions_sampled = None
            with torch.no_grad():
                root_latent = net.h(obs_t).squeeze(0)  # (latent_dim,)

                # MCTS
                visit_counts, root_v_est, _ = mcts.run(
                    root_latent=root_latent,
                    legal_actions=legal_mask,
                    num_simulations=mcts_num_simulations,
                    add_root_noise=True,
                )

            action, pi_target = mcts.select_action_from_visits(
                visit_counts, temperature=temperature, return_pi=True
            )
            if isinstance(visit_counts, dict):
                actions_sampled = visit_counts["actions"]
        else:
            agents_list = list(obs.keys())
            obs_agents = {}
            for agent in agents_list:
                obs_a = np.concatenate([obs[agent], agents_embedding[agent]], axis=-1)
                obs_agents[agent] = torch.as_tensor(
                    obs_a, dtype=torch.float32, device=device
                ).unsqueeze(0)

            action = {}
            pi_target = {}
            root_v_est = {}
            actions_sampled_dict = {}
            for agent in agents_list:
                with torch.no_grad():
                    root_latent = net.h(obs_agents[agent]).squeeze(0)  # (latent_dim,)

                    # MCTS
                    visit_counts_a, root_v_est_a, _ = mcts.run(
                        root_latent=root_latent,
                        legal_actions=legal_mask,
                        num_simulations=mcts_num_simulations,
                        add_root_noise=True,
                    )

                action_a, pi_target_a = mcts.select_action_from_visits(
                    visit_counts_a, temperature=temperature, return_pi=True
                )
                action[agent] = action_a
                pi_target[agent] = pi_target_a
                root_v_est[agent] = root_v_est_a
                if isinstance(visit_counts_a, dict):
                    actions_sampled_dict[agent] = visit_counts_a["actions"]

        # Step env
        if _multi_agent:
            # Prevents modification of the action dict in place
            action_copy = action.copy()
        else:
            action_copy = action
        next_obs, reward, terminated, truncated, _ = env.step(action_copy)
        if _multi_agent:
            done = any(terminated.values()) or any(truncated.values())
        else:
            done = bool(terminated or truncated)

        if _multi_agent:
            for agent in agents_list:
                episode_dict[agent]["obs"].append(
                    obs_agents[agent].squeeze().numpy().copy()
                )
                episode_dict[agent]["actions"].append(action[agent])
                episode_dict[agent]["rewards"].append(float(reward[agent]))
                episode_dict[agent]["terminated"].append(any(terminated.values()))
                episode_dict[agent]["truncated"].append(any(truncated.values()))
                episode_dict[agent]["pis"].append(pi_target[agent].copy())
                episode_dict[agent]["root_v_est"].append(float(root_v_est[agent]))
                episode_dict[agent]["legal_masks"].append(legal_mask.astype(np.float32))
                if len(actions_sampled_dict.keys()) > 0:
                    episode_dict[agent]["actions_sampled"].append(
                        actions_sampled_dict[agent]
                    )
        else:
            episode["obs"].append(obs)
            episode["actions"].append(action)
            if actions_sampled is not None:
                episode["actions_sampled"].append(actions_sampled)
            episode["rewards"].append(float(reward))
            episode["terminated"].append(terminated)
            episode["truncated"].append(truncated)
            episode["pis"].append(pi_target)
            episode["legal_masks"].append(legal_mask.astype(np.float32))
            episode["root_v_est"].append(float(root_v_est))

        obs = next_obs
        t += 1

        if (_multi_agent and any(truncated.values())) or truncated:
            if not _multi_agent:
                obs_t = (
                    torch.as_tensor(obs, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.no_grad():
                    root_latent = net.h(obs_t).squeeze(0)  # (latent_dim,)

                    _, root_v_est, _ = mcts.run(
                        root_latent=root_latent,
                        legal_actions=legal_mask,
                        num_simulations=mcts_num_simulations,
                        add_root_noise=True,
                    )
                episode["root_v_est"].append(float(root_v_est))
            else:
                agents_list = list(obs.keys())
                obs_agents = {}
                for agent in agents_list:
                    obs_a = np.concatenate(
                        [obs[agent], agents_embedding[agent]], axis=-1
                    )
                    obs_agents[agent] = torch.as_tensor(
                        obs_a, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                for agent in agents_list:
                    with torch.no_grad():
                        root_latent = net.h(obs_agents[agent]).squeeze(
                            0
                        )  # (latent_dim,)

                        _, root_v_est_a, _ = mcts.run(
                            root_latent=root_latent,
                            legal_actions=legal_mask,
                            num_simulations=mcts_num_simulations,
                            add_root_noise=True,
                        )
                    episode_dict[agent]["root_v_est"].append(float(root_v_est_a))

    if env_callback is not None:
        env_callback_data = env_callback(env)
    else:
        env_callback_data = {}

    if _multi_agent:
        return episode_dict, env_callback_data
    else:
        return episode, env_callback_data


class ReplayBuffer:
    def __init__(self, capacity_episodes: int = 5000):
        self.episodes = deque(maxlen=capacity_episodes)

    def add_episode(self, ep: Episode):
        self.episodes.append(ep)

    def __len__(self):
        return sum(len(ep["actions"]) for ep in self.episodes)

    def sample_positions(self, batch_size: int):
        batch = []
        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            t = random.randrange(len(ep["actions"]))
            batch.append((ep, t))
        return batch


class PrioritizedReplayBuffer:
    """
    Episode-level prioritized replay:
      - sample episodes proportional to priority^alpha
      - sample positions uniformly within chosen episode
      - importance weights based on episode sampling prob
    """

    def __init__(self, capacity_episodes: int, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = int(capacity_episodes)
        self.alpha = float(alpha)
        self.eps = float(eps)

        self.episodes: List[Episode] = []
        self._pos = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(self, ep: Episode) -> None:
        # Initialize new episodes with max priority so they get sampled at least once.
        ep.priority = self._max_priority

        if len(self.episodes) < self.capacity:
            self.episodes.append(ep)
        else:
            self.episodes[self._pos] = ep
            self._pos = (self._pos + 1) % self.capacity

    def _episode_probs(self) -> np.ndarray:
        prios = np.array(
            [max(e.priority, self.eps) for e in self.episodes], dtype=np.float64
        )
        scaled = prios**self.alpha
        scaled_sum = scaled.sum()
        if scaled_sum <= 0:
            # fallback uniform
            return np.ones_like(scaled) / len(scaled)
        return scaled / scaled_sum

    def sample_positions(
        self,
        batch_size: int,
        beta: float = 0.4,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
        """
        Returns:
          - samples: list of (episode_index, position_index)
          - is_weights: (batch_size,) importance sampling weights
          - ep_indices: (batch_size,) episode indices (for priority updates)
        """
        assert len(self.episodes) > 0, "Replay is empty."
        rng = rng or np.random.default_rng()

        probs = self._episode_probs()
        ep_indices = rng.choice(
            len(self.episodes), size=batch_size, replace=True, p=probs
        )

        samples: List[Tuple[int, int]] = []
        for epi in ep_indices:
            ep = self.episodes[int(epi)]
            T = len(ep.actions)  # number of transitions
            # sample a training position. You might want to avoid the last K steps depending on your unroll.
            pos = int(rng.integers(0, T))
            samples.append((ep, pos))

        # Importance sampling weights (episode-level)
        # w_i = (N * P(i))^{-beta}, normalized by max weight
        N = len(self.episodes)
        p_i = probs[ep_indices]
        w = (N * p_i) ** (-beta)
        w /= w.max() + 1e-12
        w = w.astype(np.float32)

        return samples, w, ep_indices.astype(np.int64)

    def update_priorities(
        self, ep_indices: np.ndarray, new_priorities: np.ndarray
    ) -> None:
        """
        Update episode priorities. A good choice is max(|td_error|) or mean over sampled positions.
        """
        ep_indices = np.asarray(ep_indices, dtype=np.int64)
        new_priorities = np.asarray(new_priorities, dtype=np.float64)

        for epi, p in zip(ep_indices, new_priorities):
            p = float(abs(p) + self.eps)
            self.episodes[int(epi)].priority = p
            if p > self._max_priority:
                self._max_priority = p
