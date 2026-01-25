import numpy as np
import torch
import random
from collections import deque
import numpy as np


def self_play_episode(
    env,
    net,
    mcts,
    temperature: float,
    device="cpu",
    max_steps=10_000,
    mcts_num_simulations=50,
    seed=None,
) -> dict:
    """
    net should expose:
      net.h(obs_tensor)-> latent
      net.f(latent_batch)-> (policy_logits, value)
      net.g(latent_batch, action_batch)-> (latent_next, r_pred)
    mcts.run(root_latent, legal_mask, num_simulations) -> visit_counts, root_value, root_value_raw
    """
    obs, _ = env.reset(seed=seed)
    done = False

    episode = {
        "obs": [],
        "actions": [],
        "rewards": [],  # rewards after action
        "dones": [],
        "pis": [],  # visit dist per step (policy targets)
        "legal_masks": [],
        "root_v_est": [],
    }

    t = 0
    while not done and t < max_steps:
        # legal actions mask (Gym usually has all legal; keep hook for later)
        num_actions = mcts.num_actions
        legal_mask = np.ones(num_actions, dtype=bool)
        # legal_mask = None

        # Encode observation -> latent
        obs_t = (
            torch.as_tensor(obs, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .to(device)
        )  # (1, obs_dim)
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

        # Step env
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        # Store
        episode["obs"].append(obs)
        episode["actions"].append(action)
        episode["rewards"].append(float(reward))
        episode["dones"].append(done)
        episode["pis"].append(pi_target)
        episode["legal_masks"].append(legal_mask.astype(np.float32))
        episode["root_v_est"].append(float(root_v_est))

        obs = next_obs
        t += 1

    return episode


class ReplayBuffer:
    def __init__(self, capacity_episodes: int = 5000):
        self.episodes = deque(maxlen=capacity_episodes)

    def add_episode(self, ep: dict):
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
