from my_zero.train.data import self_play_episode, ReplayBuffer, PrioritizedReplayBuffer
from my_zero.models.networks import scalar_to_support, MuZeroNet
from my_zero.MCTS.tree_search import MuZeroMCTS
from my_zero.train.workers import _init_self_play_worker, _worker_self_play_one_episode
import numpy as np
import torch
import torch.nn.functional as F
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import gymnasium as gym
import pathlib
import glob
import os
import json


def save_checkpoint(path, net, optimizer, config, iteration):
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def load_checkpoint(path, net, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    net.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    iteration = checkpoint.get("iteration", 0)
    config = checkpoint.get("config", None)

    return iteration, config


def _worker_self_play(args):
    (
        env_id,
        env_args,
        net_config,
        net_state,
        temperature,
        mcts_num_simulations,
        mcts_config,
        seed,
    ) = args

    # Create env inside worker
    if isinstance(env_id, str):
        env = gym.make(env_id, **env_args)
    else:
        env = env_id(**env_args)

    # Rebuild net + mcts inside worker (you implement these builders)
    net = MuZeroNet(net_config=net_config)
    net = torch.compile(net)
    net.load_state_dict(net_state)
    net.eval()

    mcts = MuZeroMCTS(net.f, net.g, num_actions=env.action_space.n, **mcts_config)

    ep = self_play_episode(
        env=env,
        net=net,
        mcts=mcts,
        temperature=temperature,
        device="cpu",  # usually best for many workers
        mcts_num_simulations=mcts_num_simulations,
        seed=seed,
    )

    env.close()
    return ep


def make_targets(ep: dict, t0: int, K: int, n_step: int, gamma: float):
    T = len(ep["actions"])

    target_pis = []
    target_rs = []
    target_vs = []

    for k in range(K + 1):
        t = t0 + k

        # Policy target (from MCTS at that time)
        if t < T:
            pi = ep["pis"][t]
        else:
            pi = np.ones_like(ep["pis"][0], dtype=np.float32) / len(ep["pis"][0])
        target_pis.append(pi)

        # Reward target corresponds to reward after action at time t
        if t < T:
            r = ep["rewards"][t]
        else:
            r = 0.0
        target_rs.append(r)

        # Value target: n-step return from t
        v = 0.0
        for i in range(n_step):
            ti = t + i
            if ti < T:
                v += (gamma**i) * ep["rewards"][ti]
                if ep["dones"][ti]:
                    break
                if i == n_step - 1:
                    if ti + 1 < T:
                        v += (gamma ** (i + 1)) * ep["root_v_est"][ti + 1]
                    break

        target_vs.append(v)

    return (
        np.array(target_pis),
        np.array(target_rs, dtype=np.float32),
        np.array(target_vs, dtype=np.float32),
    )


def train_step(
    net,
    optimizer,
    batch,
    K=5,
    n_step=5,
    gamma=0.997,
    device="cpu",
    w_policy=1.0,
    w_value=1.0,
    w_reward=1.0,
    is_weights=None,
    return_priorities=False,
):
    """
    batch: list of (episode, t0)
    """
    # Build tensors
    (
        obs0_list,
        action_seqs,
        target_pis_list,
        target_rs_list,
        target_vs_list,
        legal_masks_list,
    ) = ([], [], [], [], [], [])

    if is_weights is None:
        w = None
    else:
        w = torch.as_tensor(is_weights, dtype=torch.float32, device=device)  # (B,)

    for ep, t0 in batch:
        obs0_list.append(ep["obs"][t0])

        # actions for unroll (length K)
        acts = []
        for k in range(K):
            t = t0 + k
            acts.append(ep["actions"][t] if t < len(ep["actions"]) else 0)
        action_seqs.append(acts)

        pis, rs, vs = make_targets(ep, t0, K=K, n_step=n_step, gamma=gamma)
        target_pis_list.append(pis)  # (K+1, A)
        target_rs_list.append(rs)  # (K+1,)
        target_vs_list.append(vs)  # (K+1,)

        # optional masks aligned with policy outputs
        masks = []
        for k in range(K + 1):
            t = t0 + k
            if t < len(ep["legal_masks"]):
                masks.append(ep["legal_masks"][t])
            else:
                masks.append(ep["legal_masks"][0])
        legal_masks_list.append(masks)

    obs0 = torch.as_tensor(
        np.array(obs0_list), dtype=torch.float32, device=device
    )  # (B, obs_dim)
    actions = torch.as_tensor(
        np.array(action_seqs), dtype=torch.long, device=device
    )  # (B, K)
    target_pis = torch.as_tensor(
        np.array(target_pis_list), dtype=torch.float32, device=device
    )  # (B, K+1, A)
    target_rs = torch.as_tensor(
        np.array(target_rs_list), dtype=torch.float32, device=device
    )  # (B, K+1)
    target_vs = torch.as_tensor(
        np.array(target_vs_list), dtype=torch.float32, device=device
    )  # (B, K+1)
    legal_masks = torch.as_tensor(
        np.array(legal_masks_list), dtype=torch.float32, device=device
    )  # (B, K+1, A)

    # Forward unroll
    s = net.h(obs0)  # (B, latent_dim)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_reward_loss = 0.0

    return_logits_v = net.f.output_probabilities
    return_logits_r = net.g.output_probabilities

    for k in range(K + 1):
        # logits, v_pred = net.f(s, action_mask=legal_masks[:, k, :])
        logits, v_pred = net.f(s, return_logits=return_logits_v)
        # Policy loss: cross-entropy with target visit distribution
        logp = F.log_softmax(logits, dim=-1)
        policy_loss = -(target_pis[:, k, :] * logp).sum(dim=-1)

        # Value loss: MSE
        # value_loss = F.mse_loss(v_pred, target_vs[:, k])
        if return_logits_v:
            target_dist = scalar_to_support(
                net.f.scale_value(target_vs[:, k]), net.f.support
            )
            value_loss = -(target_dist * F.log_softmax(v_pred, dim=-1)).sum(dim=-1)
        else:
            value_loss = F.mse_loss(v_pred, target_vs[:, k])

        total_policy_loss = total_policy_loss + policy_loss / (K + 1)
        total_value_loss = total_value_loss + value_loss / (K + 1)

        # Dynamics + reward loss for k < K (reward predicted for transition at step k)
        if k < K:
            s_next, r_pred = net.g(s, actions[:, k], return_logits=return_logits_r)
            # reward_loss = F.mse_loss(r_pred, target_rs[:, k])
            if return_logits_r:
                target_dist_r = scalar_to_support(
                    net.g.scale_value(target_rs[:, k]), net.g.support
                )
                reward_loss = -(target_dist_r * F.log_softmax(r_pred, dim=-1)).sum(
                    dim=-1
                )
            else:
                reward_loss = F.mse_loss(r_pred, target_rs[:, k])
            total_reward_loss = total_reward_loss + reward_loss / K
            s = s_next

    loss_unweighted = (
        w_policy * total_policy_loss
        + w_value * total_value_loss
        + w_reward * total_reward_loss
    )

    if w is None:
        loss = loss_unweighted.mean()
    else:
        loss = (loss_unweighted * w).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    optimizer.step()

    stats = {
        "loss": float(loss.item()),
        "loss_unweighted": float(loss_unweighted.mean().detach().cpu().item()),
        "policy_loss": float(total_policy_loss.mean().detach().cpu().item()),
        "value_loss": float(total_value_loss.mean().detach().cpu().item()),
        "reward_loss": float(total_reward_loss.mean().detach().cpu().item()),
    }

    if return_priorities:
        # A good priority signal: per-sample total loss, or just value error / value CE
        priorities = loss_unweighted.detach().abs().cpu().numpy()  # (B,)
        stats["priorities"] = priorities

    return stats


class Trainer:

    def __init__(
        self,
        env,
        net,
        device="cpu",
        config={},
        net_config=None,
        env_args={},
        eval_env_args=None,
    ):
        self.env = env
        self.net = net
        self.device = device
        self.net_config = net_config
        self._set_config(config)
        self._seed = 1234
        self.env_args = env_args
        self.eval_env_args = eval_env_args if eval_env_args is not None else env_args

        self.net = torch.compile(self.net).to(self.device)

    def _set_config(self, config):
        self.SGD_steps_per_iteration = config.get("SGD_steps_per_iteration", 32)
        self.self_play_episodes_per_iteration = config.get(
            "self_play_episodes_per_iteration", 4
        )
        self.replay_capacity_episodes = config.get("replay_capacity_episodes", 5000)
        self.min_replay_episodes_for_training = config.get(
            "min_replay_episodes_for_training", 10
        )
        self.mcts_num_simulations = config.get("mcts_num_simulations", 50)
        self.mcts_config_self_play = config.get(
            "mcts_config_self_play", {"puct_opt": "muzero", "c_puct": 1.0}
        )
        self.batch_size = config.get("batch_size", 64)
        self.K = config.get("K", 5)
        self.n_step = config.get("n_step", 5)
        self.gamma = config.get("gamma", 0.997)
        self.w_policy = config.get("w_policy", 1.0)
        self.w_value = config.get("w_value", 1.0)
        self.w_reward = config.get("w_reward", 1.0)
        self.n_iterations = config.get("n_iterations", 1000)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.num_workers = config.get("num_workers", 1)

        self.temperature_scheduler_function = config.get(
            "temperature_function", lambda it: 1.0
        )
        self.c_puct_scheduler_function = config.get("c_puct_function", lambda it: 1.0)
        self.dirichlet_eps_scheduler_function = config.get(
            "dirichlet_eps_function", lambda it: 0.25
        )

        self.eval_frequency = config.get("eval_frequency", 10)
        self.checkpoint_frquency = config.get("checkpoint_frequency", 20)
        self.checkpoint_path = config.get("checkpoint_path", None)
        self.log_path = config.get("log_path", pathlib.Path("./logs"))
        self.log_name = None

        self.lr_scheduler = config.get("lr_scheduler", None)
        self.lr_scheduler_params = config.get("lr_scheduler_params", {})

        self.net_class = config.get("net_class", MuZeroNet)
        self.mcts_class = config.get("mcts_class", MuZeroMCTS)

        self.prioritized_replay = config.get("prioritized_replay", False)
        self.per_beta0 = config.get("per_beta0", 0.4)
        self.per_beta1 = config.get("per_beta1", 1.0)

    def _get_self_play_pool(self):
        if getattr(self, "_self_play_pool", None) is None:
            self._self_play_pool = ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_init_self_play_worker,
                initargs=(
                    self.env,
                    self.env_args,
                    self.net_class,
                    self.mcts_class,
                    self.net_config,
                    self.device,
                ),
            )
        return self._self_play_pool

    def run_self_play_executor(self, replay, it):
        base = self.net._orig_mod if hasattr(self.net, "_orig_mod") else self.net
        net_state = {k: v.detach().cpu() for k, v in base.state_dict().items()}

        self.mcts_config_self_play["dirichlet_eps"] = (
            self.dirichlet_eps_scheduler_function(it)
        )
        self.mcts_config_self_play["c_puct"] = self.c_puct_scheduler_function(it)

        temp = self.temperature_scheduler_function(it)

        pool = self._get_self_play_pool()

        futures = []
        for i in range(self.self_play_episodes_per_iteration):
            futures.append(
                pool.submit(
                    _worker_self_play_one_episode,
                    net_state,  # CPU weights dict
                    temp,
                    self.mcts_class,
                    self.mcts_num_simulations,
                    self.mcts_config_self_play,
                    self._seed + i,
                )
            )
        self._seed += self.self_play_episodes_per_iteration

        avg_length = 0.0
        avg_reward = 0.0
        for fut in as_completed(futures):
            ep = fut.result()
            replay.add_episode(ep)
            avg_length += len(ep["rewards"])
            avg_reward += sum(ep["rewards"])

        avg_length /= self.self_play_episodes_per_iteration
        avg_reward /= self.self_play_episodes_per_iteration

        return (avg_length, avg_reward)

    @staticmethod
    def _clean_config_dict(config):
        clean_dict = {}
        for key, value in config.items():
            if isinstance(value, (torch.Tensor)):
                clean_dict[key] = value.tolist()
            elif isinstance(value, dict):
                clean_dict[key] = Trainer._clean_config_dict(value)
            else:
                clean_dict[key] = value
        return clean_dict

    def _start_log(self):
        os.makedirs(self.log_path, exist_ok=True)
        n_logs = glob.glob(str(self.log_path / "training_log_*.jsonl"))
        log_idx = len(n_logs)
        self.log_name = self.log_path / f"training_log_{log_idx}.jsonl"
        log_dict = {"time_start": time.time(), "net_config": self.net_config}

        log_dict_clean = self._clean_config_dict(log_dict)

        with open(self.log_name, "w") as f:
            f.write(json.dumps(log_dict_clean) + "\n")

        if self.checkpoint_path is not None:
            os.makedirs(self.checkpoint_path, exist_ok=True)

    def _save_log(self, out_log):
        with open(self.log_name, "a") as f:
            f.write(json.dumps(out_log) + "\n")

    def train(self):

        output_log = []
        self._start_log()

        if self.prioritized_replay:
            replay = PrioritizedReplayBuffer(
                capacity_episodes=self.replay_capacity_episodes
            )
        else:
            replay = ReplayBuffer(capacity_episodes=self.replay_capacity_episodes)

        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if self.lr_scheduler is not None:
            assert hasattr(
                torch.optim.lr_scheduler, self.lr_scheduler
            ), f"Unknown lr_scheduler {self.lr_scheduler}"
            scheduler_class = getattr(torch.optim.lr_scheduler, self.lr_scheduler)
            scheduler = scheduler_class(optimizer, **self.lr_scheduler_params)

        for it in range(1, self.n_iterations):
            time_start = time.time()

            self.net.eval()
            if self.num_workers > 1:
                avg_length, avg_reward = self.run_self_play_executor(replay, it)
            else:
                avg_length = 0
                avg_reward = 0
                net_state = {k: v.cpu() for k, v in self.net.state_dict().items()}
                for i in range(self.self_play_episodes_per_iteration):
                    ep = _worker_self_play(
                        (
                            self.env,
                            self.env_args,
                            self.net_config,
                            net_state,
                            self.temperature_scheduler_function(it),
                            self.mcts_num_simulations,
                            self.mcts_config_self_play,
                            self._seed + i,
                        )
                    )
                    replay.add_episode(ep)
                    avg_length += len(ep["rewards"])
                    avg_reward += sum(ep["rewards"])

                self._seed += self.self_play_episodes_per_iteration
                avg_length /= self.self_play_episodes_per_iteration
                avg_reward /= self.self_play_episodes_per_iteration

            print(f"Iteration {it}: Replay buffer size: {len(replay)} transitions")

            if len(replay.episodes) < self.min_replay_episodes_for_training:
                continue

            # Train (a few SGD steps)
            self.net.train()
            stats = {}
            stats_list = []
            for _ in range(self.SGD_steps_per_iteration):
                # batch = replay.sample_positions(batch_size=self.batch_size)

                if self.prioritized_replay:
                    beta = self.per_beta0 + (self.per_beta1 - self.per_beta0) * (
                        it / max(1, self.n_iterations - 1)
                    )

                    batch, is_w, ep_indices = replay.sample_positions(
                        batch_size=self.batch_size, beta=beta
                    )
                    return_priorities = True
                else:
                    batch = replay.sample_positions(batch_size=self.batch_size)
                    is_w = None
                    return_priorities = False

                stats_step = train_step(
                    self.net,
                    optimizer,
                    batch,
                    K=self.K,
                    n_step=self.n_step,
                    gamma=self.gamma,
                    device=self.device,
                    is_weights=is_w,
                    return_priorities=return_priorities,
                )

                if self.prioritized_replay:
                    replay.update_priorities(ep_indices, stats_step.pop("priorities"))
                    stats_step["replay_buffer_beta"] = beta

                stats_list.append(stats_step)

            for k in stats_step.keys():
                stats[k] = np.mean([s[k] for s in stats_list]).item()
                stats[f"{k}_std"] = np.std([s[k] for s in stats_list]).item()
                stats[f"{k}_min"] = np.min([s[k] for s in stats_list]).item()
                stats[f"{k}_max"] = np.max([s[k] for s in stats_list]).item()
                stats[f"{k}_median"] = np.median([s[k] for s in stats_list]).item()

            if self.lr_scheduler is not None:
                scheduler.step()

            time_end = time.time()
            print(f"Iteration {it} took {time_end - time_start:.2f} seconds")
            stats["iteration_time"] = time_end - time_start
            stats["avg_self_play_length"] = avg_length
            stats["avg_self_play_reward"] = avg_reward
            stats["replay_size_episodes"] = len(replay.episodes)
            stats["replay_size_transitions"] = len(replay)
            stats["lr"] = optimizer.param_groups[0]["lr"]

            if it % self.eval_frequency == 0:
                eval_r = 0
                eval_len = 0
                net_state = {k: v.cpu() for k, v in self.net.state_dict().items()}
                for i in range(4):
                    ep = _worker_self_play(
                        (
                            self.env,
                            self.eval_env_args,
                            self.net_config,
                            net_state,
                            self.temperature_scheduler_function(it),
                            self.mcts_num_simulations,
                            self.mcts_config_self_play,
                            self._seed + i,
                        )
                    )
                    total_reward = sum(ep["rewards"])
                    eval_r += total_reward
                    eval_len += len(ep["rewards"])
                print(
                    f"Eval over 4 episodes: avg reward={eval_r/4}, avg length={eval_len/4}"
                )
                stats["eval_avg_reward"] = eval_r / 4
                stats["eval_avg_length"] = eval_len / 4
            else:
                stats["eval_avg_reward"] = None
                stats["eval_avg_length"] = None

            output_log.append((it, stats))
            print(it, stats)

            self._save_log({"iteration": it, **stats})

            if it % self.checkpoint_frquency == 0:
                if self.checkpoint_path is None:
                    save_checkpoint(
                        f"checkpoint_it{it}.pt",
                        self.net,
                        optimizer,
                        self.net.config,
                        it,
                    )
                else:
                    save_checkpoint(
                        self.checkpoint_path / f"checkpoint_it{it}.pt",
                        self.net,
                        optimizer,
                        self.net.config,
                        it,
                    )

        return output_log
