import json
import pathlib

import numpy as np
import torch
from my_zero.models.networks import (
    MuZeroNet,
)
from my_zero.train.train import Trainer

USE_TEMPERATURE_SCHEDULE = False
USE_MCTS_SCHEDULE = False


def env_callback(env) -> dict:
    data = {}

    data["obs_length"] = env.observation_space.shape[0]

    return data


if __name__ == "__main__":
    SUPPORT_SIZE = 501
    SUPPORT_MIN = 0
    SUPPORT_MAX = 500
    support = torch.linspace(SUPPORT_MIN, SUPPORT_MAX, SUPPORT_SIZE)

    net_config = {
        "h": {
            "body": {
                "width": 16,
                "block_depth": 2,
                "blocks": 2,
                "input_dim": 4,
                "output_dim": 16,
                "skip_connection": True,
            },
            "h": {"normalize": "l2"},
        },
        "g": {
            "body": {
                "width": 16,
                "block_depth": 2,
                "blocks": 2,
                "input_dim": 16 + 16,
                "output_dim": 16,
                "skip_connection": True,
            },
            "g": {"num_actions": 2, "action_embed_dim": 16},
        },
        "f": {
            "body": {
                "width": 16,
                "block_depth": 2,
                "blocks": 2,
                "input_dim": 16,
                "output_dim": 16,
                "skip_connection": True,
            },
            "f": {
                "num_actions": 2,
                "output_probabilities": True,
                "support": support,
            },
        },
    }

    mu_zero_net = MuZeroNet(net_config=net_config)

    env = "CartPole-v1"

    device = "cpu"
    if device == "mps" and torch.backends.mps.is_available():
        print("Using MPS device")
        mu_zero_net = mu_zero_net.to("mps")

    temperature_steps = 35
    temperature_max = 1.5059488004297665
    temperature_min = 0.7876559203140695

    def temperature_schedule(iteration: int) -> float:
        if iteration <= temperature_steps:
            return temperature_max
        return np.clip(
            temperature_min
            + (temperature_max - temperature_min)
            / 800
            * (temperature_steps - iteration),
            temperature_min,
            temperature_max,
        )

    c_puct_max = 1.6086948978258278
    c_puct_steps = 35

    def c_puct_schedule(iteration: int) -> float:
        if iteration <= c_puct_steps:
            return c_puct_max
        return np.clip(
            1.0 + (c_puct_max - 1.0) / 75 * (c_puct_steps - iteration),
            1.0,
            c_puct_max,
        )

    learning_rate = 0.0006749089717469685
    eta_min_divisor = 7

    # The training and search parameters below were selected by hyperparameter search.
    config = {
        "n_iterations": 40,
        "mcts_num_simulations": 75,
        "self_play_episodes_per_iteration": 8,
        "batch_size": 128,
        "min_replay_episodes_for_training": 20,
        "replay_capacity_episodes": 74,
        "learning_rate": learning_rate,
        "SGD_steps_per_iteration": 1024,
        "lr_scheduler": "CosineAnnealingLR",
        "lr_scheduler_params": {
            "T_max": 40,
            "eta_min": learning_rate / eta_min_divisor,
        },
        "num_workers": 8,
        "log_path": pathlib.Path("./logs/cart_pole/"),
        "checkpoint_path": pathlib.Path("./checkpoints/cart_pole/"),
        "env_callback": env_callback,
        "prioritized_replay": False,
        "gamma": 0.9940388587715849,
    }

    if USE_TEMPERATURE_SCHEDULE:
        config["temperature_function"] = temperature_schedule

    if USE_MCTS_SCHEDULE:
        config["mcts_config_self_play"] = {
            "puct_opt": "puct",
            "c_puct": c_puct_max,
            "gamma": config["gamma"],
        }
        config["mcts_self_play_schedule"] = {"c_puct": c_puct_schedule}

    trainer = Trainer(
        net=mu_zero_net, env=env, device="cpu", config=config, net_config=net_config
    )  # mps

    out_log = trainer.train()

    with open("cart_pole_train_log.json", "w") as f:
        json.dump(out_log, f)
