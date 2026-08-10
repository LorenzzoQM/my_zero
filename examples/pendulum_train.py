import json
import logging
import pathlib

import numpy as np
import torch
from my_zero.MCTS.tree_search_sampled import MuZeroSampledMCTS
from my_zero.models.networks import (
    MuZeroNet,
)
from my_zero.train.train import Trainer

USE_TEMPERATURE_SCHEDULE = False
USE_MCTS_SCHEDULE = False


def train_case():

    SUPPORT_SIZE = 1501
    SUPPORT_MIN = -1500
    SUPPORT_MAX = 0
    support = torch.linspace(SUPPORT_MIN, SUPPORT_MAX, SUPPORT_SIZE)

    net_config = {
        "h": {
            "body": {
                "width": 64,
                "block_depth": 2,
                "blocks": 1,
                "input_dim": 3,
                "output_dim": 64,
                "skip_connection": True,
            },
            "h": {"normalize": "l2"},
        },
        "g": {
            "body": {
                "width": 64,
                "block_depth": 2,
                "blocks": 1,
                "input_dim": 64 + 1,
                "output_dim": 64,
                "skip_connection": True,
            },
            "g": {"num_actions": 1, "action_embed_dim": 0},
        },
        "f": {
            "body": {
                "width": 128,
                "block_depth": 1,
                "blocks": 2,
                "input_dim": 64,
                "output_dim": 128,
                "skip_connection": True,
            },
            "f": {
                "num_actions": 1,
                "output_probabilities": True,
                "support": support,
                "lims": (-2, 2),
            },
        },
        "continuous_actions": True,
    }

    mu_zero_net = MuZeroNet(net_config=net_config)

    env = "Pendulum-v1"

    device = "cpu"
    if device == "mps" and torch.backends.mps.is_available():
        print("Using MPS device")
        mu_zero_net = mu_zero_net.to("mps")

    temperature_max = 1.4
    temperature_steps = 250
    temperature_min = 0.77

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

    beta_max = 1.05
    beta_steps = 20

    def beta_temperature_schedule(iteration: int) -> float:
        if iteration <= beta_steps:
            return beta_max
        return np.clip(
            1.0 + (beta_max - 1.0) / 75 * (beta_steps - iteration), 1.0, beta_max
        )

    c_puct_max = 1.5
    c_puct_steps = 20

    def c_puct_schedule(iteration: int) -> float:
        if iteration <= c_puct_steps:
            return c_puct_max
        return np.clip(
            1.0 + (c_puct_max - 1.0) / 75 * (c_puct_steps - iteration),
            1.0,
            c_puct_max,
        )

    # The training and search parameters below were selected by hyperparameter search.
    config = {
        "n_iterations": 200,
        "mcts_num_simulations": 250,
        "self_play_episodes_per_iteration": 32,
        "batch_size": 512,
        "min_replay_episodes_for_training": 32,
        "replay_capacity_episodes": 2000,
        "learning_rate": 7e-4,
        "lr_scheduler": "CosineAnnealingLR",
        "lr_scheduler_params": {
            "T_max": 600,
            "eta_min": 7e-4 / 35,
        },
        "gamma": 0.998,
        "SGD_steps_per_iteration": 256,
        "num_workers": 10,
        "log_path": pathlib.Path("./logs/pendulum/"),
        "checkpoint_path": pathlib.Path("./checkpoints/pendulum/"),
        "mcts_class": MuZeroSampledMCTS,
        "mcts_config_self_play": {
            "num_sampled_actions": 4,
            "num_sampled_actions_root": 32,
            "sample_from_uniform": 3,
            "gamma": 0.998,
            "beta_temp": 1.5,
            "c_puct": 1.5,
        },
        "logger_level": logging.INFO,
        "training_time_limit_seconds": 3600 * 48,  # 48 hours
        "prioritized_replay": True,
    }

    if USE_TEMPERATURE_SCHEDULE:
        config["temperature_function"] = temperature_schedule

    if USE_MCTS_SCHEDULE:
        config["mcts_self_play_schedule"] = {
            "beta_temp": beta_temperature_schedule,
            "c_puct": c_puct_schedule,
        }

    trainer = Trainer(
        net=mu_zero_net, env=env, device="cpu", config=config, net_config=net_config
    )  # mps

    out_log = trainer.train()

    return out_log


if __name__ == "__main__":
    output = train_case()

    with open("pendulum_train_log.json", "w") as f:
        json.dump(output, f)
