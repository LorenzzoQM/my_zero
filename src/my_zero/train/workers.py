import os
import torch
import gymnasium as gym
from my_zero.train.data import self_play_episode
import logging
import logging.handlers

_WORKER_ENV = None
_WORKER_NET = None
_WORKER_DEVICE = None
_MCTS_CLASS = None


def _init_self_play_worker(
    q, env_cls, env_args, net_cls, mcts_class, net_config, device_str="cpu"
):
    """Runs once per worker process."""
    global _WORKER_ENV, _WORKER_NET, _WORKER_DEVICE, _MCTS_CLASS

    _WORKER_DEVICE = torch.device(device_str)

    # Create env once per worker
    if isinstance(env_cls, str):
        _WORKER_ENV = gym.make(env_cls, **env_args)
    else:
        _WORKER_ENV = env_cls(**env_args)

    # Create net once per worker
    _WORKER_NET = net_cls(net_config=net_config).to(_WORKER_DEVICE).eval()

    # Compile once per worker
    _WORKER_NET = torch.compile(_WORKER_NET)
    logger = logging.getLogger("my_zero")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        qh = logging.handlers.QueueHandler(q)
        logger.addHandler(qh)
    logger.debug(
        f"Initialized worker pid {os.getpid()} on device {_WORKER_DEVICE} with net type {type(_WORKER_NET)}"
    )

    _MCTS_CLASS = mcts_class


def _worker_self_play_one_episode(
    net_state_cpu,
    temperature,
    mcts_class,
    mcts_num_simulations,
    mcts_config_self_play,
    seed,
):
    global _WORKER_ENV, _WORKER_NET, _WORKER_DEVICE

    state = {
        k: v.to(_WORKER_DEVICE, non_blocking=True) for k, v in net_state_cpu.items()
    }

    target = _WORKER_NET._orig_mod if hasattr(_WORKER_NET, "_orig_mod") else _WORKER_NET
    target.load_state_dict(state, strict=True)

    mcts = mcts_class(
        prediction_net=_WORKER_NET.f,
        dynamics_net=_WORKER_NET.g,
        num_actions=_WORKER_ENV.action_space.n,
        **mcts_config_self_play,
    )

    episode = self_play_episode(
        env=_WORKER_ENV,
        net=_WORKER_NET,
        mcts=mcts,
        temperature=temperature,
        mcts_num_simulations=mcts_num_simulations,
        seed=seed,
        device=_WORKER_DEVICE,
    )
    return episode
