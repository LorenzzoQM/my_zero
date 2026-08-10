import numpy as np
import pytest
import torch
from my_zero.train.data import (
    Episode,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    self_play_episode,
)

AGENTS = ("agent_0", "agent_1")


class FakeMultiAgentEnv:
    def __init__(self, finish_by_truncation):
        self.finish_by_truncation = finish_by_truncation
        self.step_count = 0

    def _observations(self):
        return {
            agent: np.array([self.step_count], dtype=np.float32) for agent in AGENTS
        }

    def reset(self, seed=None):
        self.step_count = 0
        return self._observations(), {}

    def step(self, actions):
        self.step_count += 1
        finished = self.step_count == 2
        terminated = {
            agent: finished and not self.finish_by_truncation for agent in AGENTS
        }
        truncated = {agent: finished and self.finish_by_truncation for agent in AGENTS}
        rewards = {agent: 1.0 for agent in AGENTS}
        return self._observations(), rewards, terminated, truncated, {}


class FakeNeverDoneMultiAgentEnv(FakeMultiAgentEnv):
    def __init__(self):
        super().__init__(finish_by_truncation=False)

    def step(self, actions):
        self.step_count += 1
        flags = {agent: False for agent in AGENTS}
        rewards = {agent: 1.0 for agent in AGENTS}
        return self._observations(), rewards, flags.copy(), flags.copy(), {}


class FakeNeverDoneSingleAgentEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        observation = np.array([self.step_count], dtype=np.float32)
        return observation, 1.0, False, False, {}


class FakeNet:
    @staticmethod
    def h(observation):
        return observation


class FakeMCTS:
    num_actions = 2

    @staticmethod
    def run(**kwargs):
        return np.array([1, 0]), 0.5, 0.5, {}

    @staticmethod
    def select_action_from_visits(visit_counts, temperature, return_pi):
        return 0, np.array([1.0, 0.0], dtype=np.float32)

    @staticmethod
    def _compute_entropy(visit_counts):
        return 0.25


class FakeSampledMCTS(FakeMCTS):
    @staticmethod
    def run(**kwargs):
        return (
            {
                "visit_counts": np.array([1, 0]),
                "actions": [
                    np.array([-0.5], dtype=np.float32),
                    np.array([0.5], dtype=np.float32),
                ],
            },
            0.5,
            0.5,
            {
                "sigmas": np.array([0.2], dtype=np.float32),
                "log_sigmas": np.array([-1.6], dtype=np.float32),
            },
        )

    @staticmethod
    def select_action_from_visits(visit_counts, temperature, return_pi):
        return visit_counts["actions"][0], np.array([1.0, 0.0], dtype=np.float32)

    @staticmethod
    def _compute_entropy(visit_counts):
        return 0.5


@pytest.mark.parametrize(
    ("finish_by_truncation", "expected_root_values"),
    [(False, 2), (True, 3)],
)
def test_multi_agent_bootstrap_is_added_only_for_truncation(
    finish_by_truncation, expected_root_values
):
    embeddings = {agent: np.array([0.0], dtype=np.float32) for agent in AGENTS}

    episodes, log_data = self_play_episode(
        env=FakeMultiAgentEnv(finish_by_truncation),
        env_callback=None,
        net=FakeNet(),
        mcts=FakeMCTS(),
        temperature=0.0,
        agents_embedding=embeddings,
        add_root_noise=False,
    )

    for episode in episodes.values():
        assert len(episode.actions) == 2
        assert len(episode.root_v_est) == expected_root_values

    for agent in AGENTS:
        assert log_data["entropy_MCTS"][agent] == [0.25, 0.25]
        assert all(isinstance(obs, np.ndarray) for obs in episodes[agent].obs)


def test_multi_agent_sampled_action_statistics_are_logged_per_agent():
    embeddings = {agent: np.array([0.0], dtype=np.float32) for agent in AGENTS}

    episodes, log_data = self_play_episode(
        env=FakeMultiAgentEnv(finish_by_truncation=False),
        env_callback=None,
        net=FakeNet(),
        mcts=FakeSampledMCTS(),
        temperature=0.0,
        agents_embedding=embeddings,
        add_root_noise=False,
    )

    for agent in AGENTS:
        assert len(episodes[agent].actions_sampled) == 2
        assert log_data["entropy_MCTS"][agent] == [0.5, 0.5]
        assert log_data["STD_sampled"][agent] == [0.5, 0.5]
        assert len(log_data["sigmas"][agent]) == 2
        assert len(log_data["log_sigmas"][agent]) == 2


def test_single_agent_max_steps_cutoff_bootstraps_as_truncation():
    episode, _ = self_play_episode(
        env=FakeNeverDoneSingleAgentEnv(),
        env_callback=None,
        net=FakeNet(),
        mcts=FakeMCTS(),
        temperature=0.0,
        max_steps=2,
        add_root_noise=False,
    )

    assert episode.terminated == [False, False]
    assert episode.truncated == [False, True]
    assert len(episode.root_v_est) == 3


def test_multi_agent_max_steps_cutoff_bootstraps_as_truncation():
    embeddings = {agent: np.array([0.0], dtype=np.float32) for agent in AGENTS}
    episodes, _ = self_play_episode(
        env=FakeNeverDoneMultiAgentEnv(),
        env_callback=None,
        net=FakeNet(),
        mcts=FakeMCTS(),
        temperature=0.0,
        max_steps=2,
        agents_embedding=embeddings,
        add_root_noise=False,
    )

    for episode in episodes.values():
        assert episode.terminated == [False, False]
        assert episode.truncated == [False, True]
        assert len(episode.root_v_est) == 3


def make_episode(length):
    return Episode(
        obs=[None] * length,
        actions=[0] * length,
        actions_sampled=[],
        rewards=[0.0] * length,
        terminated=[False] * length,
        truncated=[False] * length,
        pis=[None] * length,
        root_v_est=[0.0] * length,
        legal_masks=[None] * length,
    )


@pytest.mark.parametrize(
    "replay",
    [ReplayBuffer(capacity_episodes=2), PrioritizedReplayBuffer(capacity_episodes=2)],
)
def test_replay_length_reports_transitions(replay):
    replay.add_episode(make_episode(2))
    replay.add_episode(make_episode(3))

    assert len(replay.episodes) == 2
    assert len(replay) == 5


def accelerator_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(accelerator_device() is None, reason="No accelerator available")
def test_multi_agent_observations_are_stored_as_numpy_on_accelerator():
    embeddings = {agent: np.array([0.0], dtype=np.float32) for agent in AGENTS}
    episodes, _ = self_play_episode(
        env=FakeMultiAgentEnv(finish_by_truncation=False),
        env_callback=None,
        net=FakeNet(),
        mcts=FakeMCTS(),
        temperature=0.0,
        device=accelerator_device(),
        agents_embedding=embeddings,
        add_root_noise=False,
    )

    for episode in episodes.values():
        assert all(isinstance(obs, np.ndarray) for obs in episode.obs)
