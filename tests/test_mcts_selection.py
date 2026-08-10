import numpy as np
import pytest
import torch
from my_zero.MCTS.tree_search import MuZeroMCTS
from my_zero.MCTS.tree_search import Node as DiscreteNode
from my_zero.MCTS.tree_search_sampled import (
    MuZeroSampledMCTS,
)
from my_zero.MCTS.tree_search_sampled import (
    Node as SampledNode,
)


def make_mcts(mcts_class, puct_opt="puct", c_puct=1.5, **kwargs):
    if mcts_class is MuZeroMCTS:
        return mcts_class(None, None, num_actions=2, puct_opt=puct_opt, c_puct=c_puct)
    return mcts_class(
        None,
        None,
        num_actions=1,
        num_sampled_actions=2,
        sample_from_uniform=0,
        puct_opt=puct_opt,
        c_puct=c_puct,
        **kwargs,
    )


def make_parent(node_class):
    node_kwargs = {} if node_class is DiscreteNode else {"action": None}
    parent = node_class(prior=1.0, visit_count=10, **node_kwargs)
    child_0_kwargs = {} if node_class is DiscreteNode else {"action": 0}
    child_1_kwargs = {} if node_class is DiscreteNode else {"action": 1}
    parent.children = {
        0: node_class(
            prior=0.01,
            visit_count=1,
            value_sum=1.0,
            **child_0_kwargs,
        ),
        1: node_class(
            prior=0.99,
            visit_count=1,
            value_sum=0.0,
            **child_1_kwargs,
        ),
    }
    parent.action_values = {0: 1.0, 1: 0.0}
    parent.action_counts = {0: 1, 1: 1}
    return parent


@pytest.mark.parametrize(
    ("mcts_class", "node_class"),
    [(MuZeroMCTS, DiscreteNode), (MuZeroSampledMCTS, SampledNode)],
)
@pytest.mark.parametrize("puct_opt", ["puct", "muzero"])
def test_batched_selection_matches_loop_selection(mcts_class, node_class, puct_opt):
    mcts = make_mcts(mcts_class, puct_opt=puct_opt, c_puct=10.0)
    mcts.q_min = 0.0
    mcts.q_max = 1.0
    parent = make_parent(node_class)

    loop_action, _ = mcts._select_child(parent)
    batch_action, _ = mcts._select_child_batch(parent)

    assert batch_action == loop_action


@pytest.mark.parametrize(
    ("mcts_class", "node_class"),
    [(MuZeroMCTS, DiscreteNode), (MuZeroSampledMCTS, SampledNode)],
)
def test_batched_classic_puct_uses_configured_exploration_constant(
    mcts_class, node_class
):
    parent = make_parent(node_class)

    low_exploration = make_mcts(mcts_class, puct_opt="puct", c_puct=0.0)
    high_exploration = make_mcts(mcts_class, puct_opt="puct", c_puct=10.0)

    assert low_exploration._select_child_batch(parent)[0] == 0
    assert high_exploration._select_child_batch(parent)[0] == 1


@pytest.mark.parametrize(
    ("mcts_class", "node_class"),
    [(MuZeroMCTS, DiscreteNode), (MuZeroSampledMCTS, SampledNode)],
)
def test_classic_puct_uses_backed_up_edge_q_value(mcts_class, node_class):
    parent = make_parent(node_class)
    parent.action_values = {0: 0.0, 1: 1.0}
    mcts = make_mcts(mcts_class, puct_opt="puct", c_puct=0.0)

    assert mcts._select_child(parent)[0] == 1
    assert mcts._select_child_batch(parent)[0] == 1


@pytest.mark.parametrize("mcts_class", [MuZeroMCTS, MuZeroSampledMCTS])
def test_invalid_puct_option_is_rejected(mcts_class):
    with pytest.raises(ValueError, match="Unknown puct_opt"):
        make_mcts(mcts_class, puct_opt="typo")


def test_sampled_mcts_can_disable_batched_selection():
    mcts = make_mcts(MuZeroSampledMCTS, batched_search=False)

    assert mcts.batched_search is False


class FakeContinuousPrediction:
    lims = (-1.0, 1.0)

    def __call__(self, state):
        batch_size = state.shape[0]
        logits = torch.zeros((batch_size, 2), device=state.device)
        value = torch.zeros(batch_size, device=state.device)
        return logits, value

    @staticmethod
    def sample_action(logits, n_samples, return_prob):
        batch_size = logits.shape[0]
        actions = torch.linspace(-0.5, 0.5, n_samples, device=logits.device).reshape(
            n_samples, 1, 1
        )
        actions = actions.expand(n_samples, batch_size, 1)
        priors = torch.full(
            (n_samples, batch_size),
            1.0 / n_samples,
            device=logits.device,
        )
        std = torch.full((batch_size, 1), 0.5, device=logits.device)
        return actions, priors, (std, std.log())


class FakeContinuousDynamics:
    @staticmethod
    def __call__(state, action):
        return state, torch.zeros(state.shape[0], device=state.device)


def run_sampled_mcts_on_device(device):
    mcts = MuZeroSampledMCTS(
        prediction_net=FakeContinuousPrediction(),
        dynamics_net=FakeContinuousDynamics(),
        num_actions=1,
        num_sampled_actions=2,
        num_sampled_actions_root=2,
        sample_from_uniform=0,
        device=device,
    )
    visit_data, _, _, action_info = mcts.run(
        root_latent=torch.zeros(2, device=device),
        num_simulations=1,
        add_root_noise=False,
    )
    return visit_data, action_info


def test_sampled_mcts_returns_numpy_data_for_logging_and_replay():
    visit_data, action_info = run_sampled_mcts_on_device("cpu")

    assert all(isinstance(action, np.ndarray) for action in visit_data["actions"])
    assert all(isinstance(sigma, np.ndarray) for sigma in action_info["sigmas"])
    assert all(
        isinstance(log_sigma, np.ndarray) for log_sigma in action_info["log_sigmas"]
    )


def accelerator_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(accelerator_device() is None, reason="No accelerator available")
def test_sampled_mcts_numpy_conversion_works_on_accelerator():
    visit_data, action_info = run_sampled_mcts_on_device(accelerator_device())

    assert all(isinstance(action, np.ndarray) for action in visit_data["actions"])
    assert all(isinstance(sigma, np.ndarray) for sigma in action_info["sigmas"])
