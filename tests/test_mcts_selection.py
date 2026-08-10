import pytest
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
