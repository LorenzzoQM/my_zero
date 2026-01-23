import pytest
import numpy as np
import torch
from my_zero.MCTS.tree_search import MuZeroMCTS





def test_select_action_from_visits():

    prediction_net = lambda x: (torch.from_numpy(np.array([1.0,0,0,0,0])), torch.tensor([0.1]))
    dynamics_net = lambda s, a: (s, torch.tensor([0.1]))
    mcts = MuZeroMCTS(prediction_net, dynamics_net, num_actions=5, discount=0.9)

    visit_counts, root_value_count, root_value = mcts.run(root_latent=torch.tensor([0.0]*10), num_simulations=1)

    assert np.sum(visit_counts) == 1
    np.testing.assert_almost_equal(root_value_count, 0.19)
    np.testing.assert_almost_equal(root_value, 0.1)


if __name__ == "__main__":
    test_select_action_from_visits()