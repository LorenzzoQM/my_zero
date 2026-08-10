import torch
from my_zero.models.networks import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    BodyMLP,
    DynamicsMLP,
    PredictorMLP,
    PredictorMLPCont,
    scalar_to_support,
)
from torch.distributions import Normal


def test_scalar_to_support_preserves_integer_target_mass():
    support = torch.arange(-2, 3, dtype=torch.float32)
    targets = torch.tensor([-2.0, 0.0, 1.0, 1.5, 2.0])

    projected = scalar_to_support(targets, support)

    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(projected, expected)
    torch.testing.assert_close(projected.sum(dim=1), torch.ones(targets.shape[0]))


def test_squashed_actions_use_consistent_log_std_bounds():
    body = BodyMLP(
        width=4,
        block_depth=1,
        blocks=1,
        input_dim=2,
        output_dim=4,
    )
    predictor = PredictorMLPCont(
        body=body,
        num_actions=2,
        lims=(-2.0, 2.0),
        squashed_actions=True,
    )
    logits = torch.tensor([[0.0, 0.0, -100.0, 100.0]])

    _, _, (sample_std, sample_log_std) = predictor.sample_action_squashed(logits)

    expected_log_std = torch.tensor([[LOG_STD_MIN, LOG_STD_MAX]])
    torch.testing.assert_close(sample_log_std, expected_log_std)
    torch.testing.assert_close(sample_std, expected_log_std.exp())

    midpoint_action = torch.zeros((1, 2))
    actual_logp = predictor.logp_squashed(logits, midpoint_action)
    expected_logp = (
        Normal(torch.zeros_like(expected_log_std), expected_log_std.exp())
        .log_prob(torch.zeros_like(expected_log_std))
        .sum(dim=-1)
    )
    expected_logp -= torch.log(torch.tensor(2.0))
    torch.testing.assert_close(actual_logp, expected_logp)


def test_distributional_supports_are_registered_buffers():
    support = torch.arange(-2, 3, dtype=torch.float32)
    predictor = PredictorMLP(
        body=BodyMLP(
            width=4,
            block_depth=1,
            blocks=1,
            input_dim=3,
            output_dim=4,
        ),
        num_actions=2,
        output_probabilities=True,
        support=support,
    )
    dynamics = DynamicsMLP(
        body=BodyMLP(
            width=4,
            block_depth=1,
            blocks=1,
            input_dim=5,
            output_dim=4,
        ),
        num_actions=2,
        action_embed_dim=1,
        output_probabilities=True,
        support=support,
    )

    for model in (predictor, dynamics):
        assert "support" in dict(model.named_buffers())
        assert "support" in model.state_dict()

        model.to(dtype=torch.float64)
        assert model.support.dtype == torch.float64
