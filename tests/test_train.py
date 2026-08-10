import my_zero.train.train as train_module
import numpy as np
import torch


def test_trainer_forwards_configured_loss_weights(monkeypatch):
    captured = {}

    def fake_train_step(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"loss": 0.0}

    monkeypatch.setattr(train_module, "train_step", fake_train_step)

    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.net = object()
    trainer.K = 3
    trainer.n_step = 4
    trainer.gamma = 0.95
    trainer.device = "cpu"
    trainer.w_policy = 0.25
    trainer.w_value = 1.5
    trainer.w_reward = 2.0
    optimizer = object()
    batch = object()

    stats = trainer._train_on_batch(
        optimizer,
        batch,
        is_weights=None,
        return_priorities=False,
    )

    assert stats == {"loss": 0.0}
    assert captured["args"] == (trainer.net, optimizer, batch)
    assert captured["kwargs"]["w_policy"] == 0.25
    assert captured["kwargs"]["w_value"] == 1.5
    assert captured["kwargs"]["w_reward"] == 2.0


def test_training_iterations_include_configured_final_iteration():
    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.n_iterations = 3

    assert list(trainer._training_iterations()) == [1, 2, 3]


def test_prioritized_replay_beta_spans_first_to_final_iteration():
    trainer = train_module.Trainer.__new__(train_module.Trainer)
    trainer.n_iterations = 4
    trainer.per_beta0 = 0.4
    trainer.per_beta1 = 1.0

    assert trainer._prioritized_replay_beta(1) == 0.4
    assert trainer._prioritized_replay_beta(4) == 1.0


def test_illegal_actions_are_excluded_from_policy_normalization():
    logits = torch.tensor([[0.0, 100.0, 0.0]])
    legal_mask = torch.tensor([[1.0, 0.0, 1.0]])

    masked_logits = train_module._mask_illegal_action_logits(logits, legal_mask)
    target = torch.tensor([[1.0, 0.0, 0.0]])
    loss = -(target * torch.log_softmax(masked_logits, dim=-1)).sum(dim=-1)

    torch.testing.assert_close(loss, torch.tensor([np.log(2.0)], dtype=loss.dtype))


def test_padded_position_has_no_policy_target():
    episode = {
        "actions": [0],
        "pis": [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
        "legal_masks": [np.array([1.0, 0.0, 1.0], dtype=np.float32)],
        "rewards": [0.0],
        "terminated": [True],
        "truncated": [False],
        "root_v_est": [0.0],
    }

    target_pis, _, _ = train_module.make_targets(
        episode, t0=1, K=0, n_step=1, gamma=0.99
    )

    np.testing.assert_allclose(target_pis[0], [0.0, 0.0, 0.0])
