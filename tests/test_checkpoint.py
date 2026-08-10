import pytest
import torch

from my_zero.train.train import load_checkpoint, save_checkpoint


def maybe_compile(model, compile_model):
    return torch.compile(model) if compile_model else model


def assert_optimizer_states_equal(expected, actual):
    expected_states = list(expected.state_dict()["state"].values())
    actual_states = list(actual.state_dict()["state"].values())
    assert len(actual_states) == len(expected_states)

    for expected_state, actual_state in zip(expected_states, actual_states):
        assert actual_state.keys() == expected_state.keys()
        for key, expected_value in expected_state.items():
            actual_value = actual_state[key]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value


@pytest.mark.parametrize("save_compiled", [False, True])
@pytest.mark.parametrize("load_compiled", [False, True])
def test_checkpoint_loads_between_compiled_and_uncompiled_models(
    tmp_path, save_compiled, load_compiled
):
    source = torch.nn.Linear(2, 1)
    source_optimizer = torch.optim.Adam(source.parameters(), lr=0.01)
    source(torch.ones((1, 2))).sum().backward()
    source_optimizer.step()

    source_for_save = maybe_compile(source, save_compiled)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        source_for_save,
        source_optimizer,
        config={"name": "test"},
        iteration=7,
    )

    saved = torch.load(checkpoint_path, map_location="cpu")
    assert all(not key.startswith("_orig_mod.") for key in saved["model_state_dict"])

    target = torch.nn.Linear(2, 1)
    target_for_load = maybe_compile(target, load_compiled)
    target_optimizer = torch.optim.Adam(target_for_load.parameters(), lr=0.01)

    iteration, config = load_checkpoint(
        checkpoint_path,
        target_for_load,
        optimizer=target_optimizer,
    )

    assert iteration == 7
    assert config == {"name": "test"}
    for expected, actual in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(actual, expected)
    assert_optimizer_states_equal(source_optimizer, target_optimizer)


def test_load_checkpoint_accepts_legacy_compiled_state_dict(tmp_path):
    source = torch.nn.Linear(2, 1)
    legacy_compiled_source = torch.compile(source)
    checkpoint_path = tmp_path / "legacy_compiled_checkpoint.pt"
    torch.save(
        {
            "iteration": 3,
            "model_state_dict": legacy_compiled_source.state_dict(),
            "optimizer_state_dict": {},
            "config": None,
        },
        checkpoint_path,
    )

    target = torch.nn.Linear(2, 1)
    iteration, config = load_checkpoint(checkpoint_path, target)

    assert iteration == 3
    assert config is None
    for expected, actual in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(actual, expected)
