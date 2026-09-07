import random

import numpy as np
import torch
from torch import nn

from respiratory_sound.training_state import (
    atomic_torch_save,
    capture_rng_state,
    load_trainable_parameter_state,
    restore_rng_state,
)


def test_rng_state_round_trip_on_cpu() -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = capture_rng_state(torch.device("cpu"))
    expected = (random.random(), np.random.rand(), torch.rand(3))

    random.random()
    np.random.rand()
    torch.rand(3)
    restore_rng_state(state, torch.device("cpu"))
    actual = (random.random(), np.random.rand(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_trainable_parameter_restore_is_exact() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    model[0].requires_grad_(False)
    names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    with torch.no_grad():
        for parameter in model[1].parameters():
            parameter.add_(2.0)

    load_trainable_parameter_state(model, state, names)

    assert all(
        torch.equal(dict(model.named_parameters())[name], value)
        for name, value in state.items()
    )


def test_atomic_torch_save_replaces_complete_payload(tmp_path) -> None:
    path = tmp_path / "resume_state.pt"
    atomic_torch_save({"epoch": 1}, path)
    atomic_torch_save({"epoch": 2}, path)

    assert torch.load(path, weights_only=False) == {"epoch": 2}
    assert not (tmp_path / ".resume_state.pt.tmp").exists()
