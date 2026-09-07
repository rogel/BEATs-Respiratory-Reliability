"""Small, auditable helpers for exact epoch-boundary training recovery."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Capture host and accelerator RNG state after a complete epoch."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_mps": None,
    }
    if device.type == "mps":
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    """Restore RNG state after model, optimizer and loaders are constructed."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "mps":
        mps_state = state.get("torch_mps")
        if mps_state is None:
            raise ValueError("MPS recovery state is missing")
        torch.mps.set_rng_state(mps_state)


def load_trainable_parameter_state(
    model: nn.Module,
    state: dict[str, Tensor],
    names: list[str],
) -> None:
    """Restore exactly the parameters that the current strategy marks trainable."""
    expected = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(names) != expected or set(state) != expected:
        raise ValueError("Recovery state does not match current trainable parameters")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in names:
            target = parameters[name]
            target.copy_(state[name].to(device=target.device, dtype=target.dtype))


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Replace an epoch recovery file only after its new copy is complete."""
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
