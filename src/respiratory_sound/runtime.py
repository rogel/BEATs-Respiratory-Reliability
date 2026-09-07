"""Runtime helpers shared by training and evaluation scripts."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class RuntimeInfo:
    python_architecture: str
    torch_version: str
    mps_built: bool
    mps_available: bool
    selected_device: str

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def select_device(preference: str = "mps") -> torch.device:
    """Select MPS when requested and available, otherwise return CPU."""
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_runtime_info(preference: str = "mps") -> RuntimeInfo:
    device = select_device(preference)
    return RuntimeInfo(
        python_architecture=platform.machine(),
        torch_version=torch.__version__,
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
        selected_device=str(device),
    )
