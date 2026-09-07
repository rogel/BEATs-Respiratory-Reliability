#!/usr/bin/env python3
"""Profile model size, deployment equivalence, and forward/backward latency."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from respiratory_sound.models.network import TFDCRNet
from respiratory_sound.models.reparameterization import switch_model_to_deploy


@dataclass(frozen=True)
class ProfileResult:
    mode: str
    device: str
    input_shape: list[int]
    training_parameters: int
    deployed_parameters: int
    deployment_parameter_reduction_percent: float
    maximum_deployment_error: float
    median_forward_ms: float
    median_train_step_ms: float
    peak_mps_driver_memory_mib: float | None


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _median_latency(
    action: callable,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        action()
    _synchronize(device)
    timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        action()
        _synchronize(device)
        timings.append((time.perf_counter() - started) * 1_000)
    return float(torch.tensor(timings).median().item())


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def profile_mode(
    mode: str,
    device: torch.device,
    batch_size: int,
    frames: int,
    warmup: int,
    iterations: int,
) -> ProfileResult:
    torch.manual_seed(17)
    inputs = torch.randn(batch_size, 1, 64, frames, device=device)
    targets = torch.randint(0, 4, (batch_size,), device=device)

    training_model = TFDCRNet(mode=mode).to(device)
    training_model.eval()
    with torch.inference_mode():
        training_outputs = training_model(inputs)
        deployed_model = switch_model_to_deploy(training_model).to(device)
        deployed_outputs = deployed_model(inputs)
        maximum_error = float((training_outputs - deployed_outputs).abs().max().cpu())

    def forward_action() -> Tensor:
        with torch.inference_mode():
            return deployed_model(inputs)

    train_model = TFDCRNet(mode=mode).to(device).train()
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=1e-3)

    def train_action() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = train_model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        return loss

    forward_ms = _median_latency(
        forward_action,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    train_step_ms = _median_latency(
        train_action,
        device=device,
        warmup=max(1, warmup // 2),
        iterations=max(3, iterations // 2),
    )

    peak_memory: float | None = None
    if device.type == "mps":
        peak_memory = float(torch.mps.driver_allocated_memory() / (1024**2))

    training_parameters = _count_parameters(training_model)
    deployed_parameters = _count_parameters(deployed_model)
    reduction = 100.0 * (1.0 - deployed_parameters / training_parameters)
    return ProfileResult(
        mode=mode,
        device=str(device),
        input_shape=list(inputs.shape),
        training_parameters=training_parameters,
        deployed_parameters=deployed_parameters,
        deployment_parameter_reduction_percent=reduction,
        maximum_deployment_error=maximum_error,
        median_forward_ms=forward_ms,
        median_train_step_ms=train_step_ms,
        peak_mps_driver_memory_mib=peak_memory,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frames", type=int, default=501)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")
    device = torch.device(args.device)
    results = [
        profile_mode(
            mode=mode,
            device=device,
            batch_size=args.batch_size,
            frames=args.frames,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for mode in ("plain", "repconv", "tfdcr")
    ]
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
