"""Check the project environment without accessing research datasets."""

from __future__ import annotations

import json

import torch
import torchaudio

from respiratory_sound.runtime import get_runtime_info, select_device


def run_smoke_test() -> dict[str, object]:
    device = select_device("mps")
    waveform = torch.randn(2, 1, 16_000, dtype=torch.float32)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16_000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=64,
        f_min=50,
        f_max=2_000,
        power=2.0,
    )(waveform)
    log_mel = torch.log(mel.clamp_min(1.0e-10)).to(device)

    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(8),
        torch.nn.GELU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 4),
    ).to(device)
    targets = torch.tensor([0, 3], device=device)
    logits = model(log_mel)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    if device.type == "mps":
        torch.mps.synchronize()

    return {
        "runtime": get_runtime_info().to_dict(),
        "torchaudio_version": torchaudio.__version__,
        "waveform_shape": list(waveform.shape),
        "log_mel_shape": list(log_mel.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "gradient_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().cpu())
            for parameter in model.parameters()
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke_test(), indent=2))
