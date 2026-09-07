import torch
from torch import nn

from respiratory_sound.models.wavespec import WaveSpecEvidenceFusion


def _small_model(mode: str) -> WaveSpecEvidenceFusion:
    return WaveSpecEvidenceFusion(
        fusion_mode=mode,
        spectrum_channels=(8, 12),
        spectrum_depths=(1, 1),
        spectrum_temporal_dilations=((1,), (1,)),
        waveform_channels=(8, 12, 16),
        fusion_hidden=8,
    )


def test_fusion_starts_as_exact_spectrum_anchor() -> None:
    torch.manual_seed(7)
    model = _small_model("interaction").eval()
    spectrograms = torch.randn(3, 1, 16, 41)
    waveforms = torch.randn(3, 1, 3_200)
    frame_masks = torch.ones(3, 41, dtype=torch.bool)
    sample_masks = torch.ones(3, 3_200, dtype=torch.bool)

    final, spectrum, _, residual, features = model(
        spectrograms,
        waveforms,
        frame_masks,
        sample_masks,
        return_aux=True,
    )

    assert torch.equal(final, spectrum)
    assert torch.count_nonzero(residual) == 0
    assert torch.count_nonzero(features[:, 2:]) > 0


def test_control_and_interaction_have_identical_parameter_counts() -> None:
    control = _small_model("concat_control")
    candidate = _small_model("interaction")

    control_parameters = sum(parameter.numel() for parameter in control.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())

    assert control_parameters == candidate_parameters


def test_one_optimizer_step_activates_residual() -> None:
    torch.manual_seed(11)
    model = _small_model("interaction")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss_function = nn.CrossEntropyLoss()
    spectrograms = torch.randn(4, 1, 16, 41)
    waveforms = torch.randn(4, 1, 3_200)
    masks = torch.ones(4, 41, dtype=torch.bool)
    sample_masks = torch.ones(4, 3_200, dtype=torch.bool)
    targets = torch.tensor([0, 1, 0, 1])

    logits, _, waveform_logits, _, _ = model(
        spectrograms,
        waveforms,
        masks,
        sample_masks,
        return_aux=True,
    )
    loss = loss_function(logits, targets) + 0.3 * loss_function(
        waveform_logits,
        targets,
    )
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        _, _, _, residual, _ = model(
            spectrograms,
            waveforms,
            masks,
            sample_masks,
            return_aux=True,
        )

    assert torch.count_nonzero(residual) > 0
