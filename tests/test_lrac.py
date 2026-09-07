import torch
from torch import nn

from respiratory_sound.models.lrac import (
    LRACNet,
    masked_resample_tokens,
    reverse_valid_tokens,
)


def _small_model(mode: str) -> LRACNet:
    return LRACNet(
        alignment_mode=mode,
        spectrum_channels=(8, 12),
        spectrum_depths=(1, 1),
        spectrum_temporal_dilations=((1,), (1,)),
        waveform_channels=(8, 12, 16),
        projection_dimension=8,
        interaction_hidden=8,
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    spectrograms = torch.randn(3, 1, 16, 41)
    waveforms = torch.randn(3, 1, 3_200)
    frame_masks = torch.ones(3, 41, dtype=torch.bool)
    frame_masks[1, :8] = False
    sample_masks = torch.ones(3, 3_200, dtype=torch.bool)
    sample_masks[1, :600] = False
    return spectrograms, waveforms, frame_masks, sample_masks


def test_valid_token_reversal_preserves_masked_multiset() -> None:
    tokens = torch.arange(2 * 6 * 3).reshape(2, 6, 3).float()
    masks = torch.tensor(
        [
            [False, True, True, True, False, False],
            [True, False, True, False, True, False],
        ]
    )
    reversed_tokens = reverse_valid_tokens(tokens, masks)

    assert torch.equal(reversed_tokens[0, 1:4], tokens[0, 1:4].flip(0))
    assert torch.equal(reversed_tokens[1, [0, 2, 4]], tokens[1, [4, 2, 0]])
    assert torch.equal(reversed_tokens[~masks], tokens[~masks])


def test_masked_resample_supports_non_divisible_lengths() -> None:
    features = torch.arange(10).reshape(1, 1, 10).float()
    masks = torch.tensor([[False, True, True, True, True, True, True, True, False, False]])
    tokens, output_masks = masked_resample_tokens(features, masks, output_steps=3)

    assert tokens.shape == (1, 3, 1)
    assert output_masks.tolist() == [[True, True, True]]
    assert torch.allclose(tokens[0, :, 0], torch.tensor([1.5, 4.0, 6.5]))


def test_lrac_starts_as_exact_spectrum_anchor() -> None:
    torch.manual_seed(5)
    model = _small_model("aligned").eval()
    final, spectrum, _, residual, _, _, masks, gates = model(
        *_inputs(),
        return_aux=True,
    )

    assert torch.equal(final, spectrum)
    assert torch.count_nonzero(residual) == 0
    assert masks.any(dim=1).all()
    assert torch.isfinite(gates).all()


def test_aligned_and_reverse_models_have_identical_initialization_and_size() -> None:
    torch.manual_seed(17)
    aligned = _small_model("aligned")
    torch.manual_seed(17)
    reverse = _small_model("reverse_control")

    assert sum(p.numel() for p in aligned.parameters()) == sum(
        p.numel() for p in reverse.parameters()
    )
    for aligned_parameter, reverse_parameter in zip(
        aligned.parameters(),
        reverse.parameters(),
        strict=True,
    ):
        assert torch.equal(aligned_parameter, reverse_parameter)


def test_one_optimizer_step_activates_lrac_residual() -> None:
    torch.manual_seed(23)
    model = _small_model("aligned")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss_function = nn.CrossEntropyLoss()
    targets = torch.tensor([0, 1, 0])
    outputs = model(*_inputs(), return_aux=True)
    final, spectrum, waveform, _, positive, negative, masks, _ = outputs
    alignment_logits = torch.cat((positive[masks], negative[masks]))
    alignment_targets = torch.cat(
        (
            torch.ones_like(positive[masks]),
            torch.zeros_like(negative[masks]),
        )
    )
    loss = (
        0.5 * loss_function(final, targets)
        + 0.5 * loss_function(spectrum, targets)
        + 0.3 * loss_function(waveform, targets)
        + 0.1 * nn.functional.binary_cross_entropy_with_logits(
            alignment_logits,
            alignment_targets,
        )
    )
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        residual = model(*_inputs(), return_aux=True)[3]

    assert torch.count_nonzero(residual) > 0
