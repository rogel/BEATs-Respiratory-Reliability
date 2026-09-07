import torch

from respiratory_sound.models.pretrained_audio import (
    PANNsCnn6Transfer,
    _beats_legacy_token_valid_mask,
    _beats_token_support_fraction,
    _beats_token_valid_mask,
    _masked_max_mean,
)


def test_masked_max_mean_ignores_padding_values() -> None:
    features = torch.tensor(
        [
            [
                [1.0, 3.0, 100.0],
                [2.0, 4.0, -100.0],
            ]
        ]
    )
    mask = torch.tensor([[True, True, False]])

    pooled = _masked_max_mean(features, mask)

    assert torch.allclose(pooled, torch.tensor([[5.0, 7.0]]))


def test_panns_cnn6_transfer_produces_finite_binary_logits() -> None:
    model = PANNsCnn6Transfer(classes_num=2).eval()
    waveforms = torch.zeros(2, 32_000)
    sample_masks = torch.zeros_like(waveforms, dtype=torch.bool)
    sample_masks[0, 2_000:30_000] = True
    sample_masks[1, 5_000:28_000] = True
    waveforms[sample_masks] = torch.randn(int(sample_masks.sum())) * 0.01

    with torch.inference_mode():
        logits = model(waveforms, sample_masks)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_beats_mask_preserves_short_events_at_arbitrary_pad_positions() -> None:
    sample_masks = torch.zeros(2, 128_000, dtype=torch.bool)
    sample_masks[0, 114_458:116_474] = True
    sample_masks[1, 125_554:127_570] = True

    token_valid = _beats_token_valid_mask(
        sample_masks,
        fbank_frames=798,
        patch_size=16,
        frequency_patches=8,
    )

    assert token_valid.shape == (2, 400)
    assert token_valid.sum(dim=1).tolist() == [16, 16]


def test_beats_support_fraction_defines_all_three_token_categories() -> None:
    sample_masks = torch.zeros(1, 128_000, dtype=torch.bool)
    sample_masks[0, 1_000:10_000] = True

    fractions = _beats_token_support_fraction(
        sample_masks,
        fbank_frames=798,
        patch_size=16,
        frequency_patches=8,
    )
    token_valid = _beats_token_valid_mask(
        sample_masks,
        fbank_frames=798,
        patch_size=16,
        frequency_patches=8,
    )

    assert fractions.shape == (1, 400)
    assert bool((fractions == 0).any())
    assert bool(((fractions > 0) & (fractions < 1)).any())
    assert bool((fractions == 1).any())
    assert torch.equal(fractions > 0, token_valid)


def test_beats_support_fraction_excludes_samples_beyond_frontend_coverage() -> None:
    sample_masks = torch.zeros(1, 128_000, dtype=torch.bool)
    sample_masks[0, -40:] = True

    fractions = _beats_token_support_fraction(
        sample_masks,
        fbank_frames=798,
        patch_size=16,
        frequency_patches=8,
    )

    assert torch.equal(fractions, torch.zeros_like(fractions))


def test_legacy_mask_replays_documented_arbitrary_position_failure() -> None:
    sample_masks = torch.zeros(2, 128_000, dtype=torch.bool)
    sample_masks[0, 114_458:116_474] = True
    sample_masks[1, 125_554:127_570] = True

    token_valid = _beats_legacy_token_valid_mask(
        sample_masks,
        fbank_frames=798,
        tokens=392,
    )

    assert token_valid.shape == (2, 392)
    assert token_valid.sum(dim=1).tolist() == [7, 0]


def test_beats_mask_rejects_audio_outside_fbank_coverage() -> None:
    sample_masks = torch.zeros(1, 128_000, dtype=torch.bool)
    sample_masks[0, -40:] = True

    try:
        _beats_token_valid_mask(
            sample_masks,
            fbank_frames=798,
            patch_size=16,
            frequency_patches=8,
        )
    except ValueError as error:
        assert "all-padding" in str(error)
    else:
        raise AssertionError("Expected an all-padding BEATs mask error")
