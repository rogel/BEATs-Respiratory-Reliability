from pathlib import Path


def test_local_beats_source_remains_upstream_compatible() -> None:
    source = (
        Path(__file__).parents[1]
        / "third_party"
        / "beats"
        / "backbone.py"
    ).read_text(encoding="utf-8")

    assert "x[padding_mask] = 0" in source
    assert "attn_weights = attn_weights.masked_fill(" in source
