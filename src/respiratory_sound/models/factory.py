"""Construct respiratory-sound models from stored YAML/checkpoint dictionaries."""

from __future__ import annotations

from respiratory_sound.models.network import TFDCRNet


def tfdcr_from_config(config: dict[str, object]) -> TFDCRNet:
    return TFDCRNet(
        mode=str(config["mode"]),
        input_channels=int(config["input_channels"]),
        channels=tuple(int(value) for value in config["channels"]),
        depths=tuple(int(value) for value in config["depths"]),
        temporal_dilations=tuple(
            tuple(int(value) for value in stage)
            for stage in config["temporal_dilations"]
        ),
        expansion_ratio=int(config["expansion_ratio"]),
        num_classes=int(config["num_classes"]),
        include_joint=bool(config.get("include_joint", False)),
        branch_types=(
            tuple(str(branch) for branch in config["branch_types"])
            if "branch_types" in config
            else None
        ),
        pooling=str(config.get("pooling", "gap")),
        lse_temperature=float(config.get("lse_temperature", 1.0)),
        gate_conditioning=str(config.get("gate_conditioning", "none")),
        gate_stages=tuple(int(stage) for stage in config.get("gate_stages", ())),
        gate_reduction=int(config.get("gate_reduction", 4)),
        gate_max_delta=float(config.get("gate_max_delta", 0.5)),
        morphology_supervision=str(config.get("morphology_supervision", "none")),
        morphology_stages=tuple(
            int(stage) for stage in config.get("morphology_stages", ())
        ),
        morphology_branch_indices=tuple(
            int(index) for index in config.get("morphology_branch_indices", (1, 2))
        ),
        morphology_source=str(config.get("morphology_source", "branch_response")),
        morphology_routing=str(config.get("morphology_routing", "mean")),
        hierarchy_fusion=str(config.get("hierarchy_fusion", "none")),
        hierarchy_max_weight=float(config.get("hierarchy_max_weight", 1.0)),
        input_normalization=str(config.get("input_normalization", "none")),
        relaxed_frequency_weight=float(
            config.get("relaxed_frequency_weight", 0.5)
        ),
        normalization_eps=float(config.get("normalization_eps", 1.0e-5)),
    )
