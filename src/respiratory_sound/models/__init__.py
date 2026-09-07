"""Model definitions and deployment conversion utilities."""

from respiratory_sound.models.network import (
    AnchoredChannelGate,
    EventPool2d,
    MaskAwareRelaxedFrequencyNorm,
    TFDCRNet,
)
from respiratory_sound.models.reparameterization import (
    DifferenceDepthwiseConv2d,
    ReparamDepthwiseConv2d,
    switch_model_to_deploy,
)

__all__ = [
    "AnchoredChannelGate",
    "DifferenceDepthwiseConv2d",
    "EventPool2d",
    "MaskAwareRelaxedFrequencyNorm",
    "ReparamDepthwiseConv2d",
    "TFDCRNet",
    "switch_model_to_deploy",
]
from respiratory_sound.models.lrac import LRACNet
from respiratory_sound.models.wavespec import (
    LightweightWaveformEncoder,
    WaveSpecEvidenceFusion,
)

__all__ = ["LRACNet", "LightweightWaveformEncoder", "WaveSpecEvidenceFusion"]
