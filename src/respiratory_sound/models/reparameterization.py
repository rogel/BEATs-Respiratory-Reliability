"""Exactly foldable directional time-frequency difference convolutions."""

from __future__ import annotations

import math
from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Offset = tuple[int, int]
Edge = tuple[Offset, Offset]

BRANCH_EDGES: dict[str, tuple[Edge, ...]] = {
    "temporal": (
        ((1, 0), (1, 1)),
        ((1, 2), (1, 1)),
    ),
    "frequency": (
        ((0, 1), (1, 1)),
        ((2, 1), (1, 1)),
    ),
    "joint": (
        ((0, 0), (1, 1)),
        ((0, 2), (1, 1)),
        ((2, 0), (1, 1)),
        ((2, 2), (1, 1)),
    ),
}


class DifferenceDepthwiseConv2d(nn.Module):
    """Depthwise convolution parameterized as weighted fixed pixel-pair differences."""

    def __init__(
        self,
        channels: int,
        branch_type: str,
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        if branch_type not in BRANCH_EDGES:
            raise ValueError(f"Unsupported difference branch: {branch_type}")
        self.channels = channels
        self.branch_type = branch_type
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = self.dilation
        edges = BRANCH_EDGES[branch_type]

        incidence = torch.zeros(len(edges), 9)
        for edge_index, (positive, negative) in enumerate(edges):
            incidence[edge_index, positive[0] * 3 + positive[1]] += 1.0
            incidence[edge_index, negative[0] * 3 + negative[1]] -= 1.0
        self.register_buffer("incidence", incidence, persistent=True)
        self.edge_weights = nn.Parameter(torch.empty(channels, len(edges)))
        nn.init.kaiming_uniform_(self.edge_weights, a=math.sqrt(5))

    def equivalent_kernel(self) -> Tensor:
        """Map pairwise differences to an ordinary 3x3 depthwise kernel."""
        return (self.edge_weights @ self.incidence).reshape(self.channels, 1, 3, 3)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(
            inputs,
            self.equivalent_kernel(),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.channels,
        )


class ConvBNBranch(nn.Module):
    """A linear convolution followed by batch normalization."""

    def __init__(
        self,
        channels: int,
        branch_type: str,
        stride: int | tuple[int, int],
        dilation: int | tuple[int, int],
    ) -> None:
        super().__init__()
        self.branch_type = branch_type
        if branch_type == "standard":
            dilation_pair = _pair(dilation)
            self.conv: nn.Module = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=stride,
                padding=dilation_pair,
                dilation=dilation_pair,
                groups=channels,
                bias=False,
            )
        else:
            self.conv = DifferenceDepthwiseConv2d(
                channels=channels,
                branch_type=branch_type,
                stride=stride,
                dilation=dilation,
            )
        self.bn = nn.BatchNorm2d(channels)

    def ordinary_kernel(self) -> Tensor:
        if isinstance(self.conv, nn.Conv2d):
            return self.conv.weight
        if isinstance(self.conv, DifferenceDepthwiseConv2d):
            return self.conv.equivalent_kernel()
        raise TypeError(f"Unsupported branch convolution: {type(self.conv)}")

    def forward(self, inputs: Tensor) -> Tensor:
        return self.bn(self.conv(inputs))


class ReparamDepthwiseConv2d(nn.Module):
    """Training-time multi-branch layer with an exactly equivalent deploy convolution."""

    def __init__(
        self,
        channels: int,
        branch_types: Iterable[str] = ("standard", "temporal", "frequency"),
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        branch_types = tuple(branch_types)
        if not branch_types:
            raise ValueError("At least one branch is required")
        self.channels = channels
        self.branch_types = branch_types
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = self.dilation
        self.deploy = deploy

        if deploy:
            self.reparam_conv = self._make_deploy_conv()
        else:
            self.branches = nn.ModuleList(
                [
                    ConvBNBranch(
                        channels=channels,
                        branch_type=branch_type,
                        stride=self.stride,
                        dilation=self.dilation,
                    )
                    for branch_type in branch_types
                ]
            )
            self.branch_logits = nn.Parameter(torch.zeros(len(branch_types)))

    def _make_deploy_conv(self) -> nn.Conv2d:
        return nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.channels,
            bias=True,
        )

    def branch_weights(self) -> Tensor:
        if self.deploy:
            raise RuntimeError("Branch weights are unavailable after deployment conversion")
        return torch.softmax(self.branch_logits, dim=0)

    def forward_branches(self, inputs: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Return the fused response and differentiable pre-fusion branch responses."""
        if self.deploy:
            raise RuntimeError("Branch responses are unavailable after deployment conversion")
        branch_outputs = tuple(branch(inputs) for branch in self.branches)
        weights = self.branch_weights()
        fused = sum(
            weight * output
            for weight, output in zip(weights, branch_outputs, strict=True)
        )
        return fused, branch_outputs

    @staticmethod
    def _fold_bn(kernel: Tensor, bn: nn.BatchNorm2d) -> tuple[Tensor, Tensor]:
        if bn.running_mean is None or bn.running_var is None:
            raise RuntimeError("BatchNorm running statistics are required for deployment")
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        folded_kernel = kernel * scale.reshape(-1, 1, 1, 1)
        folded_bias = bn.bias - bn.running_mean * scale
        return folded_kernel, folded_bias

    def get_equivalent_kernel_bias(self) -> tuple[Tensor, Tensor]:
        if self.deploy:
            return self.reparam_conv.weight, self.reparam_conv.bias
        if self.training:
            raise RuntimeError("Call eval() before computing the equivalent deployment kernel")

        weights = self.branch_weights()
        kernels: list[Tensor] = []
        biases: list[Tensor] = []
        for branch in self.branches:
            kernel, bias = self._fold_bn(branch.ordinary_kernel(), branch.bn)
            kernels.append(kernel)
            biases.append(bias)
        equivalent_kernel = sum(
            weight * kernel for weight, kernel in zip(weights, kernels, strict=True)
        )
        equivalent_bias = sum(weight * bias for weight, bias in zip(weights, biases, strict=True))
        return equivalent_kernel, equivalent_bias

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam_conv = self._make_deploy_conv().to(
            device=kernel.device,
            dtype=kernel.dtype,
        )
        with torch.no_grad():
            reparam_conv.weight.copy_(kernel)
            reparam_conv.bias.copy_(bias)
        self.reparam_conv = reparam_conv
        del self.branches
        del self.branch_logits
        self.deploy = True

    def forward(self, inputs: Tensor) -> Tensor:
        if self.deploy:
            return self.reparam_conv(inputs)
        fused, _ = self.forward_branches(inputs)
        return fused


def switch_model_to_deploy(model: nn.Module, inplace: bool = False) -> nn.Module:
    """Convert every reparameterizable layer in a model to a plain convolution."""
    converted = model if inplace else deepcopy(model)
    converted.eval()
    for module in converted.modules():
        if isinstance(module, ReparamDepthwiseConv2d):
            module.switch_to_deploy()
    if hasattr(converted, "remove_training_heads"):
        converted.remove_training_heads()
    return converted


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"Expected a pair, got: {value}")
    return value
