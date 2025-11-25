#!/usr/bin/env python
"""Deformable convolution wrappers backed by PyTorch/torchvision.

The original project relied on a custom CUDA extension. Modern PyTorch
and torchvision ship native deformable kernels, so we provide a small
compatibility layer that matches the historical DCN API while keeping
weights and initialization consistent with standard convolutions. When
the torchvision ops are unavailable (e.g., CPU-only installs), we fall
back to a regular :func:`torch.nn.functional.conv2d` so the code remains
runnable, albeit without deformation offsets.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

try:  # torchvision>=0.8
    from torchvision.ops import DeformRoIPool, deform_conv2d
except Exception:  # pragma: no cover - fallback for minimal environments
    DeformRoIPool = None
    deform_conv2d = None


class _DeformConvWrapper(nn.Module):
    """Adapter that mirrors the DCN extension's call signature.

    The wrapper owns the weight and bias tensors to keep the parameter
    shapes and initialization identical to a standard :class:`nn.Conv2d`.
    When torchvision's `deform_conv2d` is unavailable, the forward pass
    gracefully degrades to a plain convolution while still accepting the
    ``offset``/``mask`` arguments.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int | Tuple[int, int],
        padding: int | Tuple[int, int],
        dilation: int | Tuple[int, int] = 1,
        deformable_groups: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        weight = torch.empty(out_channels, in_channels, *self.kernel_size)
        bias = torch.empty(out_channels)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, offset: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        if offset is None:
            offset = torch.zeros(
                (
                    input.shape[0],
                    2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1],
                    input.shape[2],
                    input.shape[3],
                ),
                device=input.device,
                dtype=input.dtype,
            )

        if mask is None:
            mask = torch.ones(
                (
                    input.shape[0],
                    self.deformable_groups * self.kernel_size[0] * self.kernel_size[1],
                    input.shape[2],
                    input.shape[3],
                ),
                device=input.device,
                dtype=input.dtype,
            )

        if deform_conv2d is None:
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
            deformable_groups=self.deformable_groups,
        )


class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.op = _DeformConvWrapper(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            deformable_groups=deformable_groups,
        )

    def forward(self, input, offset=None, mask=None):
        # The wrapper handles offset/mask defaults to maintain signature
        # compatibility with the historical CUDA extension.
        return self.op(input, offset=offset, mask=mask)


class DCN(DCNv2):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        kernel_size = _pair(kernel_size)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups)
        channels_ = deformable_groups * 3 * kernel_size[0] * kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            channels_,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return super().forward(input, offset, mask)


class DCNv2Pooling(nn.Module):
    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=.0):
        super().__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

        if DeformRoIPool is None:
            self.pool = nn.AdaptiveMaxPool2d((pooled_size, pooled_size))
        else:
            self.pool = DeformRoIPool(pooled_size, pooled_size, spatial_scale)

    def forward(self, input, rois, offset=None):
        if isinstance(self.pool, nn.AdaptiveMaxPool2d):
            outputs = []
            for roi in rois:
                batch_idx = int(roi[0].item())
                x1, y1, x2, y2 = roi[1:]
                x1 = int(x1.item())
                y1 = int(y1.item())
                x2 = int(x2.item())
                y2 = int(y2.item())
                region = input[batch_idx : batch_idx + 1, :, y1:y2, x1:x2]
                outputs.append(self.pool(region))
            return torch.cat(outputs, dim=0)

        if offset is None:
            offset = torch.zeros((rois.shape[0], 2 * self.pooled_size * self.pooled_size, 1, 1), device=input.device, dtype=input.dtype)
        return self.pool(input, rois, offset)


class DCNPooling(DCNv2Pooling):
    def __init__(self, spatial_scale, pooled_size, output_dim, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=.0):
        super().__init__(spatial_scale, pooled_size, output_dim, no_trans, group_size, part_size, sample_per_part, trans_std)
