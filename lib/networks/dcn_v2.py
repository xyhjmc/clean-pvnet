#!/usr/bin/env python
"""Deformable Convolution wrappers backed by torchvision.

The original project relied on a custom CUDA extension. Modern PyTorch
and torchvision ship DeformConv2d/DeformRoIPool2d kernels, so we build a
thin compatibility layer on top of them. When torchvision is missing the
ops (e.g. CPU-only installs), the code gracefully falls back to standard
convolutions to keep training scripts runnable.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:  # torchvision>=0.8
    from torchvision.ops import DeformConv2d, DeformRoIPool
except Exception:  # pragma: no cover - fallback for minimal environments
    DeformConv2d = None
    DeformRoIPool = None


class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

        if DeformConv2d is not None:
            self.op = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                bias=True,
                deformable_groups=deformable_groups,
            )
        else:
            self.op = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            )

    def forward(self, input, offset, mask=None):
        if isinstance(self.op, nn.Conv2d):
            return self.op(input)

        # torchvision's implementation expects offset with shape [N, 2*kH*kW, H, W]
        # and an optional modulation mask [N, kH*kW, H, W].
        if mask is None:
            mask = torch.ones(
                (input.shape[0], self.kernel_size[0] * self.kernel_size[1], input.shape[2], input.shape[3]),
                device=input.device,
                dtype=input.dtype,
            )
        return self.op(input, offset, mask)


class DCN(DCNv2):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
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
            self.pool = DeformRoIPool((pooled_size, pooled_size), spatial_scale)

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
