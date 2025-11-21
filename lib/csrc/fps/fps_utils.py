"""Pure Python farthest point sampling backed by PyTorch operations.

The original project shipped a bespoke C++ extension for FPS. That build
path has been removed in favour of a lean implementation that works with
either :class:`numpy.ndarray` inputs or :class:`torch.Tensor` inputs on CPU
and GPU alike. The routine follows the standard greedy FPS strategy while
avoiding native compilation and keeping the same return shape semantics as
the legacy code.
"""

import numpy as np
import torch


def _to_tensor(pts):
    if isinstance(pts, np.ndarray):
        return torch.from_numpy(pts)
    if torch.is_tensor(pts):
        return pts
    raise TypeError("pts must be a numpy array or torch Tensor")


def farthest_point_sampling(pts, sn, init_center=False):
    """Greedy farthest point sampling implemented with PyTorch only.

    Args:
        pts: ``(N, 3)`` array of points or a ``(N, 3)`` tensor.
        sn: number of samples to return.
        init_center: if True, start from the centroid instead of the first
            point.
    """
    pts_tensor = _to_tensor(pts).to(dtype=torch.float32)
    assert pts_tensor.shape[1] == 3, "points must be of shape (N, 3)"

    pn = pts_tensor.shape[0]
    sn = min(int(sn), int(pn))

    if pn == 0:
        return pts

    device = pts_tensor.device
    selected = torch.zeros(sn, dtype=torch.long, device=device)

    if init_center:
        center = pts_tensor.mean(dim=0, keepdim=True)
        start_idx = torch.argmax(torch.sum((pts_tensor - center) ** 2, dim=1)).item()
    else:
        start_idx = 0

    selected[0] = start_idx
    min_dist = torch.sum((pts_tensor - pts_tensor[start_idx]) ** 2, dim=1)

    for i in range(1, sn):
        farthest = torch.argmax(min_dist)
        selected[i] = farthest
        candidate_dist = torch.sum((pts_tensor - pts_tensor[farthest]) ** 2, dim=1)
        min_dist = torch.minimum(min_dist, candidate_dist)

    sampled = pts_tensor[selected]
    if isinstance(pts, np.ndarray):
        return sampled.cpu().numpy()
    return sampled
