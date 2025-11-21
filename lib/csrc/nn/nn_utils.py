import numpy as np
import torch


def _to_tensor(pts):
    if isinstance(pts, np.ndarray):
        return torch.from_numpy(pts)
    if torch.is_tensor(pts):
        return pts
    raise TypeError("points must be a numpy array or torch Tensor")


def find_nearest_point_idx(ref_pts, que_pts, exclude_self=False):
    """Nearest-neighbour search backed by PyTorch and NumPy.

    Args:
        ref_pts: reference points of shape ``(N, D)`` or ``(B, N, D)``.
        que_pts: query points of shape ``(M, D)`` or ``(B, M, D)``.
        exclude_self: optionally ignore identical indices when ``ref_pts`` and
            ``que_pts`` share the same second dimension.
    Returns:
        Indices of the closest reference point for each query, shaped like
        ``(M,)`` or ``(B, M)`` depending on the input rank.
    """

    ref_t = _to_tensor(ref_pts).to(dtype=torch.float32)
    que_t = _to_tensor(que_pts).to(dtype=torch.float32)
    assert ref_t.shape[-1] == que_t.shape[-1] and 1 < ref_t.shape[-1] <= 3

    batched = ref_t.dim() == 3
    if not batched:
        ref_t = ref_t.unsqueeze(0)
        que_t = que_t.unsqueeze(0)

    dist = torch.cdist(que_t, ref_t, p=2)
    if exclude_self and ref_t.shape[1] == que_t.shape[1]:
        mask = torch.eye(ref_t.shape[1], device=dist.device, dtype=torch.bool)
        dist = dist.masked_fill(mask.unsqueeze(0), float("inf"))

    idxs = torch.argmin(dist, dim=-1)

    if not batched:
        idxs = idxs.squeeze(0)

    if isinstance(ref_pts, np.ndarray) and isinstance(que_pts, np.ndarray):
        return idxs.cpu().numpy().astype(np.int32)
    return idxs
