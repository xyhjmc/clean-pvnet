import numpy as np


def find_nearest_point_idx(ref_pts, que_pts):
    """Vectorised nearest-neighbour search in pure NumPy.

    Args:
        ref_pts: reference points, shape ``(N, D)``.
        que_pts: query points, shape ``(M, D)``.
    Returns:
        ``(M,)`` integer array containing the index of the closest reference
        point for each query point.
    """
    assert ref_pts.shape[1] == que_pts.shape[1] and 1 < que_pts.shape[1] <= 3
    ref_pts = np.asarray(ref_pts, dtype=np.float32)
    que_pts = np.asarray(que_pts, dtype=np.float32)

    # Compute squared distances efficiently using broadcasting.
    diff = que_pts[:, None, :] - ref_pts[None, :, :]
    dist2 = np.einsum("...i,...i->...", diff, diff)
    idxs = np.argmin(dist2, axis=1).astype(np.int32)
    return idxs
