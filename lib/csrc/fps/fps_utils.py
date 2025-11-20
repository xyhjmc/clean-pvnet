"""Pure NumPy fallback implementation of farthest point sampling.

The original project relied on a custom C++/CFFI extension. To keep the
codebase compatible with modern PyTorch environments without compiling
native extensions, this implementation mirrors the greedy FPS algorithm
in vectorised NumPy. It works for both CPU and GPU workflows because the
output is converted back to NumPy before further processing.
"""

import numpy as np


def _pairwise_squared_distance(pts):
    """Compute pairwise squared distances for FPS."""
    diff = pts[:, None, :] - pts[None, :, :]
    return np.einsum("...i,...i->...", diff, diff)


def farthest_point_sampling(pts, sn, init_center=False):
    """Greedy farthest point sampling implemented with NumPy only.

    Args:
        pts: ``(N, 3)`` array of points.
        sn: number of samples to return.
        init_center: if True, start from the centroid instead of a random
            point.
    """
    assert pts.shape[1] == 3
    pts = np.asarray(pts, dtype=np.float32)
    pn = pts.shape[0]
    sn = min(sn, pn)

    if pn == 0:
        return pts

    distances = _pairwise_squared_distance(pts)
    selected = np.zeros(sn, dtype=np.int64)

    if init_center:
        center = np.mean(pts, axis=0, keepdims=True)
        start_idx = np.argmax(np.einsum("ni,ni->n", pts - center, pts - center))
    else:
        start_idx = 0

    selected[0] = start_idx
    min_dist = distances[start_idx]

    for i in range(1, sn):
        farthest = np.argmax(min_dist)
        selected[i] = farthest
        min_dist = np.minimum(min_dist, distances[farthest])

    return pts[selected]
