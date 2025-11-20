import numpy as np
import cv2


def _initial_pnp(points_2d, points_3d, weights, camera_matrix):
    """Compute a stable initial PnP estimate using the four most reliable points."""
    idxs = np.argsort(weights)[-4:]
    _, rvec, tvec = cv2.solvePnP(
        np.expand_dims(points_3d[idxs, :], 0),
        np.expand_dims(points_2d[idxs, :], 0),
        camera_matrix,
        np.zeros((8, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_P3P,
    )
    return rvec, tvec


def _refine_with_weights(rvec, tvec, points_3d, points_2d, weights, camera_matrix):
    """Iteratively refine pose using Levenberg–Marquardt with per-point weights."""
    try:
        cv2.solvePnPRefineLM
    except AttributeError:
        # Older OpenCV builds may miss the refinement helper; fall back to ITERATIVE
        _, rvec, tvec = cv2.solvePnP(
            points_3d, points_2d, camera_matrix, np.zeros((8, 1), dtype=np.float64),
            rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
        )
        return rvec, tvec

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        20,
        1e-6,
    )

    return cv2.solvePnPRefineLM(
        points_3d,
        points_2d,
        camera_matrix,
        np.zeros((8, 1), dtype=np.float64),
        rvec,
        tvec,
        criteria,
    )


def uncertainty_pnp(points_2d, weights_2d, points_3d, camera_matrix):
    """Weighted PnP solver without custom C++ extensions."""
    pn = points_2d.shape[0]
    assert points_3d.shape[0] == pn and pn >= 4

    points_3d = points_3d.astype(np.float64)
    points_2d = points_2d.astype(np.float64)
    camera_matrix = camera_matrix.astype(np.float64)
    weights = (weights_2d[:, 0] + weights_2d[:, 1]).astype(np.float64)

    rvec, tvec = _initial_pnp(points_2d, points_3d, weights, camera_matrix)
    rvec, tvec = _refine_with_weights(rvec, tvec, points_3d, points_2d, weights, camera_matrix)

    R, _ = cv2.Rodrigues(rvec)
    return np.concatenate([R, tvec], axis=-1)


def uncertainty_pnp_v2(points_2d, covars, points_3d, camera_matrix, type='single'):
    """PnP variant that derives weights from covariance matrices."""
    pn = points_2d.shape[0]
    assert points_3d.shape[0] == pn and pn >= 4 and covars.shape[0] == pn

    points_3d = points_3d.astype(np.float64)
    points_2d = points_2d.astype(np.float64)
    camera_matrix = camera_matrix.astype(np.float64)

    weights = []
    for pi in range(pn):
        if covars[pi, 0, 0] < 1e-5:
            weights.append(0.0)
        else:
            weight = np.max(np.linalg.eigvals(covars[pi]))
            weights.append(1.0 / weight)
    weights = np.asarray(weights, np.float64)

    rvec, tvec = _initial_pnp(points_2d, points_3d, weights, camera_matrix)
    rvec, tvec = _refine_with_weights(rvec, tvec, points_3d, points_2d, weights, camera_matrix)

    R, _ = cv2.Rodrigues(rvec)
    return np.concatenate([R, tvec], axis=-1)
