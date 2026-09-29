"""Planar 6-DoF pose from scratch: DLT homography -> decomposition -> LM refinement.

Pipeline for a square marker of known side length:

1. ``homography_dlt`` fits H (marker plane -> image) from the 4 corners with
   Hartley normalisation.
2. ``pose_from_homography`` uses  K^-1 H ~ [r1 r2 t]:  scale so that r1, r2 are
   unit length, complete r3 = r1 x r2 and project onto SO(3) with an SVD.
3. ``refine_pose_lm`` minimises pixel reprojection error with Levenberg-Marquardt
   on the manifold (multiplicative so(3) updates) using an analytic Jacobian.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- rotations
def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle vector -> rotation matrix (exponential map)."""
    rvec = np.asarray(rvec, float).reshape(3)
    theta = np.linalg.norm(rvec)
    Kx = skew(rvec)
    if theta < 1e-8:  # second-order Taylor expansion
        return np.eye(3) + Kx + 0.5 * Kx @ Kx
    return (np.eye(3) + np.sin(theta) / theta * Kx
            + (1 - np.cos(theta)) / theta ** 2 * Kx @ Kx)


def rodrigues_inv(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector (logarithm map), stable near 0 and pi."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-6:
        return 0.5 * w
    if np.pi - theta < 1e-4:
        # R ~ 2 a a^T - I near pi: take the axis from the largest diagonal term
        B = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(B)))
        a = B[:, i] / np.sqrt(B[i, i])
        if np.dot(a, w) < 0:
            a = -a
        return theta * a / np.linalg.norm(a)
    return theta / (2 * np.sin(theta)) * w


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees.

    Uses atan2(sin, cos) of the relative rotation rather than arccos(cos): arccos
    loses half the significant digits near 0 (its resolution is ~sqrt(eps) rad).
    """
    Q = R1.T @ R2
    s = 0.5 * np.linalg.norm([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
    c = (np.trace(Q) - 1) / 2
    return float(np.degrees(np.arctan2(s, c)))


def project_to_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


# ---------------------------------------------------------------- homography
def _normaliser(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / d
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalised DLT: H such that dst ~ H src, for n >= 4 point pairs."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    Ts, Td = _normaliser(src), _normaliser(dst)
    sh = np.column_stack([src, np.ones(len(src))]) @ Ts.T
    dh = np.column_stack([dst, np.ones(len(dst))]) @ Td.T
    rows = []
    for (x, y, w), (u, v, q) in zip(sh, dh):
        rows.append([0, 0, 0, -q * x, -q * y, -q * w, v * x, v * y, v * w])
        rows.append([q * x, q * y, q * w, 0, 0, 0, -u * x, -u * y, -u * w])
    _, _, Vt = np.linalg.svd(np.array(rows))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def pose_from_homography(H: np.ndarray, K: np.ndarray):
    """Decompose a plane->image homography into (R, t) with the plane in front."""
    M = np.linalg.inv(K) @ H
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    if M[2, 2] * lam < 0:  # t_z must be positive: plane in front of the camera
        lam = -lam
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    R = project_to_so3(np.column_stack([r1, r2, np.cross(r1, r2)]))
    return R, t


# ---------------------------------------------------------------- refinement
def _project(K, R, t, X):
    Xc = X @ R.T + t
    uv = Xc @ K.T
    return uv[:, :2] / uv[:, 2:3], Xc


def reprojection_rms(K, R, t, X, uv) -> float:
    pred, _ = _project(K, R, t, X)
    return float(np.sqrt(np.mean(np.sum((pred - uv) ** 2, axis=1))))


def refine_pose_lm(K, R, t, X, uv, iters: int = 50, tol: float = 1e-12):
    """Levenberg-Marquardt on SE(3): R <- exp(w) R, t <- t + dt.

    Analytic Jacobian of the pinhole projection:
        d(u,v)/dXc = [[fx/z, 0, -fx x/z^2], [0, fy/z, -fy y/z^2]]
        dXc/dw     = -[R X]_x      (left perturbation)
        dXc/dt     = I
    """
    fx, fy = K[0, 0], K[1, 1]
    R, t = R.copy(), np.asarray(t, float).copy()
    mu = 1e-3
    pred, Xc = _project(K, R, t, X)
    r = (pred - uv).ravel()
    cost = r @ r
    for _ in range(iters):
        J = np.zeros((2 * len(X), 6))
        RX = X @ R.T
        for i, ((x, y, z), rx) in enumerate(zip(Xc, RX)):
            Jp = np.array([[fx / z, 0, -fx * x / z ** 2], [0, fy / z, -fy * y / z ** 2]])
            J[2 * i:2 * i + 2, :3] = Jp @ (-skew(rx))
            J[2 * i:2 * i + 2, 3:] = Jp
        A = J.T @ J
        g = J.T @ r
        while True:
            delta = -np.linalg.solve(A + mu * np.diag(np.diag(A) + 1e-12), g)
            R_new = rodrigues(delta[:3]) @ R
            t_new = t + delta[3:]
            pred_new, Xc_new = _project(K, R_new, t_new, X)
            r_new = (pred_new - uv).ravel()
            cost_new = r_new @ r_new
            if cost_new <= cost:
                mu = max(mu / 10, 1e-12)
                break
            mu *= 10
            if mu > 1e12:
                return R, t
        improvement = cost - cost_new
        R, t, Xc, r, cost = R_new, t_new, Xc_new, r_new, cost_new
        if improvement < tol * max(cost, 1e-30) or np.linalg.norm(delta) < 1e-14:
            break
    return R, t


def estimate_pose(K, corners_2d: np.ndarray, side: float, refine: bool = True):
    """Full pipeline for one square marker: corners TL,TR,BR,BL in pixels."""
    h = side / 2
    X = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)
    H = homography_dlt(X[:, :2], corners_2d)
    R, t = pose_from_homography(H, K)
    if refine:
        R, t = refine_pose_lm(K, R, t, X, np.asarray(corners_2d, float))
    return R, t
