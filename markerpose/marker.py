"""Marker dictionary and a procedural pinhole renderer.

Conventions
-----------
* Camera frame is OpenCV's: x right, y down, z forward (into the scene).
* A marker lives on its own z = 0 plane, centred at the origin, x right, y up.
  Its 3-D corners are ordered TL, TR, BR, BL (clockwise when seen from the front).
* A pose (R, t) maps marker coordinates into camera coordinates: Xc = R X + t.
"""
from __future__ import annotations

import numpy as np
import cv2

BITS = 4            # 4x4 payload
CELLS = BITS + 2    # plus a one-cell black border on each side


def _rot_code(code: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(code, k)


class MarkerDictionary:
    """A small ArUco-style dictionary built by greedy search.

    Every code is at Hamming distance >= ``min_distance`` from every rotation of
    every other code, *and* from its own three non-trivial rotations, so the
    decoded bit pattern identifies both the marker ID and its orientation.
    """

    def __init__(self, size: int = 24, min_distance: int = 4, seed: int = 7):
        rng = np.random.default_rng(seed)
        candidates = rng.permutation(1 << (BITS * BITS))
        weights = 1 << np.arange(BITS * BITS)[::-1]
        codes: list[np.ndarray] = []
        taken: list[np.ndarray] = []  # all rotations of accepted codes, flattened
        for c in candidates:
            bits = ((int(c) >> np.arange(BITS * BITS)[::-1]) & 1).reshape(BITS, BITS)
            ones = bits.sum()
            if ones < 4 or ones > 12:          # avoid near-blank / near-solid payloads
                continue
            rots = [_rot_code(bits, k).ravel() for k in range(4)]
            if min(np.sum(rots[0] != r) for r in rots[1:]) < min_distance:
                continue
            if taken and np.min(np.sum(np.array(taken) != rots[0], axis=1)) < min_distance:
                continue
            codes.append(bits.astype(np.uint8))
            taken.extend(rots)
            if len(codes) == size:
                break
        if len(codes) < size:
            raise ValueError("could not build a dictionary that large")
        self.codes = codes
        self.min_distance = min_distance
        self._weights = weights

    def __len__(self) -> int:
        return len(self.codes)

    def image(self, marker_id: int, cell_px: int = 20, quiet_cells: int = 1) -> np.ndarray:
        """Marker texture (uint8, 0 = black) including a white quiet zone."""
        grid = np.zeros((CELLS, CELLS), np.uint8)
        grid[1:-1, 1:-1] = self.codes[marker_id]
        grid = np.pad(grid, quiet_cells, constant_values=1)
        return np.kron(grid, np.ones((cell_px, cell_px), np.uint8)) * 255

    def identify(self, bits: np.ndarray, max_correction: int = 1):
        """Return (id, k, distance) such that bits == rot90(code[id], k), or None."""
        best = None
        for mid, code in enumerate(self.codes):
            for k in range(4):
                d = int(np.sum(_rot_code(code, k) != bits))
                if best is None or d < best[2]:
                    best = (mid, k, d)
        if best is not None and best[2] <= max_correction:
            return best
        return None


def marker_corners_3d(side: float) -> np.ndarray:
    h = side / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xc = X @ R.T + t.reshape(1, 3)
    uvw = Xc @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def render_scene(K: np.ndarray, size: tuple[int, int], markers, dictionary: MarkerDictionary,
                 noise_sigma: float = 2.0, supersample: int = 4, seed: int = 0,
                 cell_px: int = 24, quiet_cells: int = 1) -> np.ndarray:
    """Render a grayscale image of one or more markers.

    ``markers`` is an iterable of (marker_id, R, t, side). Rendering is done at
    ``supersample`` x resolution and area-averaged down for anti-aliased edges,
    so the corner positions are faithful to the pinhole projection.
    """
    w, h = size
    s = supersample
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h * s, 0:w * s]
    canvas = (150 + 50 * (xx / (w * s)) - 30 * (yy / (h * s))).astype(np.float32)
    for mid, R, t, side in markers:
        tex = dictionary.image(mid, cell_px, quiet_cells).astype(np.float32)
        tex_px = tex.shape[0]
        span = side * (CELLS + 2 * quiet_cells) / CELLS
        # texture pixel centre j sits at plane coordinate -span/2 + (j + 0.5) * span / tex_px
        A = np.array([[span / tex_px, 0, -span / 2 + 0.5 * span / tex_px],
                      [0, -span / tex_px, span / 2 - 0.5 * span / tex_px],
                      [0, 0, 1]], float)
        P = K @ np.column_stack([R[:, 0], R[:, 1], t])     # plane (X, Y, 1) -> image
        # full-res pixel centre u covers supersampled centres s*u .. s*u+s-1 (mean s*u+(s-1)/2)
        S = np.array([[s, 0, (s - 1) / 2], [0, s, (s - 1) / 2], [0, 0, 1.0]])
        H = S @ P @ A
        dsize = (w * s, h * s)
        warped = cv2.warpPerspective(tex, H, dsize, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpPerspective(np.ones_like(tex), H, dsize, flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # linear composite; ink is not pure black nor paper pure white
        canvas = canvas * (1 - mask) + (0.85 * warped + 20 * mask)
    img = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    img = img + rng.normal(0, noise_sigma, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
