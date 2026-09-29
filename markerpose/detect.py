"""Marker detection: quad candidates -> bit sampling -> ID + orientation -> sub-pixel corners."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from .marker import CELLS, BITS, MarkerDictionary


@dataclass
class Detection:
    marker_id: int
    corners: np.ndarray    # (4, 2) float, ordered TL, TR, BR, BL of the marker texture
    hamming: int           # bits corrected


def _order_clockwise(q: np.ndarray) -> np.ndarray:
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])  # y-down => ascending = clockwise on screen
    return q[np.argsort(ang)]


def _read_bits(gray: np.ndarray, quad: np.ndarray, cell: int = 10):
    S = CELLS * cell
    dst = np.array([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    patch = cv2.warpPerspective(gray, M, (S, S), flags=cv2.INTER_LINEAR)
    _, bw = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cell // 4  # sample the central half of each cell
    cells = np.zeros((CELLS, CELLS), np.uint8)
    for i in range(CELLS):
        for j in range(CELLS):
            block = bw[i * cell + m:(i + 1) * cell - m, j * cell + m:(j + 1) * cell - m]
            cells[i, j] = 1 if block.mean() > 127 else 0
    border = np.concatenate([cells[0], cells[-1], cells[:, 0], cells[:, -1]])
    return cells[1:-1, 1:-1], int(border.sum())


def detect_markers(gray: np.ndarray, dictionary: MarkerDictionary, min_perimeter: int = 60,
                   max_border_errors: int = 1, subpix: bool = True) -> list[Detection]:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    found: dict[int, Detection] = {}
    areas: dict[int, float] = {}
    for cnt in contours:
        per = cv2.arcLength(cnt, True)
        if per < min_perimeter:
            continue
        approx = cv2.approxPolyDP(cnt, 0.04 * per, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = _order_clockwise(approx.reshape(4, 2).astype(np.float64))
        bits, border_white = _read_bits(gray, quad)
        if border_white > max_border_errors:
            continue
        hit = dictionary.identify(bits)
        if hit is None:
            continue
        mid, k, dist = hit
        # observed = rot90(code, k): texture corner j sits at observed index (j - k) mod 4
        ordered = np.array([quad[(j - k) % 4] for j in range(4)])
        area = cv2.contourArea(approx)
        if mid in found and areas[mid] >= area:
            continue
        found[mid] = Detection(mid, ordered, dist)
        areas[mid] = area
    dets = sorted(found.values(), key=lambda d: d.marker_id)
    if subpix and dets:
        pts = np.concatenate([d.corners for d in dets]).astype(np.float32).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
        pts = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), crit).reshape(-1, 4, 2)
        for d, p in zip(dets, pts):
            d.corners = p.astype(np.float64)
    return dets
