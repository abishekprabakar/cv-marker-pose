"""End-to-end demo: render markers with a known camera, detect them, recover 6-DoF pose.

Writes output/detection.png, output/distance_sweep.csv and output/orbit.gif and prints
ground-truth error tables. Everything is procedurally generated and seeded.
"""
from __future__ import annotations

import csv
import os
import sys

import cv2
import numpy as np
from PIL import Image

from markerpose import (MarkerDictionary, detect_markers, estimate_pose, marker_corners_3d,
                        project_points, render_scene, reprojection_rms, rodrigues,
                        rotation_error_deg)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def intrinsics(w, h, f):
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])


def look_at(C, target=np.zeros(3), up=np.array([0, 1.0, 0])):
    """World->camera (R, t) for a camera at C looking at target (OpenCV axes)."""
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ C


def draw_pose(bgr, K, R, t, side, color_scale=1.0):
    L = side * 0.6
    h = side / 2
    axes = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], float)
    p = project_points(K, R, t, axes).round().astype(int)
    for end, col in zip(p[1:], [(0, 0, 255), (0, 200, 0), (255, 80, 0)]):
        cv2.line(bgr, tuple(p[0]), tuple(end), col, 2, cv2.LINE_AA)
    cube = np.array([[x, y, z] for z in (0, side * 0.5) for x, y in
                     [(-h, h), (h, h), (h, -h), (-h, -h)]], float)
    c = project_points(K, R, t, cube).round().astype(int)
    for i in range(4):
        for a, b in [(i, (i + 1) % 4), (i + 4, (i + 1) % 4 + 4), (i, i + 4)]:
            cv2.line(bgr, tuple(c[a]), tuple(c[b]), (0, 220, 255), 1, cv2.LINE_AA)


def opencv_pose(K, corners, side):
    ok, rvec, tvec = cv2.solvePnP(marker_corners_3d(side), corners, K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    rvec, tvec = cv2.solvePnPRefineLM(marker_corners_3d(side), corners, K, None, rvec, tvec)
    return cv2.Rodrigues(rvec)[0], tvec.ravel()


def scene_demo(dic):
    w, h = 1280, 720
    K = intrinsics(w, h, 1000.0)
    side = 0.12
    truth = {
        3: (rodrigues([2.9, 0.25, 0.1]), np.array([-0.30, 0.02, 1.10])),
        11: (rodrigues([2.4, -0.55, 0.9]), np.array([0.05, -0.08, 0.90])),
        17: (rodrigues([-2.6, 0.7, -0.5]), np.array([0.34, 0.10, 1.30])),
    }
    gray = render_scene(K, (w, h), [(m, R, t, side) for m, (R, t) in truth.items()], dic,
                        noise_sigma=2.0, seed=42)
    dets = detect_markers(gray, dic)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    print("Scene: 1280x720, f=1000 px, marker side 12 cm, noise sigma 2 DN")
    print(f"{'id':>3} {'corner err px':>13} {'rot err deg':>11} {'trans err mm':>12} "
          f"{'reproj px':>9} {'vs OpenCV deg':>13} {'vs OpenCV mm':>12}")
    for d in dets:
        R_gt, t_gt = truth[d.marker_id]
        X = marker_corners_3d(side)
        corner_err = np.abs(d.corners - project_points(K, R_gt, t_gt, X)).max()
        R, t = estimate_pose(K, d.corners, side)
        Rcv, tcv = opencv_pose(K, d.corners, side)
        print(f"{d.marker_id:>3} {corner_err:13.3f} {rotation_error_deg(R, R_gt):11.3f} "
              f"{1000 * np.linalg.norm(t - t_gt):12.2f} {reprojection_rms(K, R, t, X, d.corners):9.4f} "
              f"{rotation_error_deg(R, Rcv):13.2e} {1000 * np.linalg.norm(t - tcv):12.2e}")
        draw_pose(bgr, K, R, t, side)
        cv2.polylines(bgr, [d.corners.round().astype(np.int32)], True, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"id {d.marker_id}", tuple(d.corners[0].round().astype(int) + [0, -8]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
    missing = set(truth) - {d.marker_id for d in dets}
    print(f"detected {len(dets)}/{len(truth)} markers" + (f", missed {sorted(missing)}" if missing else ""))
    cv2.imwrite(os.path.join(OUT, "detection.png"), bgr)
    return len(missing) == 0


def distance_sweep(dic, seeds=8):
    w, h = 640, 480
    K = intrinsics(w, h, 700.0)
    side = 0.10
    rows = []
    print("\nDistance sweep: 640x480, f=700 px, 10 cm marker, 40 deg oblique, "
          f"{seeds} noise seeds per distance")
    print(f"{'dist m':>6} {'side px':>7} {'detect':>6} {'med rot deg':>11} {'med trans %':>11}")
    for dist in (0.3, 0.5, 0.8, 1.2, 1.6, 2.0):
        C = dist * np.array([np.sin(np.radians(40)), 0.15, np.cos(np.radians(40))])
        C = C / np.linalg.norm(C) * dist
        R_gt, t_gt = look_at(C)
        rot, trans, hits = [], [], 0
        for s in range(seeds):
            gray = render_scene(K, (w, h), [(7, R_gt, t_gt, side)], dic, noise_sigma=2.0, seed=s)
            dets = [d for d in detect_markers(gray, dic) if d.marker_id == 7]
            if not dets:
                continue
            hits += 1
            R, t = estimate_pose(K, dets[0].corners, side)
            rot.append(rotation_error_deg(R, R_gt))
            trans.append(100 * np.linalg.norm(t - t_gt) / np.linalg.norm(t_gt))
        px = 700 * side / dist
        mr = float(np.median(rot)) if rot else float("nan")
        mt = float(np.median(trans)) if trans else float("nan")
        print(f"{dist:6.1f} {px:7.1f} {hits:>3}/{seeds:<2} {mr:11.3f} {mt:11.3f}")
        rows.append([dist, round(px, 1), hits, seeds, round(mr, 4), round(mt, 4)])
    with open(os.path.join(OUT, "distance_sweep.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["distance_m", "side_px", "detected", "trials",
                                  "median_rot_err_deg", "median_trans_err_pct"]] + rows)


def orbit_gif(dic, frames=36):
    w, h = 400, 300
    K = intrinsics(w, h, 420.0)
    side = 0.10
    imgs, worst = [], 0.0
    for i in range(frames):
        az = 2 * np.pi * i / frames
        el = np.radians(55)
        C = 0.45 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        R_gt, t_gt = look_at(C, up=np.array([0, 0, 1.0]))
        gray = render_scene(K, (w, h), [(0, R_gt, t_gt, side)], dic, noise_sigma=1.5, seed=i)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        dets = [d for d in detect_markers(gray, dic) if d.marker_id == 0]
        if dets:
            R, t = estimate_pose(K, dets[0].corners, side)
            err = rotation_error_deg(R, R_gt)
            worst = max(worst, err)
            draw_pose(bgr, K, R, t, side)
            label = f"rot err {err:.2f} deg"
        else:
            label = "not detected"
        cv2.putText(bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        imgs.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    imgs[0].save(os.path.join(OUT, "orbit.gif"), save_all=True, append_images=imgs[1:],
                 duration=80, loop=0)
    print(f"\nOrbit: {frames} frames at 55 deg elevation, worst rotation error {worst:.3f} deg")


def main():
    os.makedirs(OUT, exist_ok=True)
    dic = MarkerDictionary()
    ok = scene_demo(dic)
    distance_sweep(dic)
    orbit_gif(dic)
    print("\nwrote output/detection.png, output/distance_sweep.csv, output/orbit.gif")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
