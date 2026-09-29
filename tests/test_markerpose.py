"""Tests with expected values from geometry, closed forms, or an independent solver (OpenCV)."""
import itertools
import unittest

import cv2
import numpy as np

from markerpose import (MarkerDictionary, detect_markers, estimate_pose, homography_dlt,
                        marker_corners_3d, pose_from_homography, project_points,
                        refine_pose_lm, render_scene, reprojection_rms, rodrigues,
                        rodrigues_inv, rotation_error_deg)
from markerpose.marker import BITS

K = np.array([[800.0, 0, 319.5], [0, 800.0, 239.5], [0, 0, 1]])
DIC = MarkerDictionary()


class TestRotations(unittest.TestCase):
    def test_quarter_turn_about_z_by_hand(self):
        R = rodrigues([0, 0, np.pi / 2])
        np.testing.assert_allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)

    def test_matches_opencv_rodrigues(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            axis = rng.normal(size=3)
            v = axis / np.linalg.norm(axis) * rng.uniform(0, np.pi - 1e-3)
            np.testing.assert_allclose(rodrigues(v), cv2.Rodrigues(v)[0], atol=1e-12)

    def test_log_map_round_trip_including_near_pi_and_zero(self):
        for v in ([1e-10, 0, 0], [0.3, -0.2, 0.1], [0, np.pi - 1e-6, 0],
                  [np.pi / np.sqrt(2) - 1e-7, np.pi / np.sqrt(2) - 1e-7, 0]):
            v = np.array(v, float)
            np.testing.assert_allclose(rodrigues_inv(rodrigues(v)), v, atol=1e-6)

    def test_geodesic_error(self):
        self.assertAlmostEqual(rotation_error_deg(np.eye(3), rodrigues([0, 0, 0.3])),
                               np.degrees(0.3), places=10)


class TestHomography(unittest.TestCase):
    def test_similarity_by_hand(self):
        # unit square scaled by 2 and shifted by (10, 20): H = [[2,0,10],[0,2,20],[0,0,1]]
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        dst = src * 2 + [10, 20]
        np.testing.assert_allclose(homography_dlt(src, dst),
                                   [[2, 0, 10], [0, 2, 20], [0, 0, 1]], atol=1e-10)

    def test_recovers_general_projective_map(self):
        H = np.array([[1.2, 0.1, 30], [-0.05, 0.9, 12], [1e-3, -2e-3, 1]])
        src = np.array([[0, 0], [100, 0], [100, 80], [0, 80], [50, 40]], float)
        d = np.column_stack([src, np.ones(5)]) @ H.T
        dst = d[:, :2] / d[:, 2:]
        np.testing.assert_allclose(homography_dlt(src, dst), H, rtol=1e-8, atol=1e-10)


class TestPose(unittest.TestCase):
    def test_frontal_pose_closed_form(self):
        # 10 cm marker, 0.5 m straight ahead: half-side 0.05 m -> 800*0.05/0.5 = 80 px
        c = np.array([[319.5 - 80, 239.5 - 80], [319.5 + 80, 239.5 - 80],
                      [319.5 + 80, 239.5 + 80], [319.5 - 80, 239.5 + 80]])
        R_front = np.diag([1.0, -1.0, -1.0])
        np.testing.assert_allclose(
            project_points(K, R_front, np.array([0, 0, 0.5]), marker_corners_3d(0.1)), c, atol=1e-9)
        R, t = estimate_pose(K, c, 0.1)
        np.testing.assert_allclose(R, R_front, atol=1e-9)
        np.testing.assert_allclose(t, [0, 0, 0.5], atol=1e-10)

    def test_homography_decomposition_exact_on_noise_free_corners(self):
        rng = np.random.default_rng(3)
        X = marker_corners_3d(0.15)
        for _ in range(25):
            R_gt = rodrigues([np.pi, 0, 0]) @ rodrigues(rng.uniform(-0.8, 0.8, 3))
            t_gt = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.4, 1.5)])
            uv = project_points(K, R_gt, t_gt, X)
            R, t = pose_from_homography(homography_dlt(X[:, :2], uv), K)
            self.assertLess(rotation_error_deg(R, R_gt), 1e-6)
            np.testing.assert_allclose(t, t_gt, atol=1e-9)

    def test_lm_converges_from_perturbed_start(self):
        X = marker_corners_3d(0.1)
        R_gt = rodrigues([2.7, 0.4, -0.3])
        t_gt = np.array([0.03, -0.02, 0.7])
        uv = project_points(K, R_gt, t_gt, X)
        R0 = rodrigues(np.radians([4, -3, 2])) @ R_gt
        R, t = refine_pose_lm(K, R0, t_gt + [0.02, -0.01, 0.03], X, uv)
        self.assertLess(rotation_error_deg(R, R_gt), 1e-6)
        np.testing.assert_allclose(t, t_gt, atol=1e-9)
        self.assertLess(reprojection_rms(K, R, t, X, uv), 1e-7)

    def test_matches_opencv_on_noisy_corners(self):
        """Independent solver: OpenCV IPPE_SQUARE + LM minimises the same reprojection cost."""
        rng = np.random.default_rng(11)
        X = marker_corners_3d(0.1)
        for _ in range(20):
            R_gt = rodrigues([np.pi, 0, 0]) @ rodrigues(rng.uniform(-0.6, 0.6, 3))
            t_gt = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.3, 1.0)])
            uv = project_points(K, R_gt, t_gt, X) + rng.normal(0, 0.3, (4, 2))
            R, t = estimate_pose(K, uv, 0.1)
            ok, rv, tv = cv2.solvePnP(X, uv, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            rv, tv = cv2.solvePnPRefineLM(X, uv, K, None, rv, tv)
            Rcv, tcv = cv2.Rodrigues(rv)[0], tv.ravel()
            # compare on cost: ours must be at least as good as OpenCV's optimum
            self.assertLessEqual(reprojection_rms(K, R, t, X, uv),
                                 reprojection_rms(K, Rcv, tcv, X, uv) + 1e-6)
            self.assertLess(rotation_error_deg(R, Rcv), 0.05)
            self.assertLess(np.linalg.norm(t - tcv), 1e-4)

    def test_refinement_never_increases_reprojection_error(self):
        rng = np.random.default_rng(5)
        X = marker_corners_3d(0.1)
        R_gt, t_gt = rodrigues([2.5, 0.6, 0.2]), np.array([0.0, 0.0, 0.8])
        for _ in range(20):
            uv = project_points(K, R_gt, t_gt, X) + rng.normal(0, 0.5, (4, 2))
            Ra, ta = estimate_pose(K, uv, 0.1, refine=False)
            Rb, tb = estimate_pose(K, uv, 0.1, refine=True)
            self.assertLessEqual(reprojection_rms(K, Rb, tb, X, uv),
                                 reprojection_rms(K, Ra, ta, X, uv) + 1e-12)


class TestDictionary(unittest.TestCase):
    def test_min_distance_holds_across_all_rotations(self):
        rots = [[np.rot90(c, k).ravel() for k in range(4)] for c in DIC.codes]
        for i, j in itertools.combinations(range(len(DIC)), 2):
            for a, b in itertools.product(rots[i], rots[j]):
                self.assertGreaterEqual(int(np.sum(a != b)), DIC.min_distance)
        for r in rots:  # orientation must be unambiguous
            for k in (1, 2, 3):
                self.assertGreaterEqual(int(np.sum(r[0] != r[k])), DIC.min_distance)

    def test_identify_rotation_and_single_bit_correction(self):
        for mid in (0, 9, 23):
            for k in range(4):
                bits = np.rot90(DIC.codes[mid], k).copy()
                self.assertEqual(DIC.identify(bits), (mid, k, 0))
                bits[1, 2] ^= 1
                self.assertEqual(DIC.identify(bits), (mid, k, 1))
        self.assertEqual(DIC.codes[0].shape, (BITS, BITS))


class TestDetection(unittest.TestCase):
    def _check(self, mid, R_gt, t_gt, side=0.1, seed=0):
        img = render_scene(K, (640, 480), [(mid, R_gt, t_gt, side)], DIC, seed=seed)
        dets = detect_markers(img, DIC)
        self.assertEqual([d.marker_id for d in dets], [mid])
        gt = project_points(K, R_gt, t_gt, marker_corners_3d(side))
        self.assertLess(np.abs(dets[0].corners - gt).max(), 0.5)  # sub-pixel, correct order
        R, t = estimate_pose(K, dets[0].corners, side)
        self.assertLess(rotation_error_deg(R, R_gt), 1.0)
        self.assertLess(np.linalg.norm(t - t_gt) / np.linalg.norm(t_gt), 0.01)

    def test_all_four_in_plane_orientations(self):
        for k in range(4):
            R = rodrigues([np.pi, 0, 0]) @ rodrigues([0, 0, k * np.pi / 2 + 0.2]) @ rodrigues([0.3, 0.2, 0])
            with self.subTest(k=k):
                self._check(4 + k, R, np.array([0.01, 0.0, 0.55]), seed=k)

    def test_strong_tilt(self):
        R = rodrigues([np.pi, 0, 0]) @ rodrigues([np.radians(55), 0, 0])
        self._check(12, R, np.array([0.0, 0.02, 0.6]))

    def test_multiple_markers_and_no_false_positives(self):
        base = rodrigues([np.pi, 0, 0])
        scene = [(1, base @ rodrigues([0.2, 0.1, 0.3]), np.array([-0.12, 0, 0.7]), 0.08),
                 (2, base @ rodrigues([-0.3, 0.2, -1.0]), np.array([0.12, 0.05, 0.8]), 0.08)]
        img = render_scene(K, (640, 480), scene, DIC, seed=9)
        self.assertEqual([d.marker_id for d in detect_markers(img, DIC)], [1, 2])
        empty = render_scene(K, (640, 480), [], DIC, noise_sigma=6.0, seed=1)
        self.assertEqual(detect_markers(empty, DIC), [])


if __name__ == "__main__":
    unittest.main()
