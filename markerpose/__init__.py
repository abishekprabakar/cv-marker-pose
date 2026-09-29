"""Square fiducial markers: procedural rendering, detection and 6-DoF pose."""
from .marker import MarkerDictionary, marker_corners_3d, render_scene, project_points
from .detect import detect_markers, Detection
from .pose import (estimate_pose, homography_dlt, pose_from_homography,
                   refine_pose_lm, rodrigues, rodrigues_inv, rotation_error_deg,
                   reprojection_rms)

__all__ = [
    "MarkerDictionary", "marker_corners_3d", "render_scene", "project_points",
    "detect_markers", "Detection", "estimate_pose", "homography_dlt",
    "pose_from_homography", "refine_pose_lm", "rodrigues", "rodrigues_inv",
    "rotation_error_deg", "reprojection_rms",
]
