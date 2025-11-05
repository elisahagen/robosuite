import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def project_points_from_world_to_camera(points, world_to_camera_transform, camera_height, camera_width):
    """
    Projects 3D world points into camera pixel coordinates.
    Compatible with robosuite's camera conventions.
    """
    assert points.shape[-1] == 3
    assert world_to_camera_transform.shape == (4, 4)

    # Convert to homogeneous coordinates
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)  # (N, 4)
    pixels_h = (world_to_camera_transform @ points_h.T).T  # (N, 4)

    # Normalize by z
    pixels_h = pixels_h / pixels_h[:, 2:3]
    pixels = pixels_h[:, :2]

    # Swap axes for (row, col) and clip to image bounds
    pixels = np.concatenate(
        [
            pixels[:, 1:2].clip(0, camera_height - 1),  # y
            pixels[:, 0:1].clip(0, camera_width - 1),   # x
        ],
        axis=-1,
    ).astype(int)

    return pixels


# ----------- INPUTS -----------

# Keypoints (world coordinates in meters)
left_finger_pos  = np.array([0.06658499951578464, -0.11375939792076632, 1.0233016505852472])
right_finger_pos = np.array([-0.02801352846061666, -0.009485708235080742, 1.0099556345493168])
points_world = np.stack([left_finger_pos, right_finger_pos], axis=0)

# Camera intrinsics
K = np.array([
    [772.5483399593904, 0, 320.0],
    [0, 579.4112549695428, 240.0],
    [0, 0, 1]
])

# Camera extrinsics (from robosuite JSON)
cam_pos = np.array([0.55, 0.28, 1.7579572214102435])
quat_wxyz = np.array([0.15346232983028857, 0.2534970559351796, 0.853285035303346, 0.42906084007122286])

# ✅ Correct quaternion conversion: scipy expects (x, y, z, w)
R_cam = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()

# ✅ robosuite cameras look along -Z → invert the *third row* of rotation (not column)
R_cam[2, :] *= -1

# ----------- Compute world→camera transform -----------
world_to_camera_transform = np.eye(4)
world_to_camera_transform[:3, :3] = R_cam.T
world_to_camera_transform[:3, 3] = -R_cam.T @ cam_pos

# ----------- Combine with intrinsics -----------
projection_matrix = np.eye(4)
projection_matrix[:3, :3] = K @ world_to_camera_transform[:3, :3]
projection_matrix[:3, 3] = K @ world_to_camera_transform[:3, 3]

# ----------- LOAD IMAGE -----------
image_path = "/home/elisa/Documents/data/robosuite_automated/stage2/teleop_dataset_1_20251020_124527/right_side_view/00003.png"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"❌ Could not read image: {image_path}")

H, W = img.shape[:2]

# ----------- PROJECT -----------
pixels = project_points_from_world_to_camera(points_world, projection_matrix, H, W)
print("Projected pixel coords:", pixels)

# ----------- VISUALIZE -----------
for (v, u), color in zip(pixels, [(0, 0, 255), (0, 255, 0)]):  # red=left, green=right
    cv2.circle(img, (u, v), 20, color, -1)

cv2.imshow("Projected Keypoints", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("keypoints_overlay_matrix.png", img)
print("✅ Saved keypoints_overlay_matrix.png")
