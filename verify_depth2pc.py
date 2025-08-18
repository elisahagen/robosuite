import numpy as np 
import open3d as o3d
import json
import cv2
import logger

def convert_depth_buffer_to_meters(depth_buffer, near=0.01, far=10.0):
    z_n = depth_buffer
    z_e = 2.0 * z_n - 1.0  # Convert [0,1] to [-1,1] (NDC space)
    depth = (2.0 * near * far) / (far + near - z_e * (far - near))
    print(depth)
    return depth

def generate_point_clouds_bacth(depth_path, intrinsics_json, depth_scale=1000.0) -> np.ndarray:
    """
    Generate batched 3D point clouds from depth maps and intrinsics.

    Args:
        depth_batch (np.ndarray): Depth maps (B, H, W) or (B, 1, H, W)
        K (dict): Camera intrinsics, 3x3 matrix
        depth_scale (float): Scaling factor from depth units to meters

    Returns:
        np.ndarray: Point cloud batch (B, H*W, 3)
    """
    visualize = True 
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    print(depth.min(), depth.max())
    print(depth)
    if depth is None:
        logger.error(f"Could not load depth image from {depth_path}")
        raise FileNotFoundError(f"Could not load depth image from {depth_path}")
    depth = depth.astype(np.float32) / depth_scale  # Convert to meters

    try:
        with open(intrinsics_json) as f:
            intr = json.load(f)["intrinsics"]["intrinsic_matrix_K"]
            
        if isinstance(intr[0], list):
            intr = [elem for row in intr for elem in row]
        fx, _, cx, _, fy, cy = intr[:6]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        logger.error(f"Error loading intrinsics: {str(e)}")
        raise ValueError(f"Error loading intrinsics: {str(e)}")

    # if depth_batch.ndim == 4:
    #     depth_batch = depth_batch[:, 0]
    # depth_batch = convert_depth_buffer_to_meters(depth)
    # print(depth, depth.shape)
    H, W = depth.shape
    # K = np.asarray(K)
    # if K.shape == (9,):
    #     K = K.reshape(3, 3)
    # assert K.shape == (3, 3), f"K must be a 3x3 matrix, got {K.shape}"
    # fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Generate coordinate grid
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    # # Broadcast coordinates to batch
    # uu_batch = np.repeat(uu[None, :, :], B, axis=0)
    # vv_batch = np.repeat(vv[None, :, :], B, axis=0)

    z = depth / 1000.0
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    # Stack and reshape
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    if visualize:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")


    # Optionally visualize the point clouds
    # visualize_point_clouds(points)
    return points


def get_cropped_pointcloud_from_depth_only(depth_path=None, intrinsics_json=None):
        """Generate a cropped point cloud from depth image and camera parameters."""
        print("Generating cropped point cloud from depth image")
        # Use provided params or instance params
        depth_path = depth_path #or self.depth_path
        intrinsics_json = intrinsics_json #or self.intrinsics_json
        #bbox_params = bbox_params #or self.bbox_params
        if not all([depth_path, intrinsics_json]):
            logger.error("Missing required parameters for point cloud generation")
            raise ValueError("Missing required parameters for point cloud generation")
        # Load depth and intrinsics
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        print(depth)
        if depth is None:
            logger.error(f"Could not load depth image from {depth_path}")
            raise FileNotFoundError(f"Could not load depth image from {depth_path}")
        depth = depth.astype(np.float32) / 1000.0  # Convert to meters
        try:
            with open(intrinsics_json) as f:
                
                intr = json.load(f)["intrinsics"]["intrinsic_matrix_K"]
            
            if isinstance(intr[0], list):
                intr = [elem for row in intr for elem in row]
            fx, _, cx, _, fy, cy = intr[:6]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error loading intrinsics: {str(e)}")
            raise ValueError(f"Error loading intrinsics: {str(e)}")
        # Create coordinate arrays using broadcasting
        h, w = depth.shape
        x = np.arange(w)
        y = np.arange(h)
        x_coords = (x[None, :] - cx) / fx
        y_coords = (y[:, None] - cy) / fy
        # Compute 3D coordinates efficiently
        points = np.stack([
            x_coords * depth,
            y_coords * depth,
            depth
        ], axis=-1).reshape(-1, 3)
        # Filter invalid points
        valid_mask = depth.reshape(-1) > 0
        points = points[valid_mask]
        # Create point cloud and bounding box
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    generate_point_clouds_bacth(depth_path="/home/elisa/Documents/data/robosuite_automated/smooth1/3/teleop_dataset_1_20250801_103331/left_side_view_depth/00003.png", intrinsics_json="/home/elisa/Documents/data/robosuite_automated/smooth1/3/teleop_dataset_1_20250801_103331/left_side_view_camera_info.json") # , bbox_params={"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100, "msg": ""}, visualize=True)