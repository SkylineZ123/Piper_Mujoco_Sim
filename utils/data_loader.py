"""Shared point-cloud data loading for sim and real pipelines."""
import numpy as np
import open3d as o3d
import torch
from PIL import Image

from utils.path_setup import setup_graspnet_path

setup_graspnet_path()
from data_utils import CameraInfo, create_point_cloud_from_depth_image  # noqa: E402


def build_sim_camera(width: int, height: int, fovy: float, scale: float = 1.0) -> CameraInfo:
    """Build a CameraInfo for a MuJoCo sim camera given image size and vertical FoV (radians)."""
    focal = height / (2.0 * np.tan(fovy / 2.0))
    return CameraInfo(width, height, focal, focal, width / 2.0, height / 2.0, scale)


def build_real_camera(cfg: dict) -> CameraInfo:
    """Build a CameraInfo from a config dict with keys: width, height, fx, fy, ppx, ppy, scale."""
    return CameraInfo(
        cfg['width'], cfg['height'],
        cfg['fx'], cfg['fy'],
        cfg['ppx'], cfg['ppy'],
        cfg.get('scale', 1000.0),
    )


def get_and_process_data(
    color,
    depth,
    mask,
    camera: CameraInfo,
    num_points: int = 10000,
    depth_trunc: float = None,
    bgr_to_rgb: bool = False,
):
    """Load color/depth/mask (path or ndarray) and return GraspNet input tensors + o3d cloud.

    Args:
        color: file path (str) or BGR/RGB numpy array (H,W,3) uint8.
        depth: file path (str) or numpy array (H,W) uint16/float.
        mask: file path (str) or binary numpy array (H,W).
        camera: CameraInfo built via build_sim_camera or build_real_camera.
        num_points: number of points to sample.
        depth_trunc: if set, points with depth >= depth_trunc are discarded (sim-style).
        bgr_to_rgb: set True when color is a BGR ndarray (e.g. from cv2 / RealSense).
    """
    # Load color
    if isinstance(color, str):
        color_arr = np.array(Image.open(color), dtype=np.float32) / 255.0
    else:
        color_arr = color.astype(np.float32)
        if bgr_to_rgb:
            color_arr = color_arr[..., [2, 1, 0]]
        color_arr /= 255.0

    # Load depth
    if isinstance(depth, str):
        depth_arr = np.array(Image.open(depth))
    else:
        depth_arr = depth

    # Load mask
    if isinstance(mask, str):
        mask_arr = np.array(Image.open(mask))
    else:
        mask_arr = mask

    cloud = create_point_cloud_from_depth_image(depth_arr, camera, organized=True)

    valid_depth = depth_arr > 0
    if depth_trunc is not None:
        valid_depth = valid_depth & (depth_arr < depth_trunc)
    combined_mask = (mask_arr > 0) & valid_depth
    cloud_masked = cloud[combined_mask]
    color_masked = color_arr[combined_mask]

    n = len(cloud_masked)
    if n >= num_points:
        idxs = np.random.choice(n, num_points, replace=False)
    else:
        idxs = np.concatenate([
            np.arange(n),
            np.random.choice(n, num_points - n, replace=True),
        ])

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pts_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {'point_clouds': pts_tensor, 'cloud_colors': color_sampled}
    return end_points, cloud_o3d
