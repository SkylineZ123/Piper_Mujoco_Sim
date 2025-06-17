import os.path
import sys
import cv2
import time
import numpy as np
import torch
import copy
from graspnetAPI import GraspGroup
import open3d as o3d
from manipulator_grasp.real import piper_grasp_real
from PIL import Image
import spatialmath as sm

ROOT_DIR =os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))
from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from cv_process import segment_image
from manipulator_grasp.arm.motion_planning import *


# ================= 数据处理并生成输入 ====================
def get_and_process_data(color_path, depth_path, mask_path):
    """
    根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据
    """
    # 相机内参（使用深度相机参数生成点云，根据实际相机修改）
    DEPTH_INTR = {
        "ppx": 319.304,  # cx
        "ppy": 236.915,  # cy
        "fx": 387.897,  # fx
        "fy": 387.897  # fy
    }
    DEPTH_FACTOR = 1000.0  # 深度因子，根据实际数据调整
    NUM_POINT = 10000
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path.astype(np.float32)
        color /= 255.0
    else:
        raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")

    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_path, str):
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

    # 3. 加载 mask（可能是路径，也可能是数组）
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")

    print("\n=== 尺寸验证 ===")
    print("深度图尺寸:", depth.shape[::-1])
    print("颜色图尺寸:", color.shape[:2][::-1])
    print("工作空间尺寸:", workspace_mask.shape[::-1])
    print("相机参数预设尺寸:", (640, 480))

    camera = CameraInfo(
        width=640,
        height=480,
        fx=DEPTH_INTR['fx'],
        fy=DEPTH_INTR['fy'],
        cx=DEPTH_INTR['ppx'],
        cy=DEPTH_INTR['ppy'],
        scale=DEPTH_FACTOR
    )
    # 利用深度图生成点云 (H,W,3) 并保留组织结构
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    mask = (workspace_mask > 0) & (depth > 0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {'point_clouds': cloud_sampled}
    return end_points, cloud_o3d

# ==================== 碰撞检测 ====================
def collision_detection(gg, cloud_points):
    COLLISION_THRESH = 0.01
    VOXEL_SIZE = 0.01
    mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=VOXEL_SIZE)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
    return gg[~collision_mask]

# ==================== 抓取位姿打印（可选） ====================
def print_grasp_poses(gg):
    print(f"\nTotal grasps after collision detection: {len(gg)}")
    for i, grasp in enumerate(gg):
        print(f"\nGrasp {i + 1}:")
        print(f"Position (x,y,z): {grasp.translation}")
        print(f"Rotation Matrix:\n{grasp.rotation_matrix}")
        print(f"Score: {grasp.score:.4f}")
        print(f"Width: {grasp.width:.4f}")

