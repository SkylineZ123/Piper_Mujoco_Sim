import os.path
import sys
import cv2
import time
import numpy as np
import spatialmath as sm
import torch
import copy
from graspnetAPI import GraspGroup
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
from scipy.optimize import least_squares
from manipulator_grasp.env import piper_grasp_env
import pinocchio as pin

from utils.path_setup import setup_graspnet_path
from utils.config import load_config
from utils.data_loader import get_and_process_data, build_sim_camera
from utils.graspnet_runner import load_graspnet

setup_graspnet_path()
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from cv_process import segment_image
from manipulator_grasp.arm.motion_planning import *

_cfg = load_config('sim_piper.yaml')
_cam_cfg = _cfg['camera']
SIM_CAMERA = build_sim_camera(_cam_cfg['width'], _cam_cfg['height'], _cam_cfg['fovy'], _cam_cfg.get('scale', 1.0))
_gn_cfg = _cfg['graspnet']
# =================== 获取抓取预测 ====================
def generate_grasps(end_points, cloud, visual=False):
    """
    主推理流程：
    0. 数据处理并生成输入
    1. 加载网络
    2. 前向推理（进行抓取预测解码）
    3. 碰撞检测
    4. NMS 去重 + 按置信度/得分排序（降序）
    5. 对抓取预测进行垂直角度筛选
    """
    # 1. 加载网络
    net = load_graspnet(_gn_cfg['checkpoint'])
    # 2. 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    # 3. 碰撞检测
    COLLISION_THRESH = 0.05
    if COLLISION_THRESH > 0:
        voxel_size = 0.01
        collision_thresh = 0.01
        mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud.points), voxel_size=voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        gg = gg[~collision_mask]

    # 4. NMS 去重 + 按置信度/得分排序（降序）
    gg.nms().sort_by_score()
    # 5. 返回抓取得分最高的抓取（对抓取预测的接近方向进行垂直角度限制）
    # 将 gg 转换为普通列表
    all_grasps = list(gg)
    vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面） np.array([0, 0, 1])
    angle_threshold = np.deg2rad(30)  # 30度的弧度值 np.deg2rad(30)
    filtered = []
    for grasp in all_grasps:
        # 抓取的接近方向取 grasp.rotation_matrix 的第三列[:, 0]
        approach_dir = grasp.rotation_matrix[:, 0]
        # 计算夹角：cos(angle)=dot(approach_dir, vertical)
        cos_angle = np.dot(approach_dir, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)
    if len(filtered) == 0:
        # print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        return None
    else:
        filtered.sort(key=lambda g: g.score, reverse=True)
        # 取前20个抓取（如果少于20个，则全部使用）
        # top_grasps = filtered[:20]
        # 可视化过滤后的抓取，手动转换为 Open3D 物体
        grippers = [g.to_open3d_geometry() for g in filtered]
        # 选择得分最高的抓取（filtered 列表已按得分降序排序）
        best_grasp = filtered[0]
        best_translation = best_grasp.translation
        best_rotation = best_grasp.rotation_matrix
        best_width = best_grasp.width
        # 创建一个新的 GraspGroup 并添加最佳抓取
        new_gg = GraspGroup()  # 初始化空的 GraspGroup
        new_gg.add(best_grasp)  # 添加最佳抓取
        if visual:
            # grippers = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud, *grippers])
        return new_gg

# ================= 仿真执行抓取动作 ====================
def execute_grasp(env:piper_grasp_env.PiperGraspEnv,gg:GraspGroup):
    """
    执行抓取动作，控制机器人从初始位置移动到抓取位置，并完成抓取操作。

    参数:
        env (UR5GraspEnv): 机器人环境对象。
        gg (GraspGroup): 抓取预测结果。
    """
    # 0.初始准备阶段
    gripper_maxW = 0.038
    # gripper_minW = 0.0
    # 目标：计算抓取位姿 T_wo（物体相对于世界坐标系的位姿）
    n_wc = np.array([0.0, -1.0, 0.0])
    o_wc = np.array([-1.0, 0.0, -0.5])
    t_wc = np.array([1.15,0.6,1.6])
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))

    #机器人基坐标系在世界坐标系的位姿
    t_wr = np.array([0.94,0.6,0.745])
    n_wr =np.array([1,0,0])
    o_wr =np.array([0,1,0])
    T_wr = sm.SE3.Trans(t_wr) * sm.SE3(sm.SO3.TwoVectors(x=n_wr, y=o_wr))
    T_wr_inv = T_wr.inv()
    #T_base^gg=T_world^cam *T_cam^gg*T_base^world
    T_bo = T_wr_inv * T_wc * T_co * sm.SE3.Rz(np.pi / 2) * sm.SE3.Rx(np.pi / 2) * sm.SE3.Rz(-np.pi / 2)

    # 1.机器人运动到预抓取位姿
    # 目标：将机器人从当前位置移动到预抓取姿态（q1）
    time1 = 2
    q0 = env.get_joint()
    # q1 = np.array([0.0,1.2027,-1.326217,0.0,1.131054,0.10894,gripper_maxW])
    q1 = np.array([0.0, 0.644, -0.62,0.1, 0.57, 0, gripper_maxW])
    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time1)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner1 = TrajectoryPlanner(trajectory_parameter0)
    time_array = [0.0, time1]
    planner_array = [planner1]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    env.step(joint)
    env.set_piper_qpos(q1)
    #预抓取动作
    T2 = T_bo * sm.SE3(0.0, 0.0, -0.1)
    env.set_piper_qpos(q1)
    q2=env.Pose_to_Joint_IK(T2.A)
    env.trajtory_planner(q_init=q1,q_goal=q2)
    # 3.执行抓取
    T3 = T_bo
    env.set_piper_qpos(q2)
    q3 = env.Pose_to_Joint_IK(T3.A)
    env.trajtory_planner(q_init=q2,q_goal=q3)
    #夹爪闭合
    q3 = env.gripper_control(False)
    # 4.提起物体
    T4 = T_bo * sm.SE3(0.0, 0.0, -0.3)
    # env.set_piper_qpos(q3)
    q4 =env.Pose_to_Joint_IK(T4.A)
    env.trajtory_planner(q_init=q3,q_goal=q4)
    # 5.水平移动物体
    T5 = sm.SE3.Trans(0.0,-0.4,T4.t[2]) * sm.SE3(sm.SO3(T4.R))
    # env.set_piper_qpos(q4)
    q5 = env.Pose_to_Joint_IK(T5.A)
    env.trajtory_planner(q_init=q4,q_goal=q5)
    # 6.放置物体
    # 目标：垂直下降物体到接触面（T7）。逐步减小 action[-1]（夹爪信号）以释放物体。
    T6 = sm.SE3.Trans(0.0, 0.0, -0.1) * T5  # 通过在T5的基础上向下偏移0.1单位得到的，用于控制机器人下降一定的高度
    q6 = env.Pose_to_Joint_IK(T6.A)
    env.trajtory_planner(q_init=q5,q_goal=q6)
    q6 = env.gripper_control(True)
    # 7.预先复位阶段
    T7 = T5  # 通过在T5的基础上向下偏移0.1单位得到的，用于控制机器人下降一定的高度
    q7 = env.Pose_to_Joint_IK(T7.A)
    env.trajtory_planner(q_init=q6,q_goal=q7)
    #8.动作复位
    env.trajtory_planner(q_init=q7,q_goal=q1)

if __name__ == '__main__':
    env = piper_grasp_env.PiperGraspEnv()
    env.reset()
    for i in range(500):  # 1000
        env.step()
    while True:
        # 1. 获取图像和深度图
        imgs = env.render()
        color_img_path = imgs['img'] # MuJoCo 渲染的是 RGB
        depth_img_path = imgs['depth']
        # 将MuJoCo渲染的是RGB转化为OpenCV默认使用BGR颜色空间
        color_img_path = cv2.cvtColor(color_img_path, cv2.COLOR_RGB2BGR)
        # 2. SAM分割图像
        mask_img_path = segment_image(color_img_path)
        # 3. 获取物体的点云数据
        end_points, cloud_o3d = get_and_process_data(
            color_img_path, depth_img_path, mask_img_path,
            SIM_CAMERA, _gn_cfg['num_points'], _cam_cfg.get('depth_trunc', 2.0)
        )
        # 4. 获取抓取点对应的夹爪姿态
        gg = generate_grasps(end_points, cloud_o3d, False) # True or False
        if gg is None:
            print('no grasps pose')
        else:
            # 5. 仿真执行抓取
            execute_grasp(env, gg)
    env.close()
