import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
import spatialmath as sm
import cv2
import mujoco
from graspnetAPI import GraspGroup

from utils.path_setup import setup_graspnet_path
from utils.config import load_config
from utils.data_loader import get_and_process_data, build_sim_camera
from utils.graspnet_runner import load_graspnet

setup_graspnet_path()
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv
from cv_process import segment_image

_cfg = load_config('sim_ur5.yaml')
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
    COLLISION_THRESH = 0.01
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
        print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        filtered = all_grasps
    # else:
        print(f"\nFiltered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")

    # 对过滤后的抓取根据 score 排序（降序）
    filtered.sort(key=lambda g: g.score, reverse=True)

    # 取前20个抓取（如果少于20个，则全部使用）
    top_grasps = filtered[:20]
    # top_grasps = filtered[:1]

    # 可视化过滤后的抓取，手动转换为 Open3D 物体
    grippers = [g.to_open3d_geometry() for g in top_grasps]
    # print(f"\nVisualizing top {len(top_grasps)} grasps after vertical filtering...")
    # o3d.visualization.draw_geometries([cloud, *grippers])
    # for gripper in grippers:
    #     o3d.visualization.draw_geometries([cloud, gripper])
    
    # 选择得分最高的抓取（filtered 列表已按得分降序排序）
    best_grasp = top_grasps[0]
    best_translation = best_grasp.translation
    best_rotation = best_grasp.rotation_matrix
    best_width = best_grasp.width

    # 创建一个新的 GraspGroup 并添加最佳抓取
    new_gg = GraspGroup()            # 初始化空的 GraspGroup
    new_gg.add(best_grasp)           # 添加最佳抓取
    if visual:
        grippers = new_gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
    return new_gg
    # return best_translation, best_rotation, best_width


# ================= 仿真执行抓取动作 ====================
def execute_grasp(env, gg):
    """
    执行抓取动作，控制机器人从初始位置移动到抓取位置，并完成抓取操作。

    参数:
    env (UR5GraspEnv): 机器人环境对象。
    gg (GraspGroup): 抓取预测结果。
    """
    robot = env.robot
    T_wb = robot.base

    # 0.初始准备阶段
    # 目标：计算抓取位姿 T_wo（物体相对于世界坐标系的位姿）
    # n_wc = np.array([0.0, -1.0, 0.0]) # 相机朝向
    # o_wc = np.array([-1.0, 0.0, -0.5]) # 相机朝向 [0.5, 0.0, -1.0] -> [-1.0, 0.0, -0.5]
    # t_wc = np.array([1.0, 0.6, 2.0]) # 相机的位置。2.0是相机高度，与scene.xml中保持一致。
    n_wc = np.array([0.0, -1.0, 0.0]) 
    o_wc = np.array([-1.0, 0.0, -0.5]) 
    t_wc = np.array([0.85, 0.8, 1.6]) 

    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))
    T_wo = T_wc * T_co

    action = np.zeros(7)

    # 1.机器人运动到预抓取位姿
    # 目标：将机器人从当前位置移动到预抓取姿态（q1）
    time1 = 1
    q0 = robot.get_joint()
    q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time1)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner1 = TrajectoryPlanner(trajectory_parameter0)
    # 执行planner_array = [planner1]
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
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 2.接近抓取位姿
    # 目标：从预抓取位姿直线移动到抓取点附近（T2）
    # 关键点：T2 是 T_wo 沿负 x 方向偏移 0.1m，确保安全接近物体。
    time2 = 1
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)
    position_parameter1 = LinePositionParameter(T1.t, T2.t) #  位置规划（直线路径）
    attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R)) # 姿态规划（插值旋转）
    cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1) # 组合笛卡尔参数
    velocity_parameter1 = QuinticVelocityParameter(time2) # 速度曲线（五次多项式插值）
    trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1) # 将笛卡尔空间路径和速度曲线结合，生成完整的轨迹参数
    planner2 = TrajectoryPlanner(trajectory_parameter1) # 轨迹规划器，将笛卡尔空间路径和速度曲线结合，生成完整的轨迹参数
    # 执行planner_array = [planner2]
    time_array = [0.0, time2]
    planner_array = [planner2]
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
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 3.执行抓取
    # 目标：从 T2 移动到 T3（精确抓取位姿）。通过逐步增加 action[-1]（夹爪控制信号）闭合夹爪，抓取物体。
    time3 = 1
    T3 = T_wo
    position_parameter2 = LinePositionParameter(T2.t, T3.t)
    attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
    cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
    velocity_parameter2 = QuinticVelocityParameter(time3)
    trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
    planner3 = TrajectoryPlanner(trajectory_parameter2)
    # 执行planner_array = [planner3]
    time_array = [0.0, time3]
    planner_array = [planner3]
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
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
    for i in range(1000):
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        env.step(action)

    # 4.提起物体
    # 目标：抓取后垂直提升物体（避免碰撞桌面）。
    time4 = 1
    T4 = sm.SE3.Trans(0.0, 0.0, 0.3) * T3 # 通过在T3的基础上向上偏移0.3单位得到的，用于控制机器人上升一定的高度
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time4)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner4 = TrajectoryPlanner(trajectory_parameter3)

    # 5.水平移动物体
    # 目标：将物体水平移动到目标放置位置，保持高度不变。
    time5 = 1
    T5 = sm.SE3.Trans(1.4, 0.3, T4.t[2]) * sm.SE3(sm.SO3(T4.R)) #  通过在T4的基础上进行平移得到，这里的1.4, 0.3是场景中的固定点坐标，而不是偏移量
    position_parameter4 = LinePositionParameter(T4.t, T5.t)
    attitude_parameter4 = OneAttitudeParameter(sm.SO3(T4.R), sm.SO3(T5.R))
    cartesian_parameter4 = CartesianParameter(position_parameter4, attitude_parameter4)
    velocity_parameter4 = QuinticVelocityParameter(time5)
    trajectory_parameter4 = TrajectoryParameter(cartesian_parameter4, velocity_parameter4)
    planner5 = TrajectoryPlanner(trajectory_parameter4)

    # 6.放置物体
    # 目标：垂直下降物体到接触面（T7）。逐步减小 action[-1]（夹爪信号）以释放物体。
    time6 = 1
    T6 = sm.SE3.Trans(0.0, 0.0, -0.1) * T5 # 通过在T5的基础上向下偏移0.1单位得到的，用于控制机器人下降一定的高度
    position_parameter6 = LinePositionParameter(T5.t, T6.t)
    attitude_parameter6 = OneAttitudeParameter(sm.SO3(T5.R), sm.SO3(T6.R))
    cartesian_parameter6 = CartesianParameter(position_parameter6, attitude_parameter6)
    velocity_parameter6 = QuinticVelocityParameter(time6)
    trajectory_parameter6 = TrajectoryParameter(cartesian_parameter6, velocity_parameter6)
    planner6 = TrajectoryPlanner(trajectory_parameter6)

    # 执行planner_array = [planner4, planner5, planner6]
    time_array = [0.0, time4, time5, time6]
    planner_array = [planner4, planner5, planner6]
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
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
    for i in range(1000):
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        env.step(action)

    # 7.抬起夹爪
    # 目标：放置后抬起夹爪，避免碰撞物体。
    time7 = 1
    T7 = sm.SE3.Trans(0.0, 0.0, 0.1) * T6
    position_parameter7 = LinePositionParameter(T6.t, T7.t)
    attitude_parameter7 = OneAttitudeParameter(sm.SO3(T6.R), sm.SO3(T7.R))
    cartesian_parameter7 = CartesianParameter(position_parameter7, attitude_parameter7)
    velocity_parameter7 = QuinticVelocityParameter(time7)
    trajectory_parameter7 = TrajectoryParameter(cartesian_parameter7, velocity_parameter7)
    planner7 = TrajectoryPlanner(trajectory_parameter7)
    # 执行planner_array = [planner7]
    time_array = [0.0, time7]
    planner_array = [planner7]
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
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 8.回到初始位置
    # 目标：机器人返回初始姿态（q0），完成整个任务。
    time8 = 1
    q8 = robot.get_joint()
    q9 = q0
    parameter8 = JointParameter(q8, q9)
    velocity_parameter8 = QuinticVelocityParameter(time8)
    trajectory_parameter8 = TrajectoryParameter(parameter8, velocity_parameter8)
    planner8 = TrajectoryPlanner(trajectory_parameter8)
    # 执行planner_array = [planner8]
    time_array = [0.0, time8]
    planner_array = [planner8]
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
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break


if __name__ == '__main__':
    
    env = UR5GraspEnv()
    env.reset()

    n = 4 # 循环次数，连续抓取物体
    for _ in range(n): 

        for i in range(500): # 1000
            env.step()
        
        # 1. 获取图像和深度图
        imgs = env.render()
        color_img_path = imgs['img'] # MuJoCo 渲染的是 RGB
        depth_img_path = imgs['depth']

        # 将MuJoCo渲染的是RGB转化为OpenCV默认使用BGR颜色空间
        color_img_path = cv2.cvtColor(color_img_path, cv2.COLOR_RGB2BGR)
        # 保存/查看图片
        # cv2.imwrite('color_img_path.jpg', color_img_path)
        # cv2.imshow('color', color_img_path)
        # cv2.waitKey(0)
        
        # 2. SAM分割图像
        mask_img_path = segment_image(color_img_path)

        # 3. 获取物体的点云数据
        end_points, cloud_o3d = get_and_process_data(
            color_img_path, depth_img_path, mask_img_path,
            SIM_CAMERA, _gn_cfg['num_points'], _cam_cfg.get('depth_trunc', 2.0)
        )

        # 4. 获取抓取点对应的夹爪姿态
        gg = generate_grasps(end_points, cloud_o3d, True) # True or False

        # 5. 仿真执行抓取
        execute_grasp(env, gg)

    env.close()
