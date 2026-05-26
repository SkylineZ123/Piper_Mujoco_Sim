import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
from PIL import Image
import spatialmath as sm

from graspnetAPI import GraspGroup

from utils.path_setup import setup_graspnet_path
from utils.config import load_config
from utils.graspnet_runner import load_graspnet

setup_graspnet_path()
from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv

_cfg = load_config('sim_ur5.yaml')
_gn_cfg = _cfg['graspnet']


def get_net():
    return load_graspnet(_gn_cfg['checkpoint'])


def get_and_process_data(imgs):
    num_point = 20000

    color = imgs['img'] / 255.0
    depth = imgs['depth']

    height = 640
    width = 640
    fovy = np.pi / 4
    intrinsic = np.array([
        [height / (2.0 * np.tan(fovy / 2.0)), 0.0, width / 2.0],
        [0.0, height / (2.0 * np.tan(fovy / 2.0)), height / 2.0],
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0

    camera = CameraInfo(height, width, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    mask = depth < 2.0
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud


def get_grasps(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud):
    voxel_size = 0.01
    collision_thresh = 0.01

    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]

    return gg


def vis_grasps(gg, cloud):
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


def generate_grasps(net, imgs, visual=False):
    end_points, cloud = get_and_process_data(imgs)
    gg = get_grasps(net, end_points)
    gg = collision_detection(gg, np.array(cloud.points))
    gg.nms()
    gg.sort_by_score()
    gg = gg[:1]
    if visual:
        vis_grasps(gg, cloud)
    return gg


if __name__ == '__main__':
    net = get_net()

    env = UR5GraspEnv()
    env.reset()
    for i in range(1000):
        env.step()
    imgs = env.render()

    gg = generate_grasps(net, imgs, True)

    robot = env.robot
    T_wb = robot.base
    n_wc = np.array([0.0, -1.0, 0.0])
    o_wc = np.array([-1.0, 0.0, -0.5])
    t_wc = np.array([1.0, 0.6, 2.0])
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(
        sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))

    T_wo = T_wc * T_co

    time0 = 2
    q0 = robot.get_joint()
    q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])

    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time0)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner0 = TrajectoryPlanner(trajectory_parameter0)

    time1 = 2
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)
    position_parameter1 = LinePositionParameter(T1.t, T2.t)
    attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R))
    cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1)
    velocity_parameter1 = QuinticVelocityParameter(time1)
    trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1)
    planner1 = TrajectoryPlanner(trajectory_parameter1)

    time2 = 2
    T3 = T_wo
    position_parameter2 = LinePositionParameter(T2.t, T3.t)
    attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
    cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
    velocity_parameter2 = QuinticVelocityParameter(time2)
    trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
    planner2 = TrajectoryPlanner(trajectory_parameter2)

    time_array = [0, time0, time1, time2]
    planner_array = [planner0, planner1, planner2]
    total_time = np.sum(time_array)

    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0, total_time, time_step_num)

    time_cumsum = np.cumsum(time_array)
    action = np.zeros(7)
    for i, timei in enumerate(times):
        for j in range(len(time_cumsum)):
            if timei < time_cumsum[j]:
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

    for i in range(1500):
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        env.step(action)

    time3 = 2
    T4 = sm.SE3.Trans(0.0, 0.0, 0.1) * T3
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time3)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner3 = TrajectoryPlanner(trajectory_parameter3)

    time4 = 2
    T5 = sm.SE3.Trans(1.4, 0.2, T4.t[2]) * sm.SE3(sm.SO3(T4.R))
    position_parameter4 = LinePositionParameter(T4.t, T5.t)
    attitude_parameter4 = OneAttitudeParameter(sm.SO3(T4.R), sm.SO3(T5.R))
    cartesian_parameter4 = CartesianParameter(position_parameter4, attitude_parameter4)
    velocity_parameter4 = QuinticVelocityParameter(time4)
    trajectory_parameter4 = TrajectoryParameter(cartesian_parameter4, velocity_parameter4)
    planner4 = TrajectoryPlanner(trajectory_parameter4)

    time5 = 2
    T6 = sm.SE3.Trans(0.2, 0.2, T5.t[2]) * sm.SE3(sm.SO3.Rz(-np.pi / 2) * sm.SO3(T5.R))
    position_parameter5 = LinePositionParameter(T5.t, T6.t)
    attitude_parameter5 = OneAttitudeParameter(sm.SO3(T5.R), sm.SO3(T6.R))
    cartesian_parameter5 = CartesianParameter(position_parameter5, attitude_parameter5)
    velocity_parameter5 = QuinticVelocityParameter(time5)
    trajectory_parameter5 = TrajectoryParameter(cartesian_parameter5, velocity_parameter5)
    planner5 = TrajectoryPlanner(trajectory_parameter5)

    time6 = 2
    T7 = sm.SE3.Trans(0.0, 0.0, -0.1) * T6
    position_parameter6 = LinePositionParameter(T6.t, T7.t)
    attitude_parameter6 = OneAttitudeParameter(sm.SO3(T6.R), sm.SO3(T7.R))
    cartesian_parameter6 = CartesianParameter(position_parameter6, attitude_parameter6)
    velocity_parameter6 = QuinticVelocityParameter(time6)
    trajectory_parameter6 = TrajectoryParameter(cartesian_parameter6, velocity_parameter6)
    planner6 = TrajectoryPlanner(trajectory_parameter6)

    time_array = [0.0, time3, time4, time5, time6]
    planner_array = [planner3, planner4, planner5, planner6]
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

    for i in range(1500):
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        env.step(action)

    time7 = 2
    T8 = sm.SE3.Trans(0.0, 0.0, 0.2) * T7
    position_parameter7 = LinePositionParameter(T7.t, T8.t)
    attitude_parameter7 = OneAttitudeParameter(sm.SO3(T7.R), sm.SO3(T8.R))
    cartesian_parameter7 = CartesianParameter(position_parameter7, attitude_parameter7)
    velocity_parameter7 = QuinticVelocityParameter(time7)
    trajectory_parameter7 = TrajectoryParameter(cartesian_parameter7, velocity_parameter7)
    planner7 = TrajectoryPlanner(trajectory_parameter7)

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

    time8 = 2.0
    q8 = robot.get_joint()
    q9 = q0

    parameter8 = JointParameter(q8, q9)
    velocity_parameter8 = QuinticVelocityParameter(time8)
    trajectory_parameter8 = TrajectoryParameter(parameter8, velocity_parameter8)
    planner8 = TrajectoryPlanner(trajectory_parameter8)

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

    for i in range(2000):
        env.step()
