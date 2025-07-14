import os.path
import sys
import cv2
import time
import pyrealsense2 as rs
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
depth_color_dir = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/doc/image/depth_map'
depth_dir = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/doc/image/depth'
rgb_dir = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/doc/image/rgb'
mask_dir = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/doc/image/mask'
# 相机内参（使用深度相机参数生成点云，根据实际相机修改）
DEPTH_INTR = {
    "ppx": 321.100341796875,  # cx
    "ppy": 243.82957458496094,  # cy
    "fx": 386.110107421875,  # fx
    "fy": 386.110107421875  # fy
}
DEPTH_FACTOR = 1000.0  # 深度因子，根据实际数据调整
NUM_POINT = 10000
COLLISION_THRESH = 0.01
VOXEL_SIZE = 0.01
# ================= 数据处理并生成输入 ====================
def get_and_process_data(color_path, depth_path, mask_path):
    """
    根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据
    """

    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path.astype(np.float32)
        color = color[..., [2, 1, 0]]
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

# ==================== 主函数：获取抓取预测 ====================
def run_grasp_inference(color_path, depth_path, mask_path):
    """
    主推理流程：
    1. 加载网络
    2. 处理数据并生成输入
    3. 进行抓取预测解码
    4. 碰撞检测
    5. 对抓取预测进行垂直角度筛选（仅保留接近垂直±30°的抓取）
    6. 打印/可视化抓取
    7. 返回前 50 个抓取中得分最高的抓取
    """
    # 1. 加载网络
    net = GraspNet(input_feature_dim=0,
                   num_view=300,
                   num_angle=12,
                   num_depth=4,
                   cylinder_radius=0.05,
                   hmin=-0.02,
                   hmax_list=[0.01, 0.02, 0.03, 0.04],
                   is_training=False)
    net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    checkpoint = torch.load('weights/logs/log_rs/checkpoint-rs.tar') # checkpoint_path
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    # 2. 处理数据
    end_points, cloud_o3d = get_and_process_data(color_path, depth_path, mask_path)
    # 3. 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    # 4. 构造 GraspGroup 对象（这里 gg 是列表或类似列表的对象）
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
    # 5. 碰撞检测
    if COLLISION_THRESH > 0:
        gg = collision_detection(gg, np.asarray(cloud_o3d.points))

    # 6. NMS 去重 + 按照得分排序（降序）
    gg.nms().sort_by_score()

    # ===== 新增筛选部分：对抓取预测的接近方向进行垂直角度限制 =====
    # 将 gg 转换为普通列表
    all_grasps = list(gg)
    vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面）
    angle_threshold = np.deg2rad(30)  # 30度的弧度值
    filtered = []
    for grasp in all_grasps:
        # 抓取的接近方向取 grasp.rotation_matrix 的第三列
        approach_dir = grasp.rotation_matrix[:, 0]
        cos_angle = np.dot(approach_dir, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)
    if len(filtered) == 0:
        print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        return None
    else:
        print(
            f"\n[DEBUG] Filtered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")
    # 对过滤后的抓取根据 score 排序（降序）
    filtered.sort(key=lambda g: g.score, reverse=True)
    # 取前50个抓取（如果少于50个，则全部使用）
    top_grasps = filtered[:1]


    # 可视化过滤后的抓取，手动转换为 Open3D 物体
    grippers = [g.to_open3d_geometry() for g in top_grasps]
    print(f"\nVisualizing top {len(top_grasps)} grasps after vertical filtering...")

    o3d.visualization.draw_geometries([cloud_o3d, *grippers])

    # 选择得分最高的抓取（filtered 列表已按得分降序排序）
    best_grasp = top_grasps[0]
    best_translation = best_grasp.translation
    best_rotation = best_grasp.rotation_matrix
    best_width = best_grasp.width
    new_gg = GraspGroup()  # 初始化空的 GraspGroup
    new_gg.add(best_grasp)  # 添加最佳抓取
    return new_gg


def execute_grasp(piper_inference:piper_grasp_real.PiperGraspReal,gg:GraspGroup):
    """
    执行抓取动作，控制机器人从初始位置移动到抓取位置，并完成抓取操作。

    参数:
        env (UR5GraspEnv): 机器人环境对象。
        gg (GraspGroup): 抓取预测结果。
    """
    # 0.初始准备阶段
    gripper_maxW = 0.068
    #相机坐标系
    n_wc = np.array([0.07516063,0.0120253,-0.99709893])
    o_wc = np.array([0.05083823,-0.99867314,-0.00821214])
    t_wc = np.array([0.22578716,0.00588407,0.62220304])
    # n_wc = np.array([0.2261,0.0984472,-0.96911])
    # o_wc = np.array([0.18168,-0.9816833,-0.05733])
    # t_wc = np.array([0.211349,-0.02055,0.56502463])
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc,y=o_wc))
    T_cc = sm.SE3(sm.SO3.TwoVectors(x=np.array([0 ,-1 ,0]),y=np.array([0 ,0 ,-1])))
    #相机坐标系下抓取姿态的位姿
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))

    # T_bo = T_wr.inv() * T_wc * T_co * sm.SE3.Rz(np.pi / 2) * sm.SE3.Rx(np.pi / 2) * sm.SE3.Rz(-np.pi / 2)
    T_bo = T_wc * T_cc * T_co * sm.SE3.Ry(np.pi / 2)
    piper_inference.gripper_control(True)
    # 目标：将机器人从当前位置移动到预抓取姿态（q1）
    q0 = piper_inference.Get_joint()
    q1 = np.array([0.0, 0.644, -0.62, 0.1, 0.57, 0, gripper_maxW])
    piper_inference.trajtory_planner(q_init=q0,q_goal=q1)

    # #预抓取动作
    # T2 = T_bo * sm.SE3(0, -0.03, -0.3)
    # # T2 = T_bo
    # q2 = piper_inference.Pose_to_Joint_IK(T2.A)
    #
    # piper_inference.trajtory_planner(q_init = q1,q_goal = q2)
    # # 3.执行抓取
    # T3 = T_bo * sm.SE3.Trans(np.array([0.0,-0.03,0.05]))
    # q3 = piper_inference.Pose_to_Joint_IK(T3.A)
    # piper_inference.trajtory_planner(q_init=q2, q_goal=q3)
    # #夹爪闭合
    # q3[-1]=piper_inference.gripper_control(False)
    # # 4.提起物体
    # T4 = T_bo * sm.SE3.Trans(np.array([0.0,-0.0,-0.3]))
    #
    # q4 = piper_inference.Pose_to_Joint_IK(T4.A)
    # piper_inference.trajtory_planner(q_init=q3,q_goal=q4)
    # # 5.水平移动物体
    # T5 = sm.SE3.Trans(0.4,0.4,T4.t[2]) * sm.SE3(sm.SO3(T4.R))
    # # env.set_piper_qpos(q4)
    # q5 = piper_inference.Pose_to_Joint_IK(T5.A)
    # piper_inference.trajtory_planner(q_init=q4,q_goal=q5)
    #
    # # 6.放置物体
    # # 目标：垂直下降物体到接触面（T7）。逐步减小 action[-1]（夹爪信号）以释放物体。
    # T6 = sm.SE3.Trans(0.0, 0.0, -0.1) * T5
    # q6 = piper_inference.Pose_to_Joint_IK(T6.A)
    # piper_inference.trajtory_planner(q_init=q5, q_goal=q6)
    # q6[-1] = piper_inference.gripper_control(True)
    # # 7.预先复位阶段
    # T7 = T5  # 通过在T5的基础上向下偏移0.1单位得到的，用于控制机器人下降一定的高度
    # q7 = piper_inference.Pose_to_Joint_IK(T7.A)
    # piper_inference.trajtory_planner(q_init=q6, q_goal=q7)
    # piper_inference.trajtory_planner(q_init=q7,q_goal=q1)



    #预抓取阶段
    q2 =[0.660482172,1.8000353680000002,-1.291780532,0.12577124,1.086115772,1.2439316400000002,0.0597]
    piper_inference.trajtory_planner(q_init=q1, q_goal=q2)
    #抓取阶段
    q3 = [0.660482172,1.941988188,-1.211102032,0.104193012,0.9212176400000001,1.237058704, 0.05978]
    piper_inference.trajtory_planner(q_init=q2, q_goal=q3)
    piper_inference.gripper_control(False)
    #提
    q4 = [0.660482172,1.75875,-1.394787352,0.102099732, 1.216352676, 1.4054805240000001,0.0597]
    piper_inference.trajtory_planner(q_init=q3, q_goal=q4)
    #移动
    q5 = [-0.36754, 2.0711610080000002,-1.791219696,0.192634092, 1.2209753360000002, 0.69400954, 0.04823]

    piper_inference.trajtory_planner(q_init=q4, q_goal=q5)
    q6 = [-0.36754, 2.122132376, -1.604464232, 0.027945288000000002, 1.0616941720000002, 0.690677736, 0.04823]
    piper_inference.trajtory_planner(q_init=q5, q_goal=q6)
    piper_inference.gripper_control(True)
    piper_inference.trajtory_planner(q_init=q6, q_goal=q5)

    piper_inference.trajtory_planner(q_init=q5, q_goal=q1)
def get__frame_D435I(pipeline:rs.pipeline,count=0,save_file = False):
    """
        获取Realsense当前帧
    """
    frames = pipeline.wait_for_frames()
    if not frames:
        return None,None
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None,None
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = np.uint8(depth_normalized)
    # 伪彩色映射
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    if save_file:
        colormap_path = os.path.join(depth_color_dir, f"depth_colormap{count}.png")
        color_path = os.path.join(rgb_dir, f"color{count}.png")
        depth_path = os.path.join(depth_dir, f"depth{count}.png")
        cv2.imshow('Color depth', depth_colormap)
        cv2.imwrite(color_path, color_image)
        # 将16位的depth图像保存成16位png
        cv2.imwrite(depth_path, depth_image)
        cv2.imwrite(colormap_path, depth_colormap)
        cv2.waitKey(1)
    color_image = color_image[..., [2, 1, 0]]
    return color_image, depth_image

if __name__ == "__main__":
    piper_inference = piper_grasp_real.PiperGraspReal()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # 启动管道
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)
    # 深度帧与其他帧对齐
    align = rs.align(rs.stream.color)  # 对齐到彩色图像流

    #摄像头初始化
    for i in range(1,30):
        rgb,depth = get__frame_D435I(pipeline,save_file=True)
    count_frame = 6
    while True:
        color_img_path,depth_img_path = get__frame_D435I(pipeline,save_file=True,count=count_frame)
        if color_img_path is None or depth_img_path is None:
            continue
        # 将MuJoCo渲染的是RGB转化为OpenCV默认使用BGR颜色空间
        color_img_path = cv2.cvtColor(color_img_path, cv2.COLOR_RGB2BGR)
        # 2. SAM分割图像
        mask_img_path = segment_image(color_img_path,output_mask=f'{mask_dir}/mask_{count_frame}.jpg')
        count_frame+=1
        # 3. 获取抓取点对应的夹爪姿态
        gg = run_grasp_inference(color_img_path, depth_img_path, mask_img_path)
        if gg is None:
            print('no grasps pose')
        execute_grasp(piper_inference, gg)
    # # 目标：将机器人从当前位置移动到预抓取姿态（q1）
    # time1 = 2
    # # 0.初始准备阶段
    # gripper_maxW = 0.038
    # q0 = piper_inference.Get_joint()
    # # q1 = np.array([0.0, 0.0, -0.0, 0.0, 0.0, 0, 0, 0])
    # q1 = np.array([0.0, 0.644, -0.62,0.1, 0.57, 0, gripper_maxW,-gripper_maxW])
    # parameter0 = JointParameter(q0, q1)
    # velocity_parameter0 = QuinticVelocityParameter(time1)
    # trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    # planner1 = TrajectoryPlanner(trajectory_parameter0)
    # time_array = [0.0, time1]
    # planner_array = [planner1]
    # total_time = np.sum(time_array)
    # time_step_num = round(total_time / 0.002) + 1
    # times = np.linspace(0, total_time, time_step_num)
    # time_cumsum = np.cumsum(time_array)
    # for timei in times:
    #     for j in range(len(time_cumsum)):
    #         if timei == 0.0:
    #             break
    #         if timei <= time_cumsum[j]:
    #             planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
    #             if isinstance(planner_interpolate, np.ndarray):
    #                 joint = planner_interpolate
    #                 piper_inference.move_joints_ctrl(joint[:6].tolist())
    #                 time.sleep(0.002)
    #