import os
import json
import openai
import os.path
import sys
import cv2
import time
import ast
import re
import pyrealsense2 as rs
import numpy as np
import torch
import copy
from graspnetAPI import GraspGroup
import open3d as o3d
from manipulator_grasp.real import piper_grasp_real
from PIL import Image
import spatialmath as sm
from dotenv import load_dotenv

#sim mujoco
from manipulator_grasp.env import piper_grasp_env

from utils.path_setup import setup_graspnet_path
from utils.config import load_config

setup_graspnet_path()
from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
import cv_process
from manipulator_grasp.arm.motion_planning import *

load_dotenv()

_real_cfg = load_config('real_piper.yaml')
_data_root = os.environ.get('DATA_DIR', 'data')
_img_dirs = _real_cfg.get('image_dirs', {})
depth_color_dir = os.path.join(_data_root, _img_dirs.get('depth_map', 'image/depth_map'))
depth_dir = os.path.join(_data_root, _img_dirs.get('depth', 'image/depth'))
rgb_dir = os.path.join(_data_root, _img_dirs.get('rgb', 'image/rgb'))
mask_dir = os.path.join(_data_root, _img_dirs.get('mask', 'image/mask'))

qwen_api_key = os.environ.get('DASHSCOPE_API_KEY')
qwen_api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY')
deepseek_api_url = 'https://api.deepseek.com'

if not qwen_api_key:
    raise EnvironmentError("DASHSCOPE_API_KEY is not set. Copy .env.example to .env and fill in your key.")


# COLLISION_THRESH = 0.01
# VOXEL_SIZE = 0.01
class TerminalInterface:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=qwen_api_key,
            base_url=qwen_api_url,
        )

    def generate_prompt(self, text_command):
        prompt = '''
            你是我的机械臂助手，机械臂内置了一些函数，请你根据我的指令，以json形式输出要运行的对应函数和你给我的回复
            【以下是所有内置函数介绍】
            机械臂位置归零，所有关节回到原点:back_zero()
            打开夹抓:gripper_control(True)
            关闭夹抓:gripper_control(False)
            读取当前场景下realsenseD435i摄像头的RGB和深度图:get_frame()
            采用yolo-world+SAM方法寻找某个物体的表面mask:Find_obj_Segment(class_name:str)
            采用GraspNet的方法生成物体表面的6D抓取位姿:Grasp_Pose_Inference()
            控制机械臂移动到抓取点:move_to_grasp()
            控制机械臂机械臂移动到放置点:move_to_place(grasp,detect_bbox),这里主要是想保证放置的过程中旋转姿态不变,只需要根据detect_bbox的x和y中心坐标点确定放置位置，z坐标固定
            移动到指定X,Y,Z坐标，比如移动到X坐标-0.02m，Y坐标0.255m,Z坐标0.2496m，单位是米：Publish_Piper_Pose_XYZ(X:float=-0.02,Y:float=0.255,Z:float=0.2496)
            移动到指定X,Y,Z,Rx,Ry,Rz坐标，比如移动到X坐标-0.02m，Y坐标0.255m，Z坐标0.2496m，Rx坐标178度，Ry坐标-0.613度，Rz坐标-96.05度,单位是角度和米:Publish_Piper_Pose_XYZRPY(X=-0.02,Y=0.255,Z=0.2496, Roll=-178, Pitch=-0.613, Yaw=-96.05)
            调用VLM分析场景信息:Scence_Analyze_API_Call(prompt:str),这句prompt主要是根据输入的用户指令去确定。
            移动至俯视姿态：move_to_top_view()
            控制关节运动到某个姿态，比如控制joint1为1.66,joint2为1.52,joint3为-1.07,joint4为0.04,joint5为1.21,joint6为0.18(单位是弧度):Publish_Piper_Joint(J1=1.66, J2=1.52, J3=-1.07, J4=0.04, J5=1.21, J6=0.18)
            休息等待，比如等待两秒：time.sleep(2)

           【输出json格式】
            你直接输出json即可，从{开始，不要输出包含```json的开头或结尾
            在'function'键中，输出函数名列表，列表中每个元素都是字符串，代表要运行的函数名称和参数。每个函数既可以单独运行，也可以和其他函数先后运行。列表元素的先后顺序，表示执行函数的先后顺序
            在'response'键中，根据我的指令和你编排的动作，以第一人称输出你回复我的话，不要超过20个字，可以幽默和发散，用上歌词、台词、互联网热梗、名场面。
            直接输出子任务函数的形参，如Find_obj_Segment('red rectangular block'),Find_obj_Segment('blue T shape block')
            
            【以下是一些具体的例子】
            我的指令：回到原点。你输出：{'function':['back_zero()'], 'response':'回家吧，回到最初的起点'}
            我的指令：运动到俯视的状态。你输出：{'function':['move_to_top_view()'],'response':'我来啦~'｝
            我的指令：运动到俯视的状态，等待3秒后拍一张照片,然后拍一张俯视图。你输出：{'function':['move_to_top_view()','time.sleep(3)','top_view_shot()'], 'response':'我要开始拍照了'}
            我的指令：运动到俯视状态,然后打开夹抓。你输出:{'function':['move_to_top_view()','gripper_control(True)'],'response':'指令完成！'}
            我的指令：把蓝色的T形块放到红色矩形块前面。你输出:{'function':['move_to_top_view()','gripper_control(True)','Find_obj_Segment('blue T shape block')','Grasp_Pose_Inference()','move_to_grasp()','move_to_top_view()','Find_obj_Segment('blue T shape block')']}
            我的指令：
            【我现在的指令是】
        '''
        prompt = prompt + '"' + text_command + '"'
        return prompt

    def process_command(self, text_command):
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[{"role": "user", "content": self.generate_prompt(text_command)}]
            )
            raw_response = response.choices[0].message.content.strip()
            start = raw_response.find('{')
            end = raw_response.rfind('}') + 1
            return json.loads(raw_response[start:end])
        except Exception as e:
            print(f"Error processing command: {str(e)}")
            return {"error": str(e)}


# 子任务执行器
class SubTaskExecutor:
    def __init__(self, sim_flag=True):
        # 函数映射表，模拟执行
        self.sim_flag = sim_flag  # 是否为仿真环境
        self.func_map = {
            'back_zero': self.back_zero,
            'gripper_control': self.gripper_control,
            'move_to_top_view': self.move_to_top_view,
            'get_frame': self.get_frame,
            'Find_obj_Segment': self.find_obj_segment,
            'Grasp_Pose_Inference': self.Grasp_Pose_Inference,
            'move_to_grasp':self.move_to_grasp,
        }
        self.rgb = None
        self.depth = None
        self.mask_obj = None
        self.mask_obj_center = None
        self.frame_count = 0
        self.SAM_predictor = cv_process.choose_model()
        self.gripper_maxW = 0.038
        self.Graspnet = GraspNet(input_feature_dim=0,
                                 num_view=300,
                                 num_angle=12,
                                 num_depth=4,
                                 cylinder_radius=0.05,
                                 hmin=-0.02,
                                 hmax_list=[0.01, 0.02, 0.03, 0.04],
                                 is_training=False)
        self.Graspnet.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        # checkpoint = torch.load('weights/logs/log_rs/checkpoint-rs.tar')  # checkpoint_path
        self.Graspnet.load_state_dict(torch.load('weights/logs/log_rs/checkpoint-rs.tar')['model_state_dict'])
        self.Graspnet.eval()
        self.grasp_Obj_Group = None  # 初始化空的 GraspGroup
        if sim_flag:
            #************simulation****************#
            self.env = piper_grasp_env.PiperGraspEnv()
            self.env.reset()
            height = self.env.render()['img'].shape[0]
            width = self.env.render()['img'].shape[1]
            fovy = np.pi / 4  # 定义的仿真相机
            focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算（基于垂直视场角fovy和高度height）
            c_x = width / 2.0  # 水平中心
            c_y = height / 2.0  # 垂直中心
            self.DEPTH_INTR = {
                "height": height,
                "width": width,
                "ppx": c_x,  # cx
                "ppy": c_y,  # cy
                "fx": focal,  # fx
                "fy": focal  # fy
            }
            self.DEPTH_FACTOR = 1.0  # 深度因子，根据实际数据调整
            # 相机坐标系
            n_wc = np.array([0.0, -1.0, 0.0])
            o_wc = np.array([-1.0, 0.0, -0.5])
            t_wc = np.array([1.15, 0.6, 1.6])
            self.T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))

            # 机器人基坐标系在世界坐标系的位姿
            t_wr = np.array([0.94, 0.6, 0.745])
            n_wr = np.array([1, 0, 0])
            o_wr = np.array([0, 1, 0])
            self.T_wr = sm.SE3.Trans(t_wr) * sm.SE3(sm.SO3.TwoVectors(x=n_wr, y=o_wr))



        else:
            #**************real****************#
            self.D435i_init()
            self.DEPTH_INTR = {
                "height": 480,
                "width": 640,
                "ppx": 319.304,  # cx
                "ppy": 236.915,  # cy
                "fx": 387.897,  # fx
                "fy": 387.897  # fy
            }
            self.DEPTH_FACTOR = 1000.0  # 深度因子，根据实际数据调整
        self.maskcnt = 0

    def D435i_init(self):
        self.D435i_pipeline = rs.pipeline()
        D435i_config = rs.config()
        D435i_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        D435i_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # 启动管道
        profile = self.D435i_pipeline.start(D435i_config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)
        # 深度帧与其他帧对齐
        self.rgbdepth_align = rs.align(rs.stream.color)  # 对齐到彩色图像流
        # 摄像头初始化
        for i in range(1, 30):
            self.get_frame(save_file=False)

    def back_zero(self):
        print("👉 机械臂归零完成")
        if self.sim_flag:
            q0 = self.env.get_joint()
            q1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.gripper_maxW])
            self.env.trajtory_planner(q_init=q0, q_goal=q1, time=2)

    def gripper_control(self, state):
        if self.sim_flag:
            self.env.gripper_control(state)
        print(f"👉 夹抓 {'打开' if state else '关闭'}")

    def move_to_top_view(self):
        print("👉 移动至俯视姿态")
        if self.sim_flag:
            q0 = self.env.get_joint()
            q1 = np.array([0.0, 0.644, -0.62, 0.1, 0.57, 0, self.gripper_maxW])
            self.env.trajtory_planner(q_init=q0 , q_goal = q1, time=2)
    def move_to_grasp(self):
        if self.grasp_Obj_Group is None:
            print("👉 没有生成抓取姿态喔")
            return
        if self.sim_flag:
            # 预抓取动作
            T1 = self.T_bo * sm.SE3(0.0, 0.0, -0.3)
            q0 = self.env.get_joint()
            q_pred_grasp = self.env.Pose_to_Joint_IK(T1.A)
            self.env.trajtory_planner(q_init=q0 , q_goal=q_pred_grasp)
            T2 = self.T_bo * sm.SE3(0.0, 0.0, 0.0)
            q_grasp = self.env.Pose_to_Joint_IK(T2.A)
            self.env.trajtory_planner(q_init=q_pred_grasp , q_goal = q_grasp)

    def get_frame(self, save_file=False):
        print("👉 获取RGB和深度图")
        if self.sim_flag:
            # 1. 获取图像和深度图
            imgs = self.env.render()
            self.rgb = imgs['img']  # MuJoCo 渲染的是 RGB
            self.depth = imgs['depth']
            self.rgb = self.rgb[..., [2, 1, 0]]
        else:

            frames = self.D435i_pipeline.wait_for_frames()
            if not frames:
                self.depth = None
                self.rgb = None
                return None
            aligned_frames = self.rgbdepth_align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                self.depth = None
                self.rgb = None
                return None
            color_image = np.asanyarray(color_frame.get_data())
            self.rgb = color_image[..., [2, 1, 0]]
            self.depth = np.asanyarray(depth_frame.get_data())
        depth_normalized = cv2.normalize(self.depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)
        # 伪彩色映射
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        if save_file:
            colormap_path = os.path.join(depth_color_dir, f"depth_colormap{self.frame_count}.png")
            color_path = os.path.join(rgb_dir, f"color{self.frame_count}.png")
            depth_path = os.path.join(depth_dir, f"depth{self.frame_count}.png")
            cv2.imshow('Color depth', depth_colormap)
            cv2.imwrite(color_path, self.rgb)
            # 将16位的depth图像保存成16位png
            cv2.imwrite(depth_path, self.depth)
            cv2.imwrite(colormap_path, depth_colormap)
            cv2.waitKey(1)

    def find_obj_segment(self, class_name, save_mask=True):
        print("👉 获取物体的bbox和平面mask")
        # 1) 初步检测 - YOLO
        self.detections, vis_image = cv_process.detect_objects_dinoground(self.rgb, class_name)
        self.SAM_predictor.set_image(self.rgb)
        # 5) 判断是否有目标检测结果
        if self.detections:
            # 自动选最高置信度
            best_det = max(self.detections, key=lambda x: x["conf"])
            results = self.SAM_predictor(bboxes=[best_det["xyxy"]])
            self.mask_obj_center, self.mask_obj = cv_process.process_sam_results(results)
            print(f"Auto-selected {best_det['cls']} with confidence {best_det['conf']:.2f}")
        else:
            self.mask_obj = None
        # 6) 保存 mask
        if self.mask_obj is not None and save_mask:
            cv2.imwrite(os.path.join(mask_dir, f'dino_sam{self.maskcnt}.jpg'),
                        self.mask_obj, [cv2.IMWRITE_PNG_BILEVEL, 1])
            self.maskcnt += 1

    def Grasp_Pose_Inference(self):
        """
        主推理流程：
        1. 前向推理（进行抓取预测解码）
        2. 碰撞检测
        3. NMS 去重 + 按置信度/得分排序（降序）
        4. 对抓取预测进行垂直角度筛选
        """
        print("👉 生成物体表面的抓取姿态")
        end_points, cloud_o3d = cv_process.get_and_process_data(self.rgb, self.depth, self.mask_obj, self.DEPTH_INTR,
                                                                self.DEPTH_FACTOR)
        # 1. 前向推理
        with torch.no_grad():
            end_points = self.Graspnet(end_points)
            grasp_preds = pred_decode(end_points)
        # 构造 GraspGroup 对象（这里 gg 是列表或类似列表的对象）
        gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())
        # 2. 碰撞检测
        # if COLLISION_THRESH > 0:
        gg = cv_process.collision_detection(gg, np.asarray(cloud_o3d.points), COLLISION_THRESH=0.05)
        # 3. NMS 去重 + 按照得分排序（降序）
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
        top_grasps = filtered[:]
        # 可视化过滤后的抓取，手动转换为 Open3D 物体
        grippers = [g.to_open3d_geometry() for g in top_grasps]
        print(f"\nVisualizing top {len(top_grasps)} grasps after vertical filtering...")

        o3d.visualization.draw_geometries([cloud_o3d, *grippers])

        # 选择得分最高的抓取（filtered 列表已按得分降序排序）
        best_grasp = top_grasps[0]
        self.grasp_Obj_Group = GraspGroup()  # 初始化空的 GraspGroup
        self.grasp_Obj_Group.add(best_grasp)  # 添加最佳抓取
        self.T_co = sm.SE3.Trans(self.grasp_Obj_Group.translations[0]) * sm.SE3(
            sm.SO3.TwoVectors(x=self.grasp_Obj_Group.rotation_matrices[0][:, 0],
                              y=self.grasp_Obj_Group.rotation_matrices[0][:, 1]))

        self.T_bo = self.T_wr.inv() * self.T_wc * self.T_co * sm.SE3.Rz(np.pi / 2) * sm.SE3.Rx(np.pi / 2) * sm.SE3.Rz(-np.pi / 2)

    def execute_function(self, func_str):
        # 特判 sleep
        if func_str.startswith('time.sleep'):
            try:
                seconds = float(func_str[len('time.sleep('):-1])
                print(f"👉 等待{seconds}秒")
                time.sleep(seconds)
            except:
                print("⚠️ 等待时间解析错误")
            return
        # 匹配函数名和参数
        match = re.match(r'(\w+)\((.*)\)', func_str)
        if not match:
            print(f"⚠️ 无法识别函数调用: {func_str}")
            return
        func_name, arg_str = match.groups()
        func = self.func_map.get(func_name)
        if not func:
            print(f"⚠️ 未知函数: {func_name}")
            return
        # 解析参数（如果有的话）
        try:
            if not arg_str or arg_str.strip() == '':
                args = []
            else:
                args = ast.literal_eval(f'({arg_str},)')
        except Exception as e:
            print(f"⚠️ 参数解析失败: {e}")
            return
        # 执行函数
        try:
            func(*args)
        except Exception as e:
            print(f"⚠️ 调用 {func_name} 失败: {e}")


def main():
    interface = TerminalInterface()
    executor = SubTaskExecutor()
    executor.get_frame()
    executor.find_obj_segment(class_name='yellow rectangular block')
    executor.Grasp_Pose_Inference()
    while True:
        try:
            cmd = input("\n请输入指令 (输入 'exit' 退出): ").strip()
            if cmd.lower() == 'exit':
                break
            if not cmd:
                continue
            result = interface.process_command(cmd)
            if "error" in result:
                print("❌ LLM调用异常:", result["error"])
                continue
            print("\n模型响应:")
            print(json.dumps(result, indent=2))
            # 执行function列表
            print("\n🎬 执行子任务:")
            for func_str in result.get("function", []):
                executor.execute_function(func_str)
                time.sleep(2)
            print(f"\n💬 模型回复: {result.get('response', '')}")
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
