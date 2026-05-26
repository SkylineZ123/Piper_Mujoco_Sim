import os.path
import sys
from manipulator_grasp.arm.motion_planning import *

import time
import numpy as np
import casadi
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
from scipy.optimize import least_squares
from manipulator_grasp.arm.robot import Robot
import pinocchio as pin
# from pinocchio import casadi as cpin
from transformations import quaternion_from_euler, euler_from_quaternion

from pinocchio import casadi as cpin
import threading
ROOT_DIR =os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))
# from manipulator_grasp.arm.motion_planning import *
# from manipulator_grasp.utils import mj
# ---------------
# ---------------
# 逆运动学求解器（使用 Pinocchio）
# ------------------------------
_ENV_DIR = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(_ENV_DIR, '..', 'assets')
class IKSolver:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        # 获取末端执行器的 Frame ID（假设 URDF 中定义了名为 'grasp_link' 的框架）
        self.frame_id = self.model.getFrameId("grasp_link")
        # 关节上下限，用于设置优化约束
        self.lower = self.model.lowerPositionLimit
        self.upper = self.model.upperPositionLimit
        # 创建 CasADi 版的 Pinocchio 模型和数据，用于符号化计算
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.frame_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )
        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.param_tf = self.opti.parameter(4,4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q,self.param_tf))
        self.regularization = casadi.sumsqr(self.var_q)
        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit)
        )
        print("self.reduced_robot.model.lowerPositionLimit:", self.model.lowerPositionLimit)
        print("self.reduced_robot.model.upperPositionLimit:", self.model.upperPositionLimit)
        self.opti.minimize(1000 * self.totalcost + 0.01 * self.regularization)
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 10000,
                'tol': 1e-4
            },
            'print_time': True
        }
        self.opti.solver("ipopt", opts)
    def solve(self, qpos, target_pose:pin.SE3):
        """
        求解逆运动学：
        - qpos: 当前关节角向量 (长度 7 或 8)
        - target_pose: 目标末端姿态，pinocchio.SE3 类型
        返回优化后的 8 维关节角向量 joint_ctrl。
        """
        # 转换为 numpy 数组
        qpos = np.array(qpos).flatten()
        qpos = np.array(qpos).flatten()
        nq=8
        if qpos.size == nq:
            q_init = qpos.copy()
        elif qpos.size == nq - 1:
            q_init = np.zeros(nq)
            q_init[:nq - 1] = qpos
            q_init[-1] = q_init[-2]*-1  # 假设第8维为 prismatic 关节，初始为0
        else:
            raise ValueError(f"qpos 维度应为 {nq} 或 {nq - 1}，但收到 {qpos.size}")
        self.opti.set_initial(self.var_q, q_init)
        # R_casadi = casadi.SX(target_pose.rotation)
        # p_casadi = casadi.SX(target_pose.translation)
        # Target_Pose_casadi = cpin.SE3(R_casadi, p_casadi)
        # target_pose_Cx= casadi.SX(target_pose)
        self.opti.set_value(self.param_tf, casadi.DM(target_pose.homogeneous))
        try:
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

        except RuntimeError as e:
            print("IK 求解失败：", e)
            sol_q = q_init.copy()
        joint_ctrl = np.array(sol_q).flatten()
        return joint_ctrl
        # # 构造目标位姿的 CasADi SE3 常量
    def forward_kinematics(self, q):

        if len(q) == 7 and isinstance(q,list):
            q.append(q[-1]*-1)
            q = np.asarray(q, dtype=np.float64)
        elif isinstance(q,np.ndarray):
            if len(q) == 7:
                q = q.tolist()
                q.append(q[-1]*-1)
                q = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame_id]
class IKOptimizer:
    def __init__(self, urdf_path: str, ee_frame_name: str = 'grasp_link'):
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path,package_dirs=[parent_dir])
        self.model = self.robot.model
        self.collision_model = self.robot.collision_model

        # self.collision_model.addAllCollisionPairs()
        for i in range(4, 9):
            for j in range(0, 3):
                self.collision_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.collision_model)
        self.visual_model = self.robot.visual_model
        self.ee_frame_name = ee_frame_name
        self.ee_frame_id =  self.model.getFrameId(self.ee_frame_name)
        self.data = self.model.createData()

        self.target_translation = np.array([0.155, 0.0, 0.222])  # ✅ 更新初始末端位置



        q = quaternion_from_euler(0, 0, 0)
        self.model.addFrame(
            pin.Frame('ee',
                      self.model.getJointId('joint6'),
                      pin.SE3(
                          # pin.Quaternion(1, 0, 0, 0),
                          pin.Quaternion(q[3], q[0], q[1], q[2]),
                          np.array([0.0, 0.0, 0.1]),
                      ),
                      pin.FrameType.OP_FRAME)
        )#末端姿态
        self.gripper_id = self.model.getFrameId("ee")
        self.q_current = np.zeros(8)  # 初始关节角度
        self.q_current[0]=0.5
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)

        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        self.regularization = casadi.sumsqr(self.var_q)
        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit)
        )
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)
    def objective(self, q, target_pose):
        current_pose = self.forward_kinematics(q)
        error = pin.log(current_pose.inverse() * target_pose)
        return error

    def forward_kinematics(self, q):

        if len(q) == 7 and isinstance(q,list):
            q.append(q[-1]*-1)
            q = np.asarray(q, dtype=np.float64)
        elif isinstance(q,np.ndarray):
            if len(q) == 7:
                q = q.tolist()
                q.append(q[-1]*-1)
                q = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.robot.model, self.data, q)
        pin.updateFramePlacements(self.robot.model, self.data)
        return self.data.oMf[self.ee_frame_id]
    def ik_fun_agelix(self,target_pose,q0):
        if len(q0) == 7 and isinstance(q0,np.ndarray):
            q0 = q0.tolist()
            q0.append(q0[-1]*-1)
            q0 = np.asarray(q0, dtype=np.float64)

        self.init_data = q0

        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_tf, casadi.DM(target_pose.homogeneous))
        try:
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)
            max_diff = max(abs(self.q_current - sol_q))
            if max_diff > 15.0 / 180.0 * 3.1415:
                # print("Excessive changes in joint angle:", max_diff)
                self.init_data = np.zeros(8)
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            return sol_q, '', False
        # self.q_current = sol_q
        return sol_q

    def solve(self, target_pose, q0):
        if len(q0) == 7 and isinstance(q0,np.ndarray):
            q0 = q0.tolist()
            q0.append(q0[-1]*-1)
            q0 = np.asarray(q0, dtype=np.float64)

        lb = np.full_like(q0, -3.14)
        ub = np.full_like(q0, 3.14)
        res = least_squares(self.objective, q0, args=(target_pose,), bounds=(lb, ub))
        return res.x

class PiperGraspEnv:
    def __init__(self):
        # 初始化 MuJoCo 模型和数据
        # filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene_piper.xml')

        # mj_filename = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/Kinova7DoF-MuJoCo/piper_description/mujoco_model/scene_piper.xml'
        # urdf_path = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/Kinova7DoF-MuJoCo/piper_description/urdf/piper_description.urdf'
        mj_filename = os.path.join(_ENV_DIR, '..', 'assets', 'scenes', 'scene_grasp.xml')
        urdf_path = os.path.join(_ENV_DIR, '..', 'assets', 'agilex_piper', 'piper_description', 'urdf', 'piper_description.urdf')
        # self.ik_solver = IKOptimizer(urdf_path, ee_frame_name='grasp_link')
        self.ik_solver = IKSolver(urdf_path)
        self.mj_model = mujoco.MjModel.from_xml_path(mj_filename)
        self.mj_data = mujoco.MjData(self.mj_model)

        # self.sim = MjSim(self.mj_model)
        # self.arm_control = ArmControl(self.sim)

        self.height = 640 # 256 640 720
        self.width = 640 # 256 640 1280
        # 创建两个渲染器实例，分别用于生成彩色图像和深度图
        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        # self.sim_hz = 500
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        # 为了方便观察
        self.mj_viewer.cam.lookat[:] = [1.8, 1.1, 1.7]  # 对应XML中的center
        self.mj_viewer.cam.azimuth = 210      # 对应XML中的azimuth
        self.mj_viewer.cam.elevation = -35    # 对应XML中的elevation
        self.mj_viewer.cam.distance = 1.2     # 根据场景调整的距离值
        self.mj_viewer.sync() # 立即同步更新


    def reset(self):
        self.target_translation = np.array([0.19, 0.0, 0.3])  # ✅ 更新初始末端位置
        self.target_rotation = np.array([  # ✅ 更新初始末端旋转矩阵
            [-0.64925909, -0.03433515, 0.7597919],
            [-0.05494111, 0.99848793, -0.001826],
            [-0.7585, -0.0429297, -0.65016378]
        ])
        self.q_current = np.array([0,1.128,-1.327,0,1.636,0,0])# 初始关节角度
        # self.mj_data.qpos[:8] = self.q_current
        # self.q_current = np.zeros(8)  # 初始关节角度
        # 更新渲染器中的场景数据
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        # 启用深度渲染
        self.mj_depth_renderer.enable_depth_rendering()
        # self.step(action=self.q_current)
    def render_Thread(self):
        while True:
            self.step()
    def render(self):
        '''
        常用于强化学习或机器人控制任务中，提供环境的视觉观测数据。
        '''
        # 更新渲染器中的场景数据
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        # 渲染图像和深度图
        return {
            'img': self.mj_renderer.render(),
            'depth': self.mj_depth_renderer.render()
        }

    def step(self, action=None):
        if action is not None:
            self.q_current = action
            self.mj_data.ctrl = action

        mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_viewer.sync()

    def move_to_position_no_planner(self, *args):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            target_pose_orin = args[0]
            target_translation = target_pose_orin[:3,-1]
            target_rotation = target_pose_orin[:3,:3]
        elif len(args) == 6:
            x, y, z, rx, ry, rz = args
            target_translation = np.array([x, y, z])
            rot_x = R.from_euler('x', rx, degrees=False).as_matrix()
            rot_y = R.from_euler('y', ry, degrees=False).as_matrix()
            rot_z = R.from_euler('z', rz, degrees=False).as_matrix()
            target_rotation = rot_x @ rot_y @ rot_z

        else:
            raise TypeError("Invalid arguments for move_to_position_no_planner")

        target_pose = pin.SE3(target_rotation, target_translation)
        q_new = self.ik_solver.solve(self.q_current,target_pose)[:6]
        # q_new = self.ik_solver.solve(target_pose, self.q_current)[:6]

        self.q_current[:6] = q_new
        self.step(self.q_current)
    def Pose_to_Joint_IK(self,*args):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            target_pose_orin = args[0]
            target_translation = target_pose_orin[:3,-1]
            target_rotation = target_pose_orin[:3,:3]
        elif len(args) == 6:
            x, y, z, rx, ry, rz = args
            target_translation = np.array([x, y, z])
            rot_x = R.from_euler('x', rx, degrees=False).as_matrix()
            rot_y = R.from_euler('y', ry, degrees=False).as_matrix()
            rot_z = R.from_euler('z', rz, degrees=False).as_matrix()
            target_rotation = rot_x @ rot_y @ rot_z
        else:
            raise TypeError("Invalid arguments for move_to_position_no_planner")
        target_pose = pin.SE3(target_rotation, target_translation)
        q_new = self.ik_solver.solve(self.q_current,target_pose)[:6]
        q_out = np.concatenate([q_new,np.array([self.q_current[-1]])],axis = 0)
        return q_out
    def set_piper_qpos(self,q0):
        self.q_current = q0

    def get_joint(self):
        """
            获取当前机械臂关节状态
        """
        return self.mj_data.qpos[:7]
    def get_end_pose(self):
        """
            获取当前末端位姿
        """
        pose = self.ik_solver.forward_kinematics(self.mj_data.qpos[:8])
        print(pose)
    def gripper_control(self,*args):
        if len(args) == 1 and isinstance(args[0], float):
            self.q_current[-1] = args
            self.step(self.q_current)
        elif len(args) == 1 and isinstance(args[0], bool):
            if args[0]==True: #open
                for i in range(1000):
                    self.q_current[-1] += 0.00003
                    if self.q_current[-1]>= 0.037:
                        self.q_current[-1] = 0.037
                    self.step(self.q_current)
            else:
                for i in range(1000):
                    self.q_current[-1] -= 0.00003
                    if self.q_current[-1] <= 0:
                        self.q_current[-1] = 0
                    self.step(self.q_current)
            return self.q_current
    def run_circle_trajectory(self, center, radius, angular_speed, duration):
        #TODO
        """
        让机械臂末端沿圆轨迹运动

        Args:
            center: 圆心位置 (np.array [x, y, z])
            radius: 半径 (float)
            angular_speed: 角速度 (rad/s)
            duration: 总时长 (秒)
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            t = time.time() - start_time
            angle = angular_speed * t

            # 圆轨迹计算：XY平面圆
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            # 固定旋转（或者你也可以让它同步旋转）
            euler =R.from_matrix(self.target_rotation).as_euler('xyz',degrees=False)
            rx, ry, rz = euler[0], euler[1], euler[2]
            # 调用 move_to_position 控制机械臂运动
            self.move_to_position_no_planner(x, y, z, rx, ry, rz)
            # 控制频率（和 sim_hz 配合）
            time.sleep(1.0 / 20)  # 100 Hz

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_depth_renderer is not None:
            self.mj_depth_renderer.close()

    def trajtory_planner(self,q_init:list,q_goal:list,time:float = 2):
        parameter = JointParameter(q_init, q_goal)
        velocity_parameter = QuinticVelocityParameter(time)
        trajectory_parameter = TrajectoryParameter(parameter, velocity_parameter)
        planner = TrajectoryPlanner(trajectory_parameter)
        time_array = [0.0, time]
        planner_array = [planner]
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
                        self.step(joint)


if __name__ == '__main__':
    env = PiperGraspEnv()
    env.reset()
    while True:
        env.step()
    env.close()

