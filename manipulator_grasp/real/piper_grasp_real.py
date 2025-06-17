import os.path
import sys

from typing import List

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
import piper_sdk
ROOT_DIR =os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))
parent_dir = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/Kinova7DoF-MuJoCo'
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
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
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

    def solve(self,qpos,target_pose:pin.SE3):
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
            q_init[-1] = q_init[-2] * -1  # 假设第8维为 prismatic 关节，初始为0
        else:
            raise ValueError(f"qpos 维度应为 {nq} 或 {nq - 1}，但收到 {qpos.size}")
        self.opti.set_initial(self.var_q, q_init)
        self.opti.set_value(self.param_tf, casadi.DM(target_pose.homogeneous))
        try:
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)
        except RuntimeError as e:
            print("IK 求解失败：", e)
            sol_q = q_init.copy()
        joint_ctrl = np.array(sol_q).flatten()
        return joint_ctrl

    def forward_kinematics(self,q):
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
        self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)
        self.data = self.model.createData()
        self.target_translation = np.array([0.155, 0.0, 0.222])  # ✅ 更新初始末端位置

        q = quaternion_from_euler(0, 0, 0)
        self.model.addFrame(
            pin.Frame('ee',
                      self.model.getJointId('joint6'),
                      pin.SE3(
                          pin.Quaternion(q[3], q[0], q[1], q[2]),
                          np.array([0.0, 0.0, 0.1]),
                      ),
                      pin.FrameType.OP_FRAME)
        )#末端姿态
        self.gripper_id = self.model.getFrameId("ee")
        self.q_current = np.zeros(8)  # 初始关节角度
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q",self.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)

        self.error = casadi.Function(
            "error",
            [self.cq,self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],)
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        self.regularization = casadi.sumsqr(self.var_q)
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

    def objective(self,q,target_pose):
        current_pose = self.forward_kinematics(q)
        error = pin.log(current_pose.inverse() * target_pose)
        return error

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
            return self.q_current
        # self.q_current = sol_q
        return sol_q

class PiperGraspReal:
    def __init__(self):
        urdf_path = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/manipulator_grasp/assets/agilex_piper/piper_description/urdf/piper_description.urdf'
        self.ik_solver = IKSolver(urdf_path)
        self.piper_intference=piper_sdk.C_PiperInterface(
            can_name='can0',
        )
        self.piper_intference.ConnectPort(can_init=True)
        self.q_current=[0]*8
        self.piper_enable()
    def piper_enable(self, timeout: float = 5.0):
        start = time.time()
        # 初始化爪端到全开位置，确保模式切换
        self.piper_intference.EnableArm(7)
        self.piper_intference.GripperCtrl(int(0.03 * 1e6), 1000, 0x01, 0)
        while time.time() - start < timeout:
            status = self.piper_intference.GetArmLowSpdInfoMsgs()
            motors = [status.motor_1, status.motor_2, status.motor_3,
                      status.motor_4, status.motor_5, status.motor_6]
            if all(m.foc_status.driver_enable_status for m in motors):
                print("Arm enabled")
                return
            time.sleep(0.1)
        raise RuntimeError("Arm enable timeout")

    def move_joints_ctrl(self, angles: List[float], speed: float=30):
        """
            关节控制
        """
        factor = 57324.840764  # 1000*180/3.14
        if len(angles) != 6:
            raise ValueError("Expect 6 joint angles")
        angles = [round(angle*factor) for angle in angles]
        self.piper_intference.MotionCtrl_2(0x01, 0x01, int(speed), 0x00)

        self.piper_intference.JointCtrl(*[int(a) for a in angles])
    def Pose_to_Joint_IK(self,*args):
        """
            逆运动学解算成员函数
        """
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
        q_new = self.ik_solver.solve(self.q_current, target_pose)[:6]

        q_out = np.concatenate([q_new, np.array([self.q_current[-1]])], axis=0)
        return q_out

    def gripper_control(self,*args):
        if len(args) == 1 and isinstance(args[0], bool):
            gripper_state = self.q_current[-1]
            if args[0] == True:  # open
                for i in range(1000):
                    gripper_state += 0.00003
                    if gripper_state >= 0.037:
                        gripper_state = 0.037
                    self.piper_intference.GripperCtrl(int(gripper_state*1e6),1000, 0x01, 0)
            elif args[0] == False:
                for i in range(1000):
                    gripper_state -= 0.00003
                    if gripper_state <= 0:
                        self.gripper_state = 0
                    self.piper_intference.GripperCtrl(int(gripper_state*1e6),1000, 0x01, 0)

    def trajtory_planner(self, q_init: list, q_goal: list, time_: float = 2):
        parameter = JointParameter(np.array(q_init), np.array(q_goal))
        velocity_parameter = QuinticVelocityParameter(time_)
        trajectory_parameter = TrajectoryParameter(parameter, velocity_parameter)
        planner = TrajectoryPlanner(trajectory_parameter)
        time_array = [0.0,time_]
        planner_array = [planner]
        total_time = np.sum(time_array)
        time_step_num = round(total_time/0.002) + 1
        times = np.linspace(0.0, total_time, time_step_num)
        time_cumsum = np.cumsum(time_array)
        for timei in times:
            for j in range(len(time_cumsum)):
                if timei == 0.0:
                    break
                if timei <= time_cumsum[j]:
                    planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                    if isinstance(planner_interpolate, np.ndarray):
                        joint = planner_interpolate[:6].tolist()
                        self.move_joints_ctrl(joint)
                        time.sleep(0.001)

    def Get_joint(self):
        joint_0: float = (self.piper_intference.GetArmJointMsgs().joint_state.joint_1 / 1000) * 0.017444
        joint_1: float = (self.piper_intference.GetArmJointMsgs().joint_state.joint_2 / 1000) * 0.017444
        joint_2: float = (self.piper_intference.GetArmJointMsgs().joint_state.joint_3 / 1000) * 0.017444
        joint_3: float = (self.piper_intference.GetArmJointMsgs().joint_state.joint_4 / 1000) * 0.017444
        joint_4: float = (self.piper_intference.GetArmJointMsgs().joint_state.joint_5 / 1000) * 0.017444
        joint_5: float = (self.piper_intference.GetArmJointMsgs().joint_state.joint_6 / 1000) * 0.017444
        joint_6: float = self.piper_intference.GetArmGripperMsgs().gripper_state.grippers_angle / 1000000
        self.q_current = [joint_0,joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,-1*joint_6]
        return self.q_current

    def piper_disconnect(self):
        self.piper_intference.DisconnectPort()


#测试案例，关节控制轨迹插值
if __name__ == '__main__':
    Piper_ctrl = PiperGraspReal()
    # Piper_ctrl.trajtory_planner(q_init=Piper_ctrl.Get_joint(),q_goal=[0]*8)
    Piper_ctrl.trajtory_planner(q_init=Piper_ctrl.Get_joint(), q_goal=[0.0, 0.644, -0.62,0.1, 0.57, 0,0,0])
    