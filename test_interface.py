import unittest
import numpy as np
import os
import sys

# 将模块路径添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入您的Python模块
import IKSolver_interface_py as ik

class TestPythonInterface(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        # 获取配置文件路径
        config_path = os.path.join(os.environ.get('IKSolver_CONFIG_PATH', ''), 'realman', 'kinematics_rightArm.yml')
        self.solver = ik.Solver_Interface()
        self.solver.initialize(config_path)
        
    def test_initialization(self):
        """测试初始化"""
        # 检查接口是否正确初始化
        self.assertIsNotNone(self.solver)
        
        # 获取关节限制并检查是否有效
        limits = self.solver.getJointMotionLimits()
        self.assertIsNotNone(limits)
        self.assertTrue(limits.shape[0] > 0)
        self.assertEqual(limits.shape[1], 5)  # min, max, velocity, acceleration, jerk

    def test_forward_kinematics(self):
        """测试正运动学"""
        # 创建零关节角度向量
        joints = np.zeros(7)
        
        # 计算正运动学
        status, pose = self.solver.getFK(joints, -1)
        
        # 检查状态和结果
        self.assertEqual(status[0], 0)  # 检查错误码为0（成功）
        self.assertIsNotNone(pose)
        # 确保结果是一个有效的变换矩阵
        self.assertEqual(pose.shape, (4, 4))
        
    def test_inverse_kinematics(self):
        """测试逆运动学"""
        # 首先获取一个有效位姿
        joints = np.zeros(7)
        _, target_pose = self.solver.getFK(joints, -1)
        
        # 使用略微不同的种子关节尝试逆解
        seed = joints.copy()
        seed[0] += 0.1
        
        status, result_joints = self.solver.getNearestIK(target_pose, seed, 0.5)
        
        # 检查状态和结果
        self.assertEqual(status[0], 0)  # 检查错误码为0（成功）
        self.assertEqual(len(result_joints), 7)
        
        # 验证逆解结果的正解与目标位姿接近
        _, result_pose = self.solver.getFK(result_joints, -1)
        
        # 计算位姿差异
        pose_diff = np.linalg.norm(target_pose - result_pose)
        self.assertLess(pose_diff, 1e-4)
        
    def test_approximate_ik(self):
        """测试近似逆运动学"""
        # 获取一个有效位姿
        joints = np.zeros(7)
        _, target_pose = self.solver.getFK(joints, -1)
        
        # 求解近似逆运动学
        distance, result_joints = self.solver.getNearestApproxIK(target_pose, joints)
        
        # 检查结果
        self.assertLess(distance, float('inf'))
        self.assertEqual(len(result_joints), 7)
        
    def test_piecewise_ik(self):
        """测试分段逆运动学"""
        # 获取一个有效位姿
        joints = np.zeros(7)
        _, end_pose = self.solver.getFK(joints, -1)
        _, elbow_pose = self.solver.getFK(joints, 3)
        
        # 求解分段逆运动学
        distance, result_joints = self.solver.getIKPiecewise(elbow_pose, end_pose, joints)
        
        # 检查结果
        self.assertLess(distance, float('inf'))
        self.assertEqual(len(result_joints), 7)
        
if __name__ == '__main__':
    unittest.main()