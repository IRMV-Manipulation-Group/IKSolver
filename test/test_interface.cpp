#include <gtest/gtest.h>
#include "Solver_Interface.h"
#include <Eigen/Dense>

class SolverInterfaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        yml_path = std::string(IKSolver_CONFIG_PATH) + "/realman/kinematics_rightArm.yml";
        solver.initialize(yml_path);
    }

    std::string yml_path;
    Solver_Interface solver;
};

// 测试接口初始化
TEST_F(SolverInterfaceTest, Initialization) {
    Solver_Interface solver_local;
    solver_local.initialize(yml_path);
    
    // 检查关节限制是否正确加载
    const auto& limits = solver_local.getJointMotionLimits();
    ASSERT_GT(limits.rows(), 0);
    ASSERT_EQ(limits.cols(), 5); // min, max, velocity, acceleration, jerk
}

// 测试正运动学
TEST_F(SolverInterfaceTest, ForwardKinematics) {
    int num_joints = 7;
    Eigen::VectorXd joints = Eigen::VectorXd::Zero(num_joints);
    auto result = solver.getFK(joints, -1);
    
    ASSERT_EQ(result.first.first, 0); // 检查错误码
    ASSERT_FALSE(result.second.matrix().isZero()); // 检查结果矩阵非零
}

// 测试逆运动学
TEST_F(SolverInterfaceTest, InverseKinematics) {
    int num_joints = 7;
    Eigen::VectorXd seed_joints = Eigen::VectorXd::Zero(num_joints);
    
    // 首先获取一个有效位姿
    auto fk_result = solver.getFK(seed_joints, -1);
    Eigen::Isometry3d target_pose = fk_result.second;
    
    // 使用略微不同的种子关节尝试逆解
    Eigen::VectorXd slightly_different_seed = seed_joints;
    slightly_different_seed[0] += 0.1;
    
    auto ik_result = solver.getNearestIK(target_pose, slightly_different_seed, 0.5);
    ASSERT_EQ(ik_result.first.first, 0); // 检查错误码
    
    // 验证逆解结果的正解与目标位姿接近
    auto verification_fk = solver.getFK(ik_result.second, -1);
    Eigen::Isometry3d result_pose = verification_fk.second;
    
    // 计算位姿差异
    double pose_diff = (target_pose.matrix() - result_pose.matrix()).norm();
    ASSERT_LT(pose_diff, 1e-4);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}