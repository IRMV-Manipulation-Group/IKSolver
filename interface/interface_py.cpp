/*
  ************************************************************************\

                               C O P Y R I G H T

    Copyright © 2024 IRMV lab, Shanghai Jiao Tong University, China.
                          All Rights Reserved.

    Licensed under the Creative Commons Attribution-NonCommercial 4.0
    International License (CC BY-NC 4.0).
    You are free to use, copy, modify, and distribute this software and its
    documentation for educational, research, and other non-commercial purposes,
    provided that appropriate credit is given to the original author(s) and
    copyright holder(s).

    For commercial use or licensing inquiries, please contact:
    IRMV lab, Shanghai Jiao Tong University at: https://irmv.sjtu.edu.cn/

                               D I S C L A I M E R

    IN NO EVENT SHALL TRINITY COLLEGE DUBLIN BE LIABLE TO ANY PARTY FOR
    DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING,
    BUT NOT LIMITED TO, LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
    AND ITS DOCUMENTATION, EVEN IF TRINITY COLLEGE DUBLIN HAS BEEN ADVISED OF
    THE POSSIBILITY OF SUCH DAMAGES.

    TRINITY COLLEGE DUBLIN DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE. THE SOFTWARE PROVIDED HEREIN IS ON AN "AS IS" BASIS, AND TRINITY
    COLLEGE DUBLIN HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
    ENHANCEMENTS, OR MODIFICATIONS.

    The authors may be contacted at the following e-mail addresses:

            YX.E.Z yixuanzhou@sjtu.edu.cn

    Further information about the IRMV and its projects can be found at the ISG web site :

           https://irmv.sjtu.edu.cn/

  \*************************************************************************
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // 用于STL容器的转换（std::vector, std::map等）
#include <pybind11/eigen.h>    // 用于Eigen矩阵的转换
#include "include/Solver_Interface.h" // Solver_Interface的头文件
#include "IKSolver/kinematics_base.h" // 包含KinematicsImplType枚举定义

namespace py = pybind11;

PYBIND11_MODULE(iksolver, m) {
    m.doc() = "IK Solver Python Interface"; // 模块文档字符串

    // 定义KinematicsImplType枚举类
    py::enum_<bot_kinematics::KinematicsImplType>(m, "KinematicsImplType")
        .value("LM", bot_kinematics::KinematicsImplType::LM)
        .value("QP", bot_kinematics::KinematicsImplType::QP)
        .export_values();

    // 主类Solver_Interface的绑定
    py::class_<Solver_Interface>(m, "SolverInterface")
        .def(py::init<>())
        .def("initialize", &Solver_Interface::initialize, 
             py::arg("yml_path"), 
             py::arg("impl_type") = bot_kinematics::KinematicsImplType::LM,
             "初始化求解器，使用配置文件路径和实现类型")
        
        .def("getJointMotionLimits", &Solver_Interface::getJointMotionLimits,
             "获取关节运动限制")
        
        // 使用lambda函数将嵌套pair转换为简单的三元组
        .def("getFK", [](Solver_Interface& self, const Eigen::VectorXd& joint_position, int link_index) {
                auto result = self.getFK(joint_position, link_index);
                return std::make_tuple(result.first.first, result.first.second, result.second);
             }, 
             py::arg("joint_position"), py::arg("link_index"),
             "计算正向运动学，返回(错误码, 错误信息, 位姿)")
        
        // 同样转换getNearestIK的嵌套pair
        .def("getNearestIK", [](Solver_Interface& self, const Eigen::Isometry3d& cartesian_pose, 
                               const Eigen::VectorXd& joint_seed, double max_dist) {
                auto result = self.getNearestIK(cartesian_pose, joint_seed, max_dist);
                return std::make_tuple(result.first.first, result.first.second, result.second);
             },
             py::arg("cartesian_pose"), py::arg("joint_seed"), py::arg("max_dist"),
             "计算最近的逆运动学解，返回(错误码, 错误信息, 关节角)")
        
        // 简化返回类型为元组
        .def("getNearestApproxIK", [](Solver_Interface& self, const Eigen::Isometry3d& cartesian_pose, 
                                    const Eigen::VectorXd& joint_seed) {
                auto result = self.getNearestApproxIK(cartesian_pose, joint_seed);
                return std::make_tuple(result.first, result.second);
             },
             py::arg("cartesian_pose"), py::arg("joint_seed"),
             "计算最近的近似逆运动学解，返回(误差值, 关节角)")
        
        // 直接返回向量结果
        .def("getAllIKSolutions", &Solver_Interface::getAllIKSolutions,
             py::arg("cartesian_pose"), py::arg("joint_seeds"), py::arg("max_dist"),
             "计算所有可能的逆运动学解")
        
        // 直接返回map结果
        .def("getAllIKSolutionsWithCost", &Solver_Interface::getAllIKSolutionsWithCost,
             py::arg("cartesian_pose"), py::arg("joint_seeds"), py::arg("max_dist"),
             "计算所有带成本的逆运动学解")
        
        // 转换分析雅可比的结果为元组
        .def("getAnalyticalJacobian", [](Solver_Interface& self, const Eigen::VectorXd& joints) {
                Eigen::Matrix<double, 6, -1> jacob;
                auto result = self.getAnalyticalJacobian(joints, jacob);
                return std::make_tuple(result.first, result.second, jacob);
             },
             py::arg("joints"),
             "计算分析雅可比矩阵，返回(错误码, 错误信息, 雅可比矩阵)")
        
        // 转换几何雅可比的结果为元组
        .def("getGeometricJacobian", [](Solver_Interface& self, const Eigen::VectorXd& joints, bool in_ee=false) {
                Eigen::Matrix<double, 6, -1> jacob;
                auto result = self.getGeometricJacobian(joints, jacob, in_ee);
                return std::make_tuple(result.first, result.second, jacob);
             },
             py::arg("joints"), py::arg("in_ee") = false,
             "计算几何雅可比矩阵，返回(错误码, 错误信息, 雅可比矩阵)")
        
        // 直接返回map结果
        .def("getAllApproxIKSolutions", &Solver_Interface::getAllApproxIKSolutions,
             py::arg("cartesian_pose"), py::arg("seed"),
             "计算所有近似逆运动学解")
        
        // 转换分段IK结果为元组
        .def("getIKPiecewise", [](Solver_Interface& self, const Eigen::Isometry3d& elbow_pose, 
                                 const Eigen::Isometry3d& wrist_pose, const Eigen::VectorXd& current_joints) {
                auto result = self.getIKPiecewise(elbow_pose, wrist_pose, current_joints);
                return std::make_tuple(result.first, result.second);
             },
             py::arg("elbow_pose"), py::arg("wrist_pose"), py::arg("current_joints"),
             "计算分段逆运动学解，返回(误差值, 关节角)")
        
        // 人形运动学相关方法
        .def("isHuman", &Solver_Interface::IsHuman,
             "检查运动学模型是否为人形")
        
        .def("setHuman", &Solver_Interface::setHuman,
             py::arg("human"),
             "设置运动学模型是否为人形，参数为bool值");
}