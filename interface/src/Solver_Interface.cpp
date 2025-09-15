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

#include "Solver_Interface.h"
#include "IKSolver/kinematics_screw.h"

void Solver_Interface::initialize(const std::string yml_path, bot_kinematics::KinematicsImplType kin_type)
{

    try {
        kin_ = bot_kinematics::KinematicsScrew::create(yml_path, kin_type);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load config: " << e.what() << std::endl;
        kin_ = nullptr;
    }
    
}

const Eigen::Matrix<double, -1, 5>& Solver_Interface::getJointMotionLimits()
{
    if (kin_ == nullptr) {
        std::cerr << "no config is loaded, please call loadConfig first" << std::endl;
        Eigen::Matrix<double, -1, 5> empty_matrix(0,5); // Return a zero matrix as a fallback
        return empty_matrix; // Returning a reference to a local variable is not safe, so we return a new matrix instead
    }
    return kin_->getJointMotionLimits();
}

std::pair<std::pair<int, std::string>, Eigen::Isometry3d>
Solver_Interface::getFK(const Eigen::VectorXd& joint_position, int link_index)
{
    if (kin_ == nullptr) {
        std::cerr << "no config is loaded, please call loadConfig first" << std::endl;
        return std::make_pair(std::make_pair(-1, "no config is loaded"), Eigen::Isometry3d::Identity());
    }
    auto res_ =  kin_->getFK(joint_position, link_index);
    return std::make_pair(std::make_pair(res_.first.error_code(), res_.first.error_msg()), res_.second);
}

std::pair<std::pair<int, std::string>, Eigen::VectorXd>
Solver_Interface::getNearestIK(const Eigen::Isometry3d& cartesian_pose,
                               const Eigen::VectorXd& joint_seed, double max_dist)
{
    if (kin_ == nullptr) {
        std::cerr << "no config is loaded, please call loadConfig first" << std::endl;
        return  std::make_pair(std::make_pair(-1, "no config is loaded"), Eigen::VectorXd());
    }
    auto res_ =  kin_->getNearestIK(cartesian_pose, joint_seed, max_dist);
    return std::make_pair(std::make_pair(res_.first.error_code(), res_.first.error_msg()), res_.second);
}

std::pair<double, Eigen::VectorXd>
Solver_Interface::getNearestApproxIK(const Eigen::Isometry3d& cartesian_pose, const Eigen::VectorXd& joint_seed)
{
    if (kin_ == nullptr) {
        std::cerr << "no config is loaded, please call loadConfig first" << std::endl;
        return std::make_pair(-1.0, Eigen::VectorXd());
    }
    return kin_->getNearestApproxIK(cartesian_pose, joint_seed);
}


std::vector<Eigen::VectorXd>
Solver_Interface::getAllIKSolutions(const Eigen::Isometry3d& cartesian_pose,
                                    const std::vector<Eigen::VectorXd>& joint_seeds,
                                    double max_dist)
{
    if (kin_ == nullptr) {
        std::cerr << "no config is loaded, please call loadConfig first" << std::endl;
        return std::vector<Eigen::VectorXd>();
    }
    return kin_->getAllIKSolutions(cartesian_pose, joint_seeds, max_dist);
}


std::map<double, Eigen::VectorXd>
Solver_Interface::getAllIKSolutionsWithCost(
    const Eigen::Isometry3d& cartesian_pose,
    const std::vector<Eigen::VectorXd>& joint_seeds,
    double max_dist)
{
    if (kin_ == nullptr) {
        std::cerr << "no config is loaded, please call loadConfig first" << std::endl;
        return std::map<double, Eigen::VectorXd>();
    }
    return kin_->getAllIKSolutionsWithCost(cartesian_pose, joint_seeds, max_dist);
}

std::pair<int, std::string>
Solver_Interface::getAnalyticalJacobian(
    const Eigen::VectorXd& joints,
    Eigen::Matrix<double, 6, -1>& jacob)
{
    if (kin_ == nullptr) {
        std::cerr << "no config is loaded, please call loadConfig first" << std::endl;
        return std::make_pair(-1, "no config is loaded");
    }
    auto res_ = kin_->getAnalyticalJacobian(joints, jacob);
    return std::make_pair(res_.error_code(), res_.error_msg());
}

std::pair<int, std::string>
Solver_Interface::getGeometricJacobian(
    const Eigen::VectorXd& joints,
    Eigen::Matrix<double, 6, -1>& jacob,
    bool in_ee)
{
    if (kin_ == nullptr) {
        std::cerr << "no config is loaded, please call loadConfig first" << std::endl;
        return std::make_pair(-1, "no config is loaded");
    }
    auto res_ = kin_->getGeometricJacobian(joints, jacob, in_ee);
    return std::make_pair(res_.error_code(), res_.error_msg());
}

std::map<double, Eigen::VectorXd>
Solver_Interface::getAllApproxIKSolutions(
    const Eigen::Isometry3d& cartesian_pose,
    const Eigen::VectorXd& seed)
{
    if (kin_ == nullptr) {
        std::cerr << "no config is loaded, please call loadConfig first" << std::endl;
        return std::map<double, Eigen::VectorXd>();
    }
    return kin_->getAllApproxIKSolutions(cartesian_pose, seed);
}


std::pair<double, Eigen::VectorXd>
Solver_Interface::getIKPiecewise(const Eigen::Isometry3d& elbow_pose, const Eigen::Isometry3d& wrist_pose,
                                 const Eigen::VectorXd& CurrentJoints)
{
    if (kin_ == nullptr) {
        std::cerr << "no config is loaded, please call loadConfig first" << std::endl;
        return std::make_pair(-1.0, Eigen::VectorXd());
    }
    return kin_->getIKPiecewise(elbow_pose, wrist_pose, CurrentJoints);
}

bool Solver_Interface::IsHuman() const
{
    if (kin_ == nullptr) {
        std::cerr << "no config is loaded, please call loadConfig first" << std::endl;
        return false;
    }
    auto* derived_kin = dynamic_cast<bot_kinematics::KinematicsScrew*>(kin_.get());
    if (derived_kin) {
        return derived_kin->IsHuman();
    } else {
        // 如果无法转换，说明不是派生类对象或没有实现IsHuman方法
        std::cerr << "The underlying kinematics implementation does not support IsHuman method" << std::endl;
        return false;
    }
}

bool Solver_Interface::setHuman(bool human)
{
    if (kin_ == nullptr) {
        std::cerr << "no config is loaded, please call loadConfig first" << std::endl;
        return false;
    }
    auto* derived_kin = dynamic_cast<bot_kinematics::KinematicsScrew*>(kin_.get());
    if (derived_kin) {
        return derived_kin->setHuman(human);
    } else {
        // 如果无法转换，说明不是派生类对象或没有实现setHuman方法
        std::cerr << "The underlying kinematics implementation does not support setHuman method" << std::endl;
        return false;
    }
}
