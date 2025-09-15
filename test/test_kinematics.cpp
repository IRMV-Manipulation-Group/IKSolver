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
#include <gtest/gtest.h>
#include "IKSolver//kinematics_screw.h"
#include <Eigen/Dense>

using namespace bot_kinematics;

class KinematicsTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::cout << "Setting up the test environment." << std::endl;
        yml_path = std::string(IKSolver_CONFIG_PATH) + "/realman/kinematics_rightArm.yml";
    }

    std::string yml_path;
};


TEST_F(KinematicsTest, InverseKinematicsLM) {
    std::cout << "InverseKinematicsLM"<< yml_path << std::endl;
    auto kinematics = KinematicsScrew::create(yml_path, KinematicsImplType::LM);
    if (!kinematics) {
        std::cerr << "创建 KinematicsScrew 对象失败!" << std::endl;
        return;
    }
    int i = 0, max = 1000;
    int success = 0;
    while (i++ < max) {
        Eigen::VectorXd q = kinematics->getRandomValidJoints(7);
        Eigen::Isometry3d pose = kinematics->getFK(q).second;
        Eigen::VectorXd p = kinematics->getRandomValidJoints(7);
        auto result = kinematics->getNearestIK(pose, p, 0);
        if (result.first.IsOK()){
            success++;
        }else{
            double distance = SE3::distance(pose.matrix(), kinematics->getFK(result.second).second.matrix());

            // std::cout<<"distance: "<<distance<<std::endl;
        }
    }
    std::cout<<"Success proportion: "<<success<<"/"<<max<<std::endl;
    ASSERT_TRUE(success == max);
}


TEST_F(KinematicsTest, InverseKinematicsQP) {
    auto kinematics = KinematicsScrew::create(yml_path, KinematicsImplType::QP);

    int i = 0, max = 1000;
    int success = 0;
    while (i++ < max) {
        Eigen::VectorXd q = kinematics->getRandomValidJoints(7);
        Eigen::Isometry3d pose = kinematics->getFK(q).second;
        Eigen::VectorXd p = kinematics->getRandomValidJoints(7);
        auto result = kinematics->getNearestIK(pose, p, 0);
        // if (result.first.IsOK()){
        //     success++;
        // }else{
        //     double distance = SE3::distance(pose.matrix(), kinematics->getFK(result.second).second.matrix());
        //
        //     // std::cout<<"distance: "<<distance<<std::endl;
        // }

        if (result.first.error_code() != bot_common::ErrorCode::IKFailed) {
            double diff = (pose.matrix() - kinematics->getFK(result.second).second.matrix()).norm();
            if (diff < 1e-4) {
                const auto error_info = kinematics->isInsideLimits(result.second);
                success = error_info.IsOK() ? success + 1 : success;
                PLOGI_IF(!error_info.IsOK()) << error_info.error_msg();
            } else {
                //                 PLOGD << "diff: " << diff;
            }
        }
    }
    std::cout<<"Success proportion: "<<success<<"/"<<max<<std::endl;
    ASSERT_TRUE(success == max);
}

TEST_F(KinematicsTest, InverseKinematicsQPApprox) {
    auto kinematics = KinematicsScrew::create(yml_path, KinematicsImplType::QP);

    int i = 0, max = 1000;
    int success = 0;
    while (i++ < max) {
        Eigen::VectorXd q = kinematics->getRandomValidJoints(7);
        Eigen::Isometry3d pose = kinematics->getFK(q).second;
        Eigen::VectorXd p = kinematics->getRandomValidJoints(7);
        auto result = kinematics->getNearestApproxIK(pose, p);
        if (result.first<std::numeric_limits<double>::max()){
            success++;
        }else{
            double distance = SE3::distance(pose.matrix(), kinematics->getFK(result.second).second.matrix());

            std::cout<<"distance: "<<distance<<std::endl;
        }
    }
    std::cout<<"Success proportion: "<<success<<"/"<<max<<std::endl;
    ASSERT_TRUE(success == max);
}

TEST_F(KinematicsTest, InverseKinematicsLMApprox) {
    auto kinematics = KinematicsScrew::create(yml_path, KinematicsImplType::LM);

    int i = 0, max = 1000;
    int success = 0;
    while (i++ < max) {
        Eigen::VectorXd q = kinematics->getRandomValidJoints(7);
        Eigen::Isometry3d pose = kinematics->getFK(q).second;
        Eigen::VectorXd p = kinematics->getRandomValidJoints(7);
        auto result = kinematics->getNearestApproxIK(pose, p);
        if (result.first<std::numeric_limits<double>::max()){
            success++;
        }else{
            double distance = SE3::distance(pose.matrix(), kinematics->getFK(result.second).second.matrix());

            std::cout<<"distance: "<<distance<<std::endl;
        }
    }
    std::cout<<"Success proportion: "<<success<<"/"<<max<<std::endl;

}

TEST_F(KinematicsTest, InverseKinematicsLMPiec) {
    auto kinematics = KinematicsScrew::create(yml_path, KinematicsImplType::LM);

    int i = 0, max = 10000;
    int success = 0;
    while (i++ < max) {
        Eigen::VectorXd q = kinematics->getRandomValidJoints(7);
        Eigen::Isometry3d pose = kinematics->getFK(q).second;
        Eigen::Isometry3d elbow_pose = kinematics->getFK(q, 3).second;
        Eigen::VectorXd p = kinematics->getRandomValidJoints(7);
        auto result = kinematics->getIKPiecewise( elbow_pose, pose, p);
        if (result.first<std::numeric_limits<double>::max()){
            success++;
        }else{
            double distance = SE3::distance(pose.matrix(), kinematics->getFK(result.second).second.matrix());

            std::cout<<"distance: "<<distance<<std::endl;
        }
    }
    std::cout<<"Success proportion: "<<success<<"/"<<max<<std::endl;
    ASSERT_TRUE(success == max);
}
TEST_F(KinematicsTest, InverseKinematicsQPPiec) {
    auto kinematics = KinematicsScrew::create(yml_path, KinematicsImplType::QP);

    int i = 0, max = 10000;
    int success = 0;
    while (i++ < max) {
        Eigen::VectorXd q = kinematics->getRandomValidJoints(7);
        Eigen::Isometry3d pose = kinematics->getFK(q).second;
        Eigen::Isometry3d elbow_pose = kinematics->getFK(q, 3).second;
        Eigen::VectorXd p = kinematics->getRandomValidJoints(7);
        auto result = kinematics->getIKPiecewise( elbow_pose, pose, p);
        if (result.first<std::numeric_limits<double>::max()){
            success++;
        }else{
            double distance = SE3::distance(pose.matrix(), kinematics->getFK(result.second).second.matrix());

            std::cout<<"distance: "<<distance<<std::endl;
        }
    }
    std::cout<<"Success proportion: "<<success<<"/"<<max<<std::endl;
    ASSERT_TRUE(success == max);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

