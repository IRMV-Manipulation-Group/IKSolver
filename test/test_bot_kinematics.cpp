#include <gtest/gtest.h>
#include <ros/ros.h>
#include "bot_kinematics/kinematics_screw.hpp"
#include <fmt/format.h>
#include <fmt/ostream.h>
#include "yaml-cpp/yaml.h"
#include <ros/package.h>
#include "bot_math/utils/utils.hpp"
#include "bot_math/lie/robot.hpp"
#include <fstream>
#include <filesystem>

using namespace bot_kinematics;
namespace fs = std::filesystem;

class KinematicsTest : public testing::Test {
protected:
    void SetUp() override {
        spinner = std::make_shared<ros::AsyncSpinner>(1);
        spinner->start();
        kin_ = KinematicsMoveit::create("rightArm", "right_effector", false);
        std::string yml_path = ros::package::getPath("dual_arm_app") + "/config/realman/kinematics_rightArm.yml";
        kin2_ = KinematicsScrew::create(yml_path);
        q.resize(7);
        p.resize(7);
    }

    void TearDown() override {
        spinner->stop();
    }

protected:
    std::shared_ptr<ros::AsyncSpinner> spinner;
    KinematicsUniquePtr kin_;
    KinematicsUniquePtr kin2_;
    Eigen::VectorXd q, p;
protected:
    robot::TMat FK(const robot::ThetaList &q) {
        return kin_->getFK(q).second.matrix();
    }
};

/**
 * This test checks whether the validator is working with an user-defined out limitation configuration
 */
// TEST_F(KinematicsTest, FKTest) {
//     // the arm limits are 180,130,180,135,180,128,360 degrees
//     //q << 0., -M_PI_4, 0., -3. * M_PI_4, 0., M_PI_2, M_PI_4; // panda test case
//     q << 0.0, -1.544, 0.0, 0.0, 0.0, 0.0, -1.2269;
//     const auto ee_pose = kin_->getFK(q).second;
//     SE3::TVec l;
//     //l << -2.90245, 1.20224, 0, -0.0928815, -0.965135, 0.184477; // panda test
//     l << 0.936407, -1.33, -0.961842, -0.325851, -0.0822722, 0.662848;
//     ASSERT_LE((ee_pose.matrix() - SE3::Exp(l)).norm(), 1e-5);

// }

 TEST_F(KinematicsTest, FKIdenticalTest) {
     // the arm limits are 180,130,180,135,180,128,360 degrees
     //q << 0., -M_PI_4, 0., -3. * M_PI_4, 0., M_PI_2, M_PI_4; // panda test case
     q = kin2_->getRandomValidJoints(7);
     //q.setZero();
     for(int i=0; i < 7; ++i){
         const auto T1 = kin_->getFK(q, i).second;
         const auto T2 = kin2_->getFK(q, i).second;
         double dis = SE3::distance(T1.matrix(), T2.matrix());
         PLOGD<<"I: "<<i;
         PLOGD<<"T1: "<<T1.matrix();
         PLOGD<<"T2: "<<T2.matrix();
         PLOGD<<"dis: "<<dis;
         ASSERT_TRUE(dis < 1e-4);
     }
 }

TEST_F(KinematicsTest, FKStatistics) {
    auto start = std::chrono::steady_clock::now();
    int counter = 0, MAX = 1e4;
    Eigen::Isometry3d ee_pose;
    double total_time = 0.;
    while (counter++ < MAX) {
        q = kin2_->getRandomValidJoints(7);
        const auto T1 = kin_->getFK(q).second;
        const auto T2 = kin2_->getFK(q).second;
        start = std::chrono::steady_clock::now();
        double dis = SE3::distance(T1.matrix(), T2.matrix());
        ASSERT_TRUE(dis < 1e-4);
        total_time += (std::chrono::steady_clock::now() - start).count() / 1e9;
    }
    PLOGI << "The average time consumption of FK method is: " << total_time / MAX;
}

TEST_F(KinematicsTest, IKTest) {
//    q << -0.17709955251622067, -1.5758234303490546, -1.198518114531418, 1.5835371208127134, 0.6742177327728198, 0.19272128059333857, -0.6380054291920239;
//    q << 1.0331335646837239, -0.9966127790127296, -3.1066899998447988, 0.48775491234236285, -0.06328974272930496, 0.8891697327079396, -2.0149128058647525;
    q << 0.8064226553396053, -1.174633868999962, -1.6925050800013912, 0.48705019696726826, -1.510548878163597, 1.1177022471735238, -1.7367134761732608;
//    q = kin_->getRandomValidJoints(7);
    const auto ee_pose = kin_->getFK(q).second;
    Eigen::VectorXd q2 = kin_->getRandomValidJoints(7);


    auto screw_impl = dynamic_cast<KinematicsScrew *>(kin2_.get());
    const auto ret2 = screw_impl->getNearestIK(ee_pose, q2, 0);
//    ASSERT_TRUE(ret2.first.IsOK());
    double H1 = screw_impl->computeHumanoidIndex(q);
    double H2 = screw_impl->computeHumanoidIndex(ret2.second);


    PLOGD << "The origin q is " << q.transpose();
    PLOGD << "The q with low H is " << ret2.second.transpose();
    PLOGD << "The origin H is " << H1;
    PLOGD << "The H with low H is " << H2;
    PLOGD << "The difference of H is " << H2 - H1;

    double diff = (ee_pose.matrix() - kin_->getFK(ret2.second).second.matrix()).norm();
    PLOGD << "The difference between the original and computed pose is " << diff;
    ASSERT_LE(diff, 1e-5);

    PLOGD << "difference between two configuration: " << (q - ret2.second).cwiseAbs().maxCoeff();
    ASSERT_TRUE(ret2.first.error_code() != bot_common::ErrorCode::IKFailed);
}


 TEST_F(KinematicsTest, IKStatistics) {
     auto start = std::chrono::steady_clock::now();
     int counter = 0, MAX = 1e4;
     Eigen::Isometry3d ee_pose;
     double total_time = 0.;
     p = kin2_->getRandomValidJoints(7);
     int success = 0;
     while (counter++ < MAX) {
         q = kin2_->getRandomValidJoints(7);
         ee_pose = kin2_->getFK(q).second;
         start = std::chrono::steady_clock::now();
         const auto ret = kin2_->getNearestIK(ee_pose, p, 0);
         total_time += (std::chrono::steady_clock::now() - start).count() / 1e9;
         if (ret.first.error_code() != bot_common::ErrorCode::IKFailed) {
             double diff = (ee_pose.matrix() - kin2_->getFK(ret.second).second.matrix()).norm();
             if (diff < 1e-4) {
                 const auto error_info = kin2_->isInsideLimits(ret.second);
                 success = error_info.IsOK() ? success + 1 : success;
                 PLOGI_IF(!error_info.IsOK()) << error_info.error_msg();
             } else {
                 PLOGD << "diff: " << diff;
             }
         }
     }
     PLOGI << "The average time consumption of IK method is: " << total_time / MAX * 1e3 <<" ms";
     PLOGI << "The total time consumption of IK method is: " << total_time <<" s";
     PLOGI << "The success proportion is : " << (double) success / MAX;
 }

/**
 * This test check whether the validator is working with an user-defined in collision configuration
 */
TEST_F(KinematicsTest, screwSet) {
    robot::ScrewList S(6, 7);
    auto fk = [this](const robot::ThetaList &theta) { return this->FK(theta); };
    auto M = robot::computeScrewList(fk, S);
    std::string yml_path = ros::package::getPath("dual_arm_app") + "/config/realman/kinematics_rightArm.yml";
    YAML::Node doc = YAML::LoadFile(yml_path);
    for (int i = 0; i < 7; ++i) {
        for(int j = 0; j < S.rows(); ++j){
            S(j, i) = std::abs(S(j, i)) < 1e-14 ? 0. : S(j, i);
        }
        doc["screws"][std::to_string(i)] = utils::covertToStdVector(S.col(i));
    }
    for(int j = 0 ; j < M.matrix().size(); ++j){
        *(M.data() + j) = std::abs( *(M.data() + j) ) < 1e-14 ? 0. :  *(M.data() + j);
    }
    doc["M"] = utils::covertToStdVector(SE3::Log(M));

    const auto JL = kin_->getJointMotionLimits();
    for (int i = 0; i < 7; ++i) {
        doc["limits"][std::to_string(i)] = utils::covertToStdVector(JL.row(i).transpose());
        auto Mi = kin_->getFK(Eigen::VectorXd::Zero(7), i).second;
        for(int j = 0 ; j < Mi.matrix().size(); ++j){
            *(Mi.data() + j) = std::abs( *(Mi.data() + j) ) < 1e-14 ? 0. :  *(Mi.data() + j);
        }
        doc["links"][std::to_string(i)] = utils::covertToStdVector(SE3::Log(Mi.matrix()));
    }
    Eigen::Matrix3d R = Eigen::Quaterniond {1, 0, 0, 0}.toRotationMatrix();
    doc["R"] = utils::covertToStdVector(SO3::Log(R));

    doc["isSpace"] = true;
    doc["ik_tolerance_emog"] = 3e-6;
    doc["ik_tolerance_ev"] = 3e-6;
    doc["time_out"] = 2e-3;
    std::ofstream out_file(yml_path);
    if (out_file.is_open()) {
        out_file.clear();
        out_file << doc;
        out_file.close();
    }
    std::cout << S.matrix() << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << M.matrix() << std::endl;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "kinematics_test");

    std::string base_path = getenv("HOME");

    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_c);
    std::stringstream ss;
    ss << std::put_time(&tm, "_%m%d_%H_%M%S");
    std::string folderPath = base_path + "/logs/" + ss.str();
    if (!fs::exists(folderPath)) {
        if (fs::create_directory(folderPath))
            std::cout << "create_directory : " << folderPath << std::endl;
        else
            std::cout << "create_directory failed: " << folderPath << std::endl;
    }
    std::map<std::string, std::string> keysAndFilenames = {
            {"Alpha", folderPath + "/Alpha_" + ss.str() + ".csv"},
            {"Humanoid", folderPath + "/Humanoid_" + ss.str() + ".csv"},
    };

    setPlog(false, "Debug", folderPath + "/logsbase_log_traj.csv", 1024 * 1024 * 10, 50, keysAndFilenames, true);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
