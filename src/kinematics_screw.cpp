#include "bot_kinematics/kinematics_screw.hpp"
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <unordered_set>
#include <fmt/format.h>
#include "bot_math/utils/utils.hpp"
#include "robot.hpp"
#define  gravity_const  9.8

namespace bot_kinematics {
    KinematicsScrew::KinematicsScrew(const std::string &screw_yml) : KinematicsBase() {
        YAML::Node doc = YAML::LoadFile(screw_yml);
        std::vector<double> data_raw;
        //read screws
        const auto &screws = doc["screws"];
        SCREW_LIST.resize(6, screws.size());

        int i = 0;
        for (YAML::const_iterator item = screws.begin(); item != screws.end(); ++item) {
            data_raw = item->second.as<std::vector<double>>();
            if (data_raw.size() != 6) {
                throw std::invalid_argument("The screw given is not 6 elements");
            }
            SCREW_LIST.col(i++) = Eigen::Map<utils::Vector6D>(data_raw.data());
        }

        //read M
        const auto &m = doc["M"];
        M = SE3::Exp(Eigen::Map<const Eigen::VectorXd>(m.as<std::vector<double>>().data(), 6, 1));

        const auto &r = doc["R"];
        R = SO3::Exp(Eigen::Map<const Eigen::Vector3d>(r.as<std::vector<double>>().data(), 3, 1));

        const auto &links = doc["links"];
        Links.resize(links.size());
        i = 0;
        for (YAML::const_iterator item = links.begin(); item != links.end(); ++item) {
            data_raw = item->second.as<std::vector<double>>();
            if (data_raw.size() != 6) {
                throw std::invalid_argument("The links given is not 6 elements");
            }
            Links[i++] = SE3::Exp(Eigen::Map<const Eigen::VectorXd>(data_raw.data(), 6, 1));
        }

        //read isSpace
        isSpace = doc["isSpace"].as<bool>();

        //read ik_tolerance
        ik_tolerance_emog = doc["ik_tolerance_emog"].as<double>();
        ik_tolerance_ev = doc["ik_tolerance_ev"].as<double>();

        //read time_out restrain
        time_out = doc["time_out"].as<double>();

        //read max_size
        max_size = doc["max_size"].as<int>();

        //read joint limits;
        const auto &limits = doc["limits"];
        JointMotionLimits.resize((long)limits.size(), 5);
        i = 0;
        for (YAML::const_iterator item = limits.begin(); item != limits.end(); ++item) {
            data_raw = item->second.as<std::vector<double>>();
            if (data_raw.size() != 5) {
                throw std::invalid_argument("The limits given is not 5 elements");
            }
            JointMotionLimits.row(i++) = Eigen::Map<Eigen::RowVectorXd>(data_raw.data(), 1, 5);
        }

        //read isHuman
        isHuman = doc["isHuman"].as<bool>();
    }

    KinematicsScrew::~KinematicsScrew() = default;

    KinematicsUniquePtr KinematicsScrew::create(const std::string &screw_yml) {
        return bot_common::AlgorithmFactory<KinematicsBase, const std::string &>::CreateAlgorithm(KinematicsScrewName,
                                                                                                  screw_yml);
    }

    const Eigen::Matrix<double, -1, 5> &KinematicsScrew::getJointMotionLimits() const {
        return JointMotionLimits;
    }

    std::pair<bot_common::ErrorInfo, Eigen::Isometry3d>
    KinematicsScrew::getFK(const Eigen::VectorXd &joint_position, int link_index) const {
        if (isSpace) {
            if (link_index >= SCREW_LIST.cols() || link_index < 0)
                return {bot_common::ErrorInfo::OK(),
                        Eigen::Isometry3d(robot::fkInSpace(M, SCREW_LIST, joint_position))};
            else {
                return {bot_common::ErrorInfo::OK(),
                        Eigen::Isometry3d(robot::fkInSpace(Links[link_index], SCREW_LIST.leftCols(link_index + 1),
                                                           joint_position.head(link_index + 1)))};
            }
        } else {
            if (link_index >= SCREW_LIST.cols() || link_index < 0)
                return {bot_common::ErrorInfo::OK(),
                        Eigen::Isometry3d(robot::fkInBody(M, SCREW_LIST, joint_position))};
            else {
                return {bot_common::ErrorInfo::OK(),
                        Eigen::Isometry3d(robot::fkInBody(Links[link_index], SCREW_LIST.leftCols(link_index + 1),
                                                          joint_position.head(link_index + 1)))};
            }
        }
    }

    std::pair<bot_common::ErrorInfo, Eigen::VectorXd>
    KinematicsScrew::getNearestIK(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd &joint_seed,
                                  double max_dist) const {

        return getNearestIKImpl(cartesian_pose, joint_seed, max_dist);
        // 获取所有有效逆解
        std::vector<Eigen::VectorXd> allIKSolutions = getAllIKSolutions(cartesian_pose, {joint_seed}, max_dist);
        if(allIKSolutions.empty()){
            return {bot_common::ErrorInfo(bot_common::ErrorCode::IKFailed,
                                          "Ik solve failed, return the original seed"),
                    joint_seed};
        }
        // 存储逆解和对应的H值
        return {bot_common::ErrorInfo::OK(),
                allIKSolutions.front()};

    }

    std::pair<bot_common::ErrorInfo, Eigen::VectorXd>
    KinematicsScrew::getNearestIKImpl(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd &joint_seed,
                                      double max_dist) const {
        robot::ThetaList angles = joint_seed;
        if (angles.isZero())
            angles = getRandomValidJoints(angles.size());
        bool ret = false;
        auto start = std::chrono::steady_clock::now();
        robot::gradient_func getHumanoidGradient = [this](double &H, const robot::ThetaList &q,
                                                          const robot::Jacobian &J) {
            Eigen::RowVectorXd gH;
            H = this->computeHumanoidIndexAndGradient(gH, q, J);
            return gH;
        };

        while (!ret) {
            double elapsed = (double) (std::chrono::steady_clock::now() - start).count() / 1e9;
            if (elapsed > time_out)
                break;
            if (isSpace) {
                if(isHuman) {
                    PLOGD << "solve humanoid IK with QP methods";
//                    ret = robot::humanoidIKLMInSpace(getHumanoidGradient, cartesian_pose.matrix(), angles, M, SCREW_LIST, ik_tolerance_emog,
//                                                      ik_tolerance_ev);
                    ret = robot::humanoidIKQPInSpace(getHumanoidGradient, cartesian_pose.matrix(), angles, M,
                                                     SCREW_LIST, JointMotionLimits.leftCols(3), ik_tolerance_emog,
                                                     ik_tolerance_ev);
                }
                else{
//                    ret = robot::numericalIKQPInSpace(cartesian_pose.matrix(), angles, M, SCREW_LIST, JointMotionLimits.leftCols(3), ik_tolerance_emog,
//                                                      ik_tolerance_ev);
                    ret = robot::numericalIKLMInSpace(cartesian_pose.matrix(), angles, M, SCREW_LIST, ik_tolerance_emog,
                                                      ik_tolerance_ev);
//                    PLOGD << "ExtraJoints :" << angles.transpose();
                }
            } else {
                ret = robot::numericalIKLMInBody(cartesian_pose.matrix(), angles, M, SCREW_LIST, ik_tolerance_emog,
                                                 ik_tolerance_ev);
            }
            //do randomly seeding
            bool inLimits = isInsideLimits(angles).IsOK();

            bool notValid = this->validCallback_ != nullptr && !this->validCallback_(angles);
            if (!ret || !inLimits || notValid ) {
                angles = getRandomValidJoints(7);
                ret = false;
            }
        }
        wrap(angles);
        if (ret) {
            bool isMaxDisExceeded = max_dist > 0. && (angles - joint_seed).cwiseAbs().maxCoeff() > max_dist;
            double H = computeHumanoidIndex(angles);
//                    PLOGD << "Human Index: " << H << " and joints: " << angles.transpose();
            PLOGD << "Humanoid :" << H;
            PLOGD << "ExtraJoints :" << angles.transpose();
            return isMaxDisExceeded ? std::make_pair(bot_common::ErrorInfo(bot_common::ErrorCode::IkExceedMaxDis,
                                                                           "Ik solved but the specified max distance is exceeded"),
                                                     angles)
                                    : std::make_pair(bot_common::ErrorInfo::OK(), angles);
        } else {
            return {bot_common::ErrorInfo(bot_common::ErrorCode::IKFailed,
                                          "Ik solve failed, return the computed IK"),
                    joint_seed};
        }
    }

    std::pair<double, Eigen::VectorXd>
    KinematicsScrew::getNearestApproxIK(const Eigen::Isometry3d &cartesian_pose,
                                        const Eigen::VectorXd &joint_seed) const {
        robot::ThetaList angles = joint_seed;
        double error;
        robot::gradient_func getHumanoidGradient = [this](double &H, const robot::ThetaList &q,
                                                          const robot::Jacobian &J) {
            Eigen::RowVectorXd gH;
            H = this->computeHumanoidIndexAndGradient(gH, q, J);
            return gH;
        };

        if (isSpace) {
//             robot::numericalIKQPInSpace(cartesian_pose.matrix(), angles, M, SCREW_LIST, JointMotionLimits.leftCols(3), ik_tolerance_emog,
//                                             ik_tolerance_ev, &error);
            robot::numericalIKLMInSpace(cartesian_pose.matrix(), angles, M, SCREW_LIST, ik_tolerance_emog,
                                        ik_tolerance_ev, &error);
//            PLOGD << "ExtraJoints :" << angles.transpose();
        } else {
            robot::numericalIKLMInBody(cartesian_pose.matrix(), angles, M, SCREW_LIST, ik_tolerance_emog, ik_tolerance_ev, &error);
        }
        //do randomly seeding
        wrap(angles, 2. * M_PI, -2. * M_PI);
        auto inLimits_ret = isInsideLimits(angles);
        if(inLimits_ret.IsOK())
            return std::make_pair(error, angles);
        return std::make_pair(std::numeric_limits<double>::max(), angles);
    }

    std::vector<Eigen::VectorXd> KinematicsScrew::getAllIKSolutions(const Eigen::Isometry3d &cartesian_pose,
                                                                    const std::vector<Eigen::VectorXd> &joint_seeds,
                                                                    double max_dist) const {
        const auto out = getAllIKSolutionsWithCost(cartesian_pose, joint_seeds, max_dist);
        std::vector<Eigen::VectorXd> real_out(out.size());
        uint maxSize = max_size <= 0 ? std::numeric_limits<uint>::max() : (uint) max_size;
        maxSize = std::min((uint)real_out.size(), maxSize);

        for(int j = 0; j < maxSize; ++j){
            auto iter = out.begin();
            std::advance(iter, j);
            real_out[j] = iter->second;
        }
        return real_out;
    }

    std::map<double, Eigen::VectorXd>
    KinematicsScrew::getAllIKSolutionsWithCost(const Eigen::Isometry3d &cartesian_pose, const std::vector<Eigen::VectorXd> &joint_seeds,
                              double max_dist) const{
        assert(!joint_seeds.empty()); //must contain at least on joint seeds;
        robot::ThetaList angles;
        std::map<double, Eigen::VectorXd> out;
        auto start = std::chrono::steady_clock::now();
        double elapsed = (double) (std::chrono::steady_clock::now() - start).count() / 1e9;
        uint i = 0;
        // at least try all seeds
        while (elapsed < time_out || i < joint_seeds.size()) {
            elapsed = (double) (std::chrono::steady_clock::now() - start).count() / 1e9;
            if (i < joint_seeds.size()) {
                angles = joint_seeds[i++];
            } else {
                angles = getRandomValidJoints(angles.size());
            }
            const auto ret = getNearestIKImpl(cartesian_pose, angles, 0);
            if (ret.first.IsOK()) {
                double dis = (ret.second - angles).cwiseAbs().maxCoeff();
                if (max_dist <= 0 || (max_dist > 0 && dis < max_dist)){
                    double distance = IsHuman() ? computeHumanoidIndex(ret.second) : dis;
                    out[distance] = ret.second;
                }
            }
        }
        return out;
    }

    bot_common::ErrorInfo KinematicsScrew::getAnalyticalJacobian(const Eigen::VectorXd &joints,
                                                                 Eigen::Matrix<double, 6, -1> &jacob) const {

        if (isSpace) {
            jacob = robot::analyticalJacobianSpace(M, SCREW_LIST, joints);
        } else {
            jacob = robot::analyticalJacobianBody(M, SCREW_LIST, joints);
        }
        return bot_common::ErrorInfo::OK();
    }

    bot_common::ErrorInfo
    KinematicsScrew::getGeometricJacobian(const Eigen::VectorXd &joints,
                                          Eigen::Matrix<double, 6, -1> &jacob, bool in_ee) const {
        if (isSpace) {
            robot::jacobianSpaceInPlace(SCREW_LIST, joints, jacob);
        } else {
            robot::jacobianBodyInPlace(SCREW_LIST, joints, jacob);
        }
        if (in_ee && isSpace) {
            jacob = SE3::adjoint(getFK(joints, -1).second.matrix(), true) * jacob;
        } else if (!in_ee && !isSpace) {
            jacob = SE3::adjoint(getFK(joints, -1).second.matrix(), false) * jacob;
        }

        return bot_common::ErrorInfo::OK();
    }

    double KinematicsScrew::computeHumanoidIndex(const Eigen::VectorXd &joints) const {
        Eigen::Vector3d cw = getFK(joints, 5).second.translation(); //current wrist
        Eigen::Vector3d ce = getFK(joints, 3).second.translation(); //current elbow
        Eigen::Vector3d cs = getFK(joints, 1).second.translation(); //current shoulder;

        //edge: shoulder to wrist, shoulder to elbow, elbow to wrist: Notice: the array is transferred into world frame
        Eigen::Vector3d edge_sw(R * (cw - cs)), edge_se(R * (ce - cs));

        Eigen::Vector3d n_vertical = edge_sw.cross(Eigen::Vector3d::UnitZ());
        n_vertical = n_vertical.normalized();

        Eigen::Vector3d n_current = edge_sw.cross(edge_se);
        n_current = n_current.normalized();

        Eigen::Vector3d edge_sw_ = n_vertical.cross(n_current);
        double cosval = n_vertical.dot(n_current);
        cosval = 1 - cosval < SE3::eps ? 9e-3 : cosval; // cosval = 1, val = 0, is undefined;
        double alpha = 0;
        if (edge_sw_.dot(edge_sw) > 0) // same direction alpha > 0
            alpha = M_PI - acos(cosval); // alpha > 0
        else
            alpha = -M_PI + acos(cosval); // alpha < 0
        PLOGD << "Alpha :" << alpha;
        const double b_const = 1.0;
        const double w_const = 1.0;

        const double mu = 1.45772;  // 0.59356 + 0.43285 + 0.43131;
        const double ml = 0.52903;  // 0.28963 + 0.2394;
        double hu = 0.5 * Eigen::Vector3d::UnitZ().dot(R * (ce + cs)) + 1.;
        double hl = 0.5 * Eigen::Vector3d::UnitZ().dot(R * (cw + ce)) + 1.;

        double he = Eigen::Vector3d::UnitZ().dot(edge_se);
        double hw = Eigen::Vector3d::UnitZ().dot(edge_sw);
        double lu = edge_se.norm();
        double ll = edge_sw.norm();

//        double H = mu * gravity_const * hu + ml * gravity_const * hl + 0.5 * b_const * pow((M_PI - alpha), 2);
        double H = (1 + he) / (4 * lu) + (1 + hw) / (4 * (lu + ll)) + 0.5 * b_const * pow((M_PI - alpha), 2);
        H += w_const * pow(joints[5] - 0., 2);
        //H += w_const * pow(joints[6] - 0., 2);
        return H;
    }

    double KinematicsScrew::computeHumanoidIndexAndGradient(Eigen::RowVectorXd &gH, const Eigen::VectorXd &joints,
                                                            const robot::Jacobian &J) const {
        Eigen::Vector3d cw = getFK(joints, 5).second.translation(); //current wrist
        Eigen::Vector3d ce = getFK(joints, 3).second.translation(); //current elbow
        Eigen::Vector3d cs = getFK(joints, 1).second.translation(); //current shoulder;
        //edge: shoulder to wrist, shoulder to elbow, elbow to wrist: Notice: the array is transferred into world frame
        Eigen::Vector3d edge_sw(R * (cw - cs)), edge_se(R * (ce - cs));

        Eigen::Vector3d n_vertical = edge_sw.cross(Eigen::Vector3d::UnitZ());
        double n1_norm = n_vertical.norm();
        n_vertical = n_vertical.normalized();


        Eigen::Vector3d n_current = edge_sw.cross(edge_se);
        double n2_norm = n_current.norm();
        n_current = n_current.normalized();


        Eigen::Vector3d edge_sw_ = n_vertical.cross(n_current);
        double cosval = n_vertical.dot(n_current);
        cosval = std::clamp(cosval, -1.0 + SE3::eps, 1.0 - SE3::eps); // cosval = 1, val = 0, is undefined;

        double alpha = 0;
        if (edge_sw_.dot(edge_sw) > 0) // same direction alpha > 0
            alpha = M_PI - acos(cosval); // alpha > 0
        else
            alpha = -M_PI + acos(cosval); // alpha < 0
        const double b_const = 1.0;
        const double w_const = 1.0;

        const double mu = 1.45772;  // 0.59356 + 0.43285 + 0.43131;
        const double ml = 0.52903;  // 0.28963 + 0.2394;
        double hu = 0.5 * Eigen::Vector3d::UnitZ().dot(R * (ce + cs)) + 1.;
        double hl = 0.5 * Eigen::Vector3d::UnitZ().dot(R * (cw + ce)) + 1.;


        double he = Eigen::Vector3d::UnitZ().dot(edge_se);
        double hw = Eigen::Vector3d::UnitZ().dot(edge_sw);
        double lu = edge_se.norm();
        double ll = edge_sw.norm();

//        double H = mu * gravity_const * hu + ml * gravity_const * hl + 0.5 * b_const * pow((M_PI - alpha), 2) +
//                   w_const * pow(joints[5] - 0, 2);

        double H = (1 + he) / (4 * lu) + (1 + hw) / (4 * (lu + ll)) + 0.5 * b_const * pow((M_PI - alpha), 2);
        H += w_const * pow(joints[5] - 0., 2);
//        PLOGD << "Alpha : " << alpha;
//        PLOGD << "Humanoid : " << H;
        // gradient of hu
        Eigen::Matrix<double, 3, -1> Ju, Jl, Jf, Jcu, Jcl; //J6, J4, J2
        robot::Jacobian Js = J;
        const auto &Jw = Js.topRows(3);
        const auto &Jv = Js.bottomRows(3);

        //J6
        Js.rightCols(1).setZero();
        Ju = Jv - SO3::skew(cw) * Jw;

        //J4
        Js.rightCols(3).setZero();
        Jl = Jv - SO3::skew(ce) * Jw;

        //J2
        Js.rightCols(5).setZero();
        Jf = Jv - SO3::skew(cs) * Jw;

        // real Ju, Jl, Jf
        Jcu = R * (Jl + Jf);
        Jcl = R * (Ju + Jl);


        Jl = R * (Jl - Jf);
        Jf = R * (Ju - Jf);

        Eigen::RowVectorXd gAlpha(joints.size());
        if (cosval + 1. > SE3::eps) {
            Eigen::Matrix<double, 3, -1> gN1 = -SO3::skew(Eigen::Vector3d::UnitZ()) * Jf;
            gN1 = (Eigen::Matrix3d::Identity() / n1_norm - n_vertical * n_vertical.transpose() / n1_norm) * gN1;

            Eigen::Matrix<double, 3, -1> gN2 = SO3::skew(edge_sw) * Jl - SO3::skew(edge_se) * Jf;
            gN2 = (Eigen::Matrix3d::Identity() / n2_norm - n_current * n_current.transpose() / n2_norm) * gN2;

            if (alpha >= 0)
                gAlpha =  1. / std::sqrt(1. - cosval * cosval + 1e-6) *
                         (n_vertical.transpose() * gN2 + n_current.transpose() * gN1);
            else
                gAlpha = - 1. / std::sqrt(1. - cosval * cosval + 1e-6) *
                         (n_vertical.transpose() * gN2 + n_current.transpose() * gN1);
        } else {
            gAlpha.setZero();
        }
        assert(!gAlpha.hasNaN());
//        gH = 0.5 * gravity_const * (mu * Jcu.row(2) + ml * Jcl.row(2)) - b_const * (M_PI - alpha) * gAlpha;
        gH = 0.25 / lu * Jl.row(2) + 0.25 / (lu + ll) * Jf.row(2) - b_const * (M_PI - alpha) * gAlpha;
        gH[5] += 2. * w_const * (joints[5] - 0);
//        PLOGD << "gAlpha:" << gAlpha.transpose();
        return H;
    }

    std::map<double, Eigen::VectorXd> KinematicsScrew::getAllApproxIKSolutions(const Eigen::Isometry3d &cartesian_pose,
                                                                               const Eigen::VectorXd &seed) const {
        robot::ThetaList angles;
        std::map<double, Eigen::VectorXd> out;
        auto start = std::chrono::steady_clock::now();
        double elapsed = (double) (std::chrono::steady_clock::now() - start).count() / 1e9;
        uint i = 0;
        // at least try all seeds
        bool first_time = true;
        while (elapsed < time_out ) {
            elapsed = (double) (std::chrono::steady_clock::now() - start).count() / 1e9;
            angles = first_time ? seed : getRandomValidJoints(seed.size());
            if(first_time)
                first_time = false;
            double error;
            if (isSpace) {
                robot::numericalIKLMInSpace(cartesian_pose.matrix(), angles, M, SCREW_LIST, ik_tolerance_emog, ik_tolerance_ev, &error);
            } else {
                robot::numericalIKLMInBody(cartesian_pose.matrix(), angles, M, SCREW_LIST, ik_tolerance_emog, ik_tolerance_ev, &error);
            }
            wrap(angles);
            auto inLimits_ret = isInsideLimits(angles);
            if (inLimits_ret.IsOK()) {
                out[error] = angles;
            }
        }
        return out;
    }
}
