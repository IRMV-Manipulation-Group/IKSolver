#include "bot_kinematics/kinematics_base.hpp"
#include <random>
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace bot_kinematics {
    Eigen::VectorXd KinematicsBase::getRandomValidJoints(const long &dim) const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<double> dis(0., 1.0);
        static Eigen::Matrix<double, -1, 5> limits = this->getJointMotionLimits();
        Eigen::VectorXd random = Eigen::VectorXd::NullaryExpr(dim, 1, [&]() { return dis(gen); });
        assert(limits.rows() == dim);
        const auto &max_p_limit = limits.col(0);
        const auto &min_p_limit = limits.col(1);
        Eigen::VectorXd result = min_p_limit + random.cwiseProduct((max_p_limit - min_p_limit));
        return result;
    }

    Eigen::VectorXd
    KinematicsBase::getRandomValidJointsNearby(const Eigen::VectorXd &seed, const double &maxDist) const {
        assert(seed.size() > 0);
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<double> dis(0., 1.);
        static Eigen::Matrix<double, -1, 5> limits = this->getJointMotionLimits();
        Eigen::VectorXd random = Eigen::VectorXd::NullaryExpr(seed.size(), 1, [&]() { return dis(gen); });
        Eigen::VectorXd result;
        const auto &max_p_limit = limits.col(0);
        const auto &min_p_limit = limits.col(1);
        if (maxDist <= 0.) {
            result = min_p_limit + random.cwiseAbs().cwiseProduct(max_p_limit - min_p_limit);
        } else {
            Eigen::VectorXd shrink_min = min_p_limit;
            Eigen::VectorXd shrink_max = max_p_limit;
            for(int i = 0 ; i < shrink_min.size(); ++i){
                shrink_min[i] = seed[i] - std::min(seed[i] - shrink_min[i], maxDist);
                shrink_max[i] = seed[i] + std::min(shrink_max[i] - seed[i], maxDist);
            }
            result = shrink_min +  random.cwiseProduct(shrink_max - shrink_min);
        }

        return result;

    }

    void KinematicsBase::enforceLimits(double &val, double min, double max) const {
        val = fmod(val, 2 * M_PI);
        while (val > max) {
            val -= 2 * M_PI;
        }

        // If the joint_value is less than the min, add 2 * PI until it is more than the min
        while (val < min) {
            val += 2 * M_PI;
        }
    }

    bot_common::ErrorInfo KinematicsBase::isInsideLimits(const Eigen::VectorXd &joints) const {
        const auto &JOINT_MOTION_LIMITS = getJointMotionLimits();
        assert(joints.size() == JOINT_MOTION_LIMITS.rows());
        const auto &upper_limit = JOINT_MOTION_LIMITS.col(0);
        const auto &lower_limit = JOINT_MOTION_LIMITS.col(1);
        Eigen::VectorXd q = joints;
        double tolerance = 1e-5;
        for (int j = 0; j < q.size(); ++j) {
            auto &qj = q[j];
            qj = fmod(qj, 2. * M_PI);

            if (qj > upper_limit[j] + tolerance || qj < lower_limit[j] - tolerance) {
                return {bot_common::ErrorCode::OutLimitation,
                        fmt::format("the given joints {} of {}th joint is outside of the limits", joints.transpose(), j)};
            }
        }

        return bot_common::ErrorInfo::OK();
    }

    void KinematicsBase::wrap(Eigen::VectorXd &joints, double qu, double ql) const {
        const auto &JOINT_MOTION_LIMITS = getJointMotionLimits();
        assert(joints.size() == JOINT_MOTION_LIMITS.rows());

        for (int j = 0; j < joints.size(); ++j) {
            auto &qj = joints[j];

            // enforce joint value into [-pi, pi]
            enforceLimits(qj, ql, qu);
        }
    }

    void KinematicsBase::setGlobalValidCheckingCallback(std::function<bool(const Eigen::VectorXd &)> callback) {
        validCallback_ = callback;
    }

    void KinematicsBase::clearGlobalValidCheckingCallback() {
        validCallback_ = nullptr;
    }
}
