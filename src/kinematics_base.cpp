
#include "IKSolver/kinematics_base.h"
#include <random>
#include <utility>
#include <irmv/third_party/fmt/format.h>
#include <irmv/third_party/fmt/ostream.h>
#include <irmv/spoon_math/spoon_math.h>
namespace bot_kinematics {

    inline Eigen::VectorXd generateRandomVector(const long &dim) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<double> dis(0., 1.0);
        return Eigen::VectorXd::NullaryExpr(dim, 1, [&]() { return dis(gen); });
    }

    Eigen::VectorXd KinematicsBase::getRandomValidJoints(const long &dim) const {
        const Eigen::Matrix<double, -1, 5>& limits = this->getJointMotionLimits();
        Eigen::VectorXd random = generateRandomVector(dim);
        return limits.col(1) + random.cwiseProduct(limits.col(0) - limits.col(1));
    }

    Eigen::VectorXd
    KinematicsBase::getRandomValidJointsNearby(const Eigen::VectorXd &seed, const double &maxDist) const {
        Eigen::VectorXd random = generateRandomVector(seed.size());
        const Eigen::Matrix<double, -1, 5> &limits = this->getJointMotionLimits();
        const auto &max_p_limit = limits.col(0);
        const auto &min_p_limit = limits.col(1);

        if (maxDist <= 0.) {
            return min_p_limit + random.cwiseAbs().cwiseProduct(max_p_limit - min_p_limit);
        }

        Eigen::VectorXd shrink_min = min_p_limit;
        Eigen::VectorXd shrink_max = max_p_limit;
        for (int i = 0; i < shrink_min.size(); ++i) {
            shrink_min[i] = seed[i] - std::min(seed[i] - shrink_min[i], maxDist);
            shrink_max[i] = seed[i] + std::min(shrink_max[i] - seed[i], maxDist);
        }
        return shrink_min + random.cwiseProduct(shrink_max - shrink_min);
    }

    void KinematicsBase::enforceLimits(double &val, double min, double max) {
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
        const auto &limits = getJointMotionLimits();
        assert(joints.size() == limits.rows());
        const auto &upper = limits.col(0);
        const auto &lower = limits.col(1);
        Eigen::VectorXd q = joints;
        double tol = 1e-6;

        for (int i = 0; i < q.size(); ++i) {
            q[i] = fmod(q[i], 2.0 * M_PI);

            if (q[i] > upper[i] + tol || q[i] < lower[i] - tol) {
                return {bot_common::ErrorCode::OutLimitation,
                        fmt::format("the given joints {} of {}th joint is outside of the limits", joints.transpose(), i)};
            }
        }

        return bot_common::ErrorInfo::OK();
    }

    bool KinematicsBase::wrap(Eigen::VectorXd &raw, double qu, double ql) const {
        bool isWrapped = false;
        for (int i = 0; i < raw.size(); ++i) {
            if (raw[i] > M_PI) {
                raw[i] = fmod(raw[i] + M_PI, 2. * M_PI) - M_PI;
                isWrapped = true;
            } else if (raw[i] < -M_PI){
                raw[i] = fmod(raw[i] - M_PI, 2. * M_PI) + M_PI;
                isWrapped = true;
            }
        }
        return isWrapped;
    }

    void KinematicsBase::setGlobalValidCheckingCallback(std::function<bool(const Eigen::VectorXd &)> callback) {
        validCallback_ = std::move(callback);
    }

    void KinematicsBase::clearGlobalValidCheckingCallback() {
        validCallback_ = nullptr;
    }
}
