
/**
 * A general KinematicsBase interface
 * Author: YX.E.Z
 * Date: 2025/07/12
 */

#ifndef DUAL_ARM_APP_KINEMATICS_BASE_HPP
#define DUAL_ARM_APP_KINEMATICS_BASE_HPP

#include <memory>
#include <map>
#include <vector>
#include <Eigen/Dense>
#include "irmv/bot_common/state/error_code.h"

namespace bot_kinematics {

    enum KinematicsImplType {
        LM = 0,
        QP = 1
    };

    class KinematicsBase : public std::enable_shared_from_this<KinematicsBase> {
    public:
        KinematicsBase() = default;

        virtual ~KinematicsBase() = default;

    protected:
        double time_out = 2e-3;
    public:
        /**
         * @brief Get the robot motion limits in joint space
         *
         * @return Eigen::Matrix<double, -1, 5> max_limits, min_limits, velocity, acceleration and jerk
         * limits of each joint DoF
         */
        virtual const Eigen::Matrix<double, -1, 5> &getJointMotionLimits() const = 0;

        /**
         * @brief Compute robot TCP cartesian pose with forward KinematicsBase
         *
         * @param joint_position robot joint position
         * @return bot_common::OK for get right FK;
         */
        std::pair<bot_common::ErrorInfo, Eigen::Isometry3d>
        getFK(const Eigen::VectorXd &joint_position) const {
            return getFK(joint_position, -1);
        }

        /**
         * @brief Compute robot link cartesian pose with forward KinematicsBase
         *
         * @param joint_position robot joint position
         * @param link_index the index of the required link
         * @return bot_common::OK for get right FK;
         */
        virtual std::pair<bot_common::ErrorInfo, Eigen::Isometry3d>
        getFK(const Eigen::VectorXd &joint_position, int link_index) const = 0;

        /**
         * @brief Compute robot joint positions of given TCP cartesian pose, return
         * the most human-like one relative to joint_seed
         *
         * @param cartesian_pose robot TCP cartesian pose
         * @param joint_seed joint position close to desired cartesian target pose,
         * set to empty value to use current robot joint position
         * @param max_dist the maximum acceptable distance from joint seed to
         * potential IK solutions, solution which is far than this will be rejected, all solutions will be accepted when
         * max_dist is set as non-positive;
         * @return bot_common::OK for get right IK;
         */
        virtual std::pair<bot_common::ErrorInfo, Eigen::VectorXd>
        getNearestIK(const Eigen::Isometry3d &cartesian_pose,
                     const Eigen::VectorXd &joint_seed, double max_dist) const = 0;

        /**
         * @brief Compute the nearest approximate IK solution
         *
         * @param cartesian_pose robot TCP cartesian pose
         * @param joint_seed joint position close to desired cartesian target pose
         * @return A pair containing the distance and the joint positions
         */
        virtual std::pair<double, Eigen::VectorXd>
        getNearestApproxIK(const Eigen::Isometry3d &cartesian_pose,
                           const Eigen::VectorXd &joint_seed) const = 0;

        /**
         * @brief Compute all approximate IK solutions of given TCP cartesian pose
         *
         * @param cartesian_pose robot TCP cartesian pose
         * @param seed The given joint seed for numerical methods
         * @return A map of distances to joint positions
         */
        virtual std::map<double, Eigen::VectorXd>
        getAllApproxIKSolutions(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd& seed) const = 0;

        /**
         * @brief Compute all IK solutions of given TCP cartesian pose
         *
         * @param cartesian_pose robot TCP cartesian pose
         * @param joint_seeds The given joint seeds for numerical methods
         * @param max_dist the maximum acceptable distance from joint seed to
         * potential IK solutions, solution which is far than this will be rejected, all solutions will be accepted when
         * max_dist is set as non-positive
         * @return A vector of joint positions
         */
        virtual std::vector<Eigen::VectorXd>
        getAllIKSolutions(const Eigen::Isometry3d &cartesian_pose, const std::vector<Eigen::VectorXd> &joint_seeds,
                          double max_dist) const = 0;

        /**
         * @brief Compute all IK solutions of given TCP cartesian pose with cost
         *
         * @param cartesian_pose robot TCP cartesian pose
         * @param joint_seeds The given joint seeds for numerical methods
         * @param max_dist the maximum acceptable distance from joint seed to
         * potential IK solutions, solution which is far than this will be rejected, all solutions will be accepted when
         * max_dist is set as non-positive
         * @return A map of distances to joint positions
         */
        virtual std::map<double, Eigen::VectorXd>
        getAllIKSolutionsWithCost(const Eigen::Isometry3d &cartesian_pose, const std::vector<Eigen::VectorXd> &joint_seeds,
                                  double max_dist) const = 0;

        /**
         * @brief Compute corresponding geometric jacobian for given joints
         *
         * @param joints The given joints
         * @param jacob The output jacobian
         * @param in_ee true for computing the jacobian w.r.t the end-effector frame
         * @return bot_common::OK for success
         */
        virtual bot_common::ErrorInfo
        getGeometricJacobian(const Eigen::VectorXd &joints, Eigen::Matrix<double, 6, -1> &jacob, bool in_ee) const = 0;

        /**
         * @brief Compute corresponding analytical jacobian for given joints
         *
         * @param joints The given joints
         * @param jacob The output jacobian
         * @return bot_common::OK for success
         */
        virtual bot_common::ErrorInfo
        getAnalyticalJacobian(const Eigen::VectorXd &joints, Eigen::Matrix<double, 6, -1> &jacob) const = 0;

        /**
         * @brief Compute one set of random joints inside the joint limits
         *
         * @param dim The dimension of the joints
         * @return Eigen::VectorXd random joints
         */
        Eigen::VectorXd
        getRandomValidJoints(const long &dim) const;

        /**
         * @brief Get random valid joints nearby a seed
         *
         * @param seed The seed joint positions
         * @param maxDist The maximum distance from the seed
         * @return The output value won't exceed seed +/- maxDist
         */
        Eigen::VectorXd
        getRandomValidJointsNearby(const Eigen::VectorXd &seed, const double &maxDist) const;

        /**
         * @brief Check if the given joints are inside the joint motion limits
         *
         * @param joints The given joints (will be wrapped into -pi, pi)
         * @return bot_common::OK for inside
         */
        bot_common::ErrorInfo
        isInsideLimits(const Eigen::VectorXd &joints) const;

        /**
         * @brief Enforce limits of joint angles
         *
         * @param val The input and output value
         * @param min The given lower limits
         * @param max The given upper limits
         */
        static void enforceLimits(double &val, double min = -M_PI, double max = M_PI) ;

        /**
         * @brief Set a global valid checking callback
         *
         * @param callback The callback function
         */
        void setGlobalValidCheckingCallback(std::function<bool(const Eigen::VectorXd &thetaList)> callback);

        /**
         * @brief Clear the global valid checking callback
         */
        void clearGlobalValidCheckingCallback();


        /**
         * @brief Compute all IK solutions of given elbow pose and wrist pose
         * @param elbow_pose robot elbow pose
         * @param wrist_pose robot elbow pose;
         * @return std::vector<JointPosition> the IK solutions
         */
        virtual std::pair<double, Eigen::VectorXd>
        getIKPiecewise(const Eigen::Isometry3d &elbow_pose, const Eigen::Isometry3d &wrist_pose,
                       const Eigen::VectorXd &CurrentJoints) const = 0;


    protected:
        /**
         * @brief Wrap joint values within limits
         *
         * @param joints The joint values
         * @param qu The upper limit
         * @param ql The lower limit
         */
        bool wrap(Eigen::VectorXd &joints, double qu = M_PI, double ql=-M_PI) const;

    protected:
        std::function<bool(const Eigen::VectorXd &thetaList)> validCallback_;
    };

    /**
     * @typedef KinematicsPtr
     * @brief A shared pointer to a KinematicsBase object
     */
    typedef std::shared_ptr<KinematicsBase> KinematicsPtr;

    /**
     * @typedef KinematicsUniquePtr
     * @brief A unique pointer to a KinematicsBase object
     */
    typedef std::unique_ptr<KinematicsBase> KinematicsUniquePtr;
}
#endif //DUAL_ARM_APP_KINEMATICS_BASE_HPP
