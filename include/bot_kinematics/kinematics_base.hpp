/**
 * A general KinematicsBase interface
 * Author: YX.E.Z
 * Date: 2023/7/25
 */

#ifndef DUAL_ARM_APP_KINEMATICS_BASE_HPP
#define DUAL_ARM_APP_KINEMATICS_BASE_HPP

#include <memory>
#include <map>
#include <vector>
#include <Eigen/Dense>
#include "state/error_code.h"

namespace bot_kinematics {
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
           * @return Eigen::Matrix<double, -1, 5> max_limits, min_limits, velocity,acceleration and jerk
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
       * @param time_out The given time out for numerical methods;
       * @param max_dist the maximium acceptable distance from joint seed to
       * potential IK solutions, solution which is far than this will be rejected, all solutions will be accepted when
       * max_dist is set as non-positive;
       * @return bot_common::OK for get right IK;
       */

        virtual std::pair<bot_common::ErrorInfo, Eigen::VectorXd>
        getNearestIK(const Eigen::Isometry3d &cartesian_pose,
                    const Eigen::VectorXd &joint_seed, double max_dist) const = 0;

        virtual std::pair<double, Eigen::VectorXd>
        getNearestApproxIK(const Eigen::Isometry3d &cartesian_pose,
                     const Eigen::VectorXd &joint_seed) const = 0;

        /**
      * @brief Compute all IK solutions of given TCP cartesian pose
      * @param cartesian_pose robot TCP cartesian pose
      * @param joint_seeds The given joint seeds for numerical methods;
      * @param time_out The given time out for numerical methods;
      * @param max_dist the maximium acceptable distance from joint seed to
      * potential IK solutions, solution which is far than this will be rejected, all solutions will be accepted when
      * max_dist is set as non-positive;
      * @return std::vector<JointPosition>  all the IK solutions
      */
        virtual std::map<double, Eigen::VectorXd>
        getAllApproxIKSolutions(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd& seed) const = 0;

        /**
       * @brief Compute all IK solutions of given TCP cartesian pose
       * @param cartesian_pose robot TCP cartesian pose
       * @param joint_seeds The given joint seeds for numerical methods;
       * @param time_out The given time out for numerical methods;
       * @param max_dist the maximium acceptable distance from joint seed to
       * potential IK solutions, solution which is far than this will be rejected, all solutions will be accepted when
       * max_dist is set as non-positive;
       * @return std::vector<JointPosition>  all the IK solutions
       */
        virtual std::vector<Eigen::VectorXd>
        getAllIKSolutions(const Eigen::Isometry3d &cartesian_pose, const std::vector<Eigen::VectorXd> &joint_seeds,
                          double max_dist) const = 0;

        /**
       * @brief Compute all IK solutions of given TCP cartesian pose
       * @param cartesian_pose robot TCP cartesian pose
       * @param joint_seeds The given joint seeds for numerical methods;
       * @param time_out The given time out for numerical methods;
       * @param max_dist the maximium acceptable distance from joint seed to
       * potential IK solutions, solution which is far than this will be rejected, all solutions will be accepted when
       * max_dist is set as non-positive;
       * @return std::vector<JointPosition>  all the IK solutions
       */
        virtual std::map<double, Eigen::VectorXd>
        getAllIKSolutionsWithCost(const Eigen::Isometry3d &cartesian_pose, const std::vector<Eigen::VectorXd> &joint_seeds,
                          double max_dist) const = 0;

        /**
         * @brief Compute corresponding geometric jacobian for given joints
         * @param joints The given joints
         * @param jacob The output jacobian
         * @param in_ee true for computing the jacobian w.r.t the end-effector frame
         * @return bot_common::OK for success;
         */
        virtual bot_common::ErrorInfo
        getGeometricJacobian(const Eigen::VectorXd &joints, Eigen::Matrix<double, 6, -1> &jacob, bool in_ee) const = 0;

        /**
         * @brief Compute corresponding analytical jacobian for given joints
         * @param joints The given joints
         * @param jacob The output jacobian
         * @return bot_common::OK for success;
         */
        virtual bot_common::ErrorInfo
        getAnalyticalJacobian(const Eigen::VectorXd &joints, Eigen::Matrix<double, 6, -1> &jacob) const = 0;

        /**
         * @brief Compute one set of random joints in side the joint limits;
         * @return Eigen::VectorXd random joints
         */
        Eigen::VectorXd
        getRandomValidJoints(const long &dim) const;

        /**
         *
         * @param seed
         * @param maxDist
         * @return The output value won't exceed seed +/- maxDist
         */
        Eigen::VectorXd
        getRandomValidJointsNearby(const Eigen::VectorXd &seed, const double &maxDist) const;

        /**
         * @brief A method to check is the given joints inside the joint motion limits
         * @param joints The given joints (will be wrapped into -pi, pi)
         * @return bot_common::OK for inside;
         */
        bot_common::ErrorInfo
        isInsideLimits(const Eigen::VectorXd &joints) const;

        /**
         * @ enforce limits of joint angles
         * @param val The input and output value
         * @param min The given lower limits
         * @param max The given upper limits
         */
        void enforceLimits(double &val, double min = -M_PI, double max = M_PI) const;

        void setGlobalValidCheckingCallback(std::function<bool(const Eigen::VectorXd &thetaList)> callback);

        void clearGlobalValidCheckingCallback();

    protected:
        void wrap(Eigen::VectorXd &joints, double qu = M_PI, double ql=-M_PI) const;

    protected:
        std::function<bool(const Eigen::VectorXd &thetaList)> validCallback_;
    };

    typedef std::shared_ptr<KinematicsBase> KinematicsPtr;
    typedef std::unique_ptr<KinematicsBase> KinematicsUniquePtr;
}
#endif //DUAL_ARM_APP_KINEMATICS_BASE_HPP
