/**
 * A kinematics implementation using screw-theory
 * Author: YX.E.Z
 * Date: 2023/7/23
 */
#ifndef DUAL_ARM_APP_KINEMATICS_SCREW_HPP
#define DUAL_ARM_APP_KINEMATICS_SCREW_HPP

#include "bot_kinematics/kinematics_base.hpp"
#include "alg_factory/algorithm_factory.h"

namespace robot{
    typedef Eigen::Matrix4d TMat;
    typedef Eigen::Matrix<double, 6, -1> ScrewList;
    typedef Eigen::Matrix<double, 6, -1> Jacobian;
}

namespace bot_kinematics {
    constexpr char KinematicsScrewName[] = "KinematicsScrewName";

    class KinematicsScrew : public KinematicsBase {
    public:
        explicit KinematicsScrew(const std::string &screw_yml);

        KinematicsScrew() = delete;

        ~KinematicsScrew() override;

    protected:
        robot::ScrewList SCREW_LIST;
        robot::TMat M;
        std::vector<robot::TMat> Links;
        bool isSpace;
        Eigen::Matrix<double, -1, 5> JointMotionLimits;
        double ik_tolerance_emog;
        double ik_tolerance_ev;
        double max_size;
        bool isHuman;

        Eigen::Matrix3d R; //transfer from world to arm base
    public:

        /**
         * @brief a convenient function to create
         * @param screw_yml The path name to the motion limits config yml file
         * @return A unique pointer to the base class
         */
        static KinematicsUniquePtr create(const std::string &screw_yml);

        /**
           * @brief Get the robot motion limits in joint space
           *
           * @return Eigen::Matrix<double, -1, 5> max_limits, min_limits, velocity,acceleration and jerk
           * limits of each joint DoF
           */
        const Eigen::Matrix<double, -1, 5> &getJointMotionLimits() const override;

        /**
         * @brief Compute robot link cartesian pose with forward KinematicsBase
         *
         * @param joint_position robot joint position
         * @param link_index the index of the required link
         * @return bot_common::OK for get right FK;
         */
        std::pair<bot_common::ErrorInfo, Eigen::Isometry3d>
        getFK(const Eigen::VectorXd &joint_position, int link_index) const override;


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
        std::pair<bot_common::ErrorInfo, Eigen::VectorXd>
        getNearestIK(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd &joint_seed,
                     double max_dist) const override;

        std::pair<double, Eigen::VectorXd>
        getNearestApproxIK(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd &joint_seed) const override;

        /**
        * @brief Compute all IK solutions of given TCP cartesian pose
        * @param cartesian_pose robot TCP cartesian pose
        * @param joint_seeds The given joint seed for numerical methods;
       * @param time_out The given time out for numerical methods;
       * @param max_dist the maximium acceptable distance from joint seed to
       * potential IK solutions, solution which is far than this will be rejected, all solutions will be accepted when
       * max_dist is set as non-positive;
        * @return std::vector<JointPosition>  all the IK solutions
        */
        std::vector<Eigen::VectorXd>
        getAllIKSolutions(const Eigen::Isometry3d &cartesian_pose, const std::vector<Eigen::VectorXd> &joint_seeds,
                          double max_dist) const override;


        /**
        * @brief Compute all IK solutions of given TCP cartesian pose
        * @param cartesian_pose robot TCP cartesian pose
        * @param joint_seeds The given joint seed for numerical methods;
       * @param time_out The given time out for numerical methods;
       * @param max_dist the maximium acceptable distance from joint seed to
       * potential IK solutions, solution which is far than this will be rejected, all solutions will be accepted when
       * max_dist is set as non-positive;
        * @return std::vector<JointPosition>  all the IK solutions
        */
        std::map<double, Eigen::VectorXd> getAllIKSolutionsWithCost(const Eigen::Isometry3d &cartesian_pose,
                                                                    const std::vector<Eigen::VectorXd> &joint_seeds,
                                                                    double max_dist) const override;

        /**
         * @brief Compute corresponding analytical jacobian for given joints
         * @param joints The given joints
         * @param jacob The output jacobian
         * @return bot_common::OK for success;
         */
        bot_common::ErrorInfo getAnalyticalJacobian(const Eigen::VectorXd &joints,
                                                    Eigen::Matrix<double, 6, -1> &jacob) const override;

        /**
         * @brief Compute corresponding geometric jacobian for given joints
         * @param joints The given joints
         * @param jacob The output jacobian
         * @param in_ee true for computing the jacobian w.r.t the end-effector frame
         * @return bot_common::OK for success;
         */
        bot_common::ErrorInfo
        getGeometricJacobian(const Eigen::VectorXd &joints, Eigen::Matrix<double, 6, -1> &jacob,
                             bool in_ee) const override;

        std::map<double, Eigen::VectorXd>
        getAllApproxIKSolutions(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd &seed) const override;

    public:
        double computeHumanoidIndex(const Eigen::VectorXd &joints) const;

        double computeHumanoidIndexAndGradient(Eigen::RowVectorXd &gH, const Eigen::VectorXd &joints,
                                               const robot::Jacobian &J) const;

        std::pair<bot_common::ErrorInfo, Eigen::VectorXd>
        getNearestIKImpl(const Eigen::Isometry3d &cartesian_pose,
                         const Eigen::VectorXd &joint_seed, double max_dist) const;

        bool IsHuman() const { return isHuman; };

        bool setHuman(bool human) { isHuman = true; };

    };

    inline bot_common::REGISTER_ALGORITHM(KinematicsBase, KinematicsScrewName, KinematicsScrew, const std::string&);
}
#endif //DUAL_ARM_APP_KINEMATICS_SCREW_HPP
