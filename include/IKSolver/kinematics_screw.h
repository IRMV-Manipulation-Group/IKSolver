
/**
 * A kinematics implementation using screw-theory
 * Author: YX.E.Z
 * Date: 2023/7/23
 */
#ifndef DUAL_ARM_APP_KINEMATICS_SCREW_HPP
#define DUAL_ARM_APP_KINEMATICS_SCREW_HPP

#include <irmv/bot_math/lie/robot.hpp>
#include "IKSolver/kinematics_base.h"
#include "irmv/bot_common/alg_factory/algorithm_factory.h"

namespace bot_kinematics {
    constexpr char KinematicsScrewName[] = "KinematicsScrewName";


    /**
     * @brief A kinematics implementation using screw-theory.
     */
    class KinematicsScrew : public KinematicsBase {
    public:
        /**
         * @brief Constructor that initializes the KinematicsScrew with a given configuration file.
         * @param screw_yml The path name to the motion limits config yml file.
         * @param impl_type The implementation type.
         */
        explicit KinematicsScrew(const std::string &screw_yml, KinematicsImplType impl_type = KinematicsImplType::LM);

        /**
         * @brief Deleted default constructor.
         */
        KinematicsScrew() = delete;

        /**
         * @brief Destructor.
         */
        ~KinematicsScrew() override;

    protected:
        robot::ScrewList SCREW_LIST; ///< List of screw axes.
        robot::TMat M; ///< Home configuration of the end-effector.
        std::vector<robot::TMat> Links; ///< List of transformation matrices for each link.
        Eigen::Matrix<double, -1, 5> JointMotionLimits; ///< Joint motion limits.
        double ik_tolerance_emog; ///< Tolerance for inverse kinematics (emog).
        double ik_tolerance_ev; ///< Tolerance for inverse kinematics (ev).
        double ik_tolerance_emog_elbow; ///< Tolerance for inverse kinematics (emog).
        double ik_tolerance_ev_elbow; ///< Tolerance for inverse kinematics (ev).
        double ik_tolerance_emog_wrist; ///< Tolerance for inverse kinematics (emog).
        double ik_tolerance_ev_wrist; ///< Tolerance for inverse kinematics (ev).
        double max_size; ///< Maximum size for some internal data structures.
        bool isHuman; ///< Flag indicating if the kinematics is human-like.
        KinematicsImplType impl_type; ///< The implementation type.
        Eigen::Matrix3d R; ///< Transformation from world to arm base.

    public:
        /**
         * @brief A convenient function to create a KinematicsScrew object.
         * @param screw_yml The path name to the motion limits config yml file.
         * @param impl_type The implementation type.
         * @return A unique pointer to the base class.
         */
        static KinematicsUniquePtr
        create(const std::string &screw_yml, KinematicsImplType impl_type = KinematicsImplType::LM);

        /**
         * @brief Get the robot motion limits in joint space.
         * @return Eigen::Matrix<double, -1, 5> max_limits, min_limits, velocity, acceleration and jerk limits of each joint DoF.
         */
        const Eigen::Matrix<double, -1, 5> &getJointMotionLimits() const override;

        /**
         * @brief Compute robot link cartesian pose with forward kinematics.
         * @param joint_position Robot joint position.
         * @param link_index The index of the required link.
         * @return A pair containing the error info and the cartesian pose.
         */
        std::pair<bot_common::ErrorInfo, Eigen::Isometry3d>
        getFK(const Eigen::VectorXd &joint_position, int link_index) const override;

        /**
         * @brief Compute robot joint positions of given TCP cartesian pose, return the most human-like one relative to joint_seed.
         * @param cartesian_pose Robot TCP cartesian pose.
         * @param joint_seed Joint position close to desired cartesian target pose, set to empty value to use current robot joint position.
         * @param max_dist The maximum acceptable distance from joint seed to potential IK solutions.
         * @return A pair containing the error info and the joint positions.
         */
        std::pair<bot_common::ErrorInfo, Eigen::VectorXd>
        getNearestIK(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd &joint_seed,
                     double max_dist) const override;

        /**
         * @brief Compute the nearest approximate IK solution.
         * @param cartesian_pose Robot TCP cartesian pose.
         * @param joint_seed Joint position close to desired cartesian target pose.
         * @return A pair containing the distance and the joint positions.
         */
        std::pair<double, Eigen::VectorXd>
        getNearestApproxIK(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd &joint_seed) const override;

        /**
         * @brief Compute all IK solutions of given TCP cartesian pose.
         * @param cartesian_pose Robot TCP cartesian pose.
         * @param joint_seeds The given joint seeds for numerical methods.
         * @param max_dist The maximum acceptable distance from joint seed to potential IK solutions.
         * @return A vector of joint positions.
         */
        std::vector<Eigen::VectorXd>
        getAllIKSolutions(const Eigen::Isometry3d &cartesian_pose, const std::vector<Eigen::VectorXd> &joint_seeds,
                          double max_dist) const override;

        /**
         * @brief Compute all IK solutions of given TCP cartesian pose with cost.
         * @param cartesian_pose Robot TCP cartesian pose.
         * @param joint_seeds The given joint seeds for numerical methods.
         * @param max_dist The maximum acceptable distance from joint seed to potential IK solutions.
         * @return A map of distances to joint positions.
         */
        std::map<double, Eigen::VectorXd> getAllIKSolutionsWithCost(const Eigen::Isometry3d &cartesian_pose,
                                                                    const std::vector<Eigen::VectorXd> &joint_seeds,
                                                                    double max_dist) const override;

        /**
         * @brief Compute corresponding analytical jacobian for given joints.
         * @param joints The given joints.
         * @param jacob The output jacobian.
         * @return Error info indicating success or failure.
         */
        bot_common::ErrorInfo getAnalyticalJacobian(const Eigen::VectorXd &joints,
                                                    Eigen::Matrix<double, 6, -1> &jacob) const override;

        /**
         * @brief Compute corresponding geometric jacobian for given joints.
         * @param joints The given joints.
         * @param jacob The output jacobian.
         * @param in_ee True for computing the jacobian w.r.t the end-effector frame.
         * @return Error info indicating success or failure.
         */
        bot_common::ErrorInfo
        getGeometricJacobian(const Eigen::VectorXd &joints, Eigen::Matrix<double, 6, -1> &jacob,
                             bool in_ee) const override;

        /**
         * @brief Compute all approximate IK solutions of given TCP cartesian pose.
         * @param cartesian_pose Robot TCP cartesian pose.
         * @param seed The given joint seed for numerical methods.
         * @return A map of distances to joint positions.
         */
        std::map<double, Eigen::VectorXd>
        getAllApproxIKSolutions(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd &seed) const override;


        /**
          * @brief Compute all IK solutions of given elbow pose and wrist pose
          * @param elbow_pose robot elbow pose
          * @param wrist_pose robot elbow pose;
          * @return std::vector<JointPosition> the IK solutions
          */
        std::pair<double, Eigen::VectorXd>
        getIKPiecewise(const Eigen::Isometry3d &elbow_pose, const Eigen::Isometry3d &wrist_pose,
                       const Eigen::VectorXd &CurrentJoints) const override;

    public:
        /**
         * @brief Compute the humanoid index for given joints.
         * @param joints The given joints.
         * @return The humanoid index.
         */
        double computeHumanoidIndex(const Eigen::VectorXd &joints) const;

        /**
         * @brief Compute the humanoid index and its gradient for given joints.
         * @param gH The gradient of the humanoid index.
         * @param joints The given joints.
         * @param J The jacobian matrix.
         * @return The humanoid index.
         */
        double computeHumanoidIndexAndGradient(Eigen::RowVectorXd &gH, const Eigen::VectorXd &joints,
                                               const robot::Jacobian &J) const;

        /**
         * @brief Implementation of nearest IK computation.
         * @param cartesian_pose Robot TCP cartesian pose.
         * @param joint_seed Joint position close to desired cartesian target pose.
         * @param max_dist The maximum acceptable distance from joint seed to potential IK solutions.
         * @return A pair containing the error info and the joint positions.
         */
        std::pair<bot_common::ErrorInfo, Eigen::VectorXd>
        getNearestIKImpl(const Eigen::Isometry3d &cartesian_pose,
                         const Eigen::VectorXd &joint_seed, double max_dist) const;

        /**
         * @brief Check if the kinematics is human-like.
         * @return True if the kinematics is human-like.
         */
        bool IsHuman() const { return isHuman; };

        /**
         * @brief Set the kinematics to be human-like.
         * @param human Flag indicating if the kinematics should be human-like.
         * @return True if the operation is successful.
         */
        bool setHuman(bool human) { isHuman = human; };

    };

    /**
     * @brief Register the KinematicsScrew algorithm.
     */
    inline bot_common::REGISTER_ALGORITHM(KinematicsBase, KinematicsScrewName, KinematicsScrew, const std::string&,
                                          KinematicsImplType);
}
#endif //DUAL_ARM_APP_KINEMATICS_SCREW_HPP
