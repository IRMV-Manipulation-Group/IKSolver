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

#ifndef IKSolver_INTERFACE_H
#define IKSolver_INTERFACE_H

#include <Eigen/Dense>
#include <vector>
#include "IKSolver/kinematics_base.h"

namespace bot_kinematics {
    class bot_kinematics;
}

class Solver_Interface {
    public:
        /**
         * @brief
         */
        Solver_Interface() = default;
        /**
         * @brief Destructor
         */
        ~Solver_Interface() = default;
    protected:
        bot_kinematics::KinematicsUniquePtr kin_;
    public:
        /**
         * @brief Set the kinematics ptr object
         * @param kin_ The kinematics ptr object
         */
        void initialize(const std::string yml_path, 
            bot_kinematics::KinematicsImplType impl_type = bot_kinematics::KinematicsImplType::LM);


        /**
         * @brief Get the robot motion limits in joint space.
         * @return Eigen::Matrix<double, -1, 5> max_limits, min_limits, velocity, acceleration and jerk limits of each joint DoF.
         */
        const Eigen::Matrix<double, -1, 5> &getJointMotionLimits();

        /**
         * @brief Compute robot link cartesian pose with forward kinematics.
         * @param joint_position Robot joint position.
         * @param link_index The index of the required link.
         * @return A pair containing the error info and the cartesian pose.
         */
        std::pair<std::pair<int, std::string>, Eigen::Isometry3d>
        getFK(const Eigen::VectorXd &joint_position, int link_index);

        /**
         * @brief Compute robot joint positions of given TCP cartesian pose, return the most human-like one relative to joint_seed.
         * @param cartesian_pose Robot TCP cartesian pose.
         * @param joint_seed Joint position close to desired cartesian target pose, set to empty value to use current robot joint position.
         * @param max_dist The maximum acceptable distance from joint seed to potential IK solutions.
         * @return A pair containing the error info and the joint positions.
         */
        std::pair<std::pair<int, std::string>, Eigen::VectorXd>
        getNearestIK(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd &joint_seed,
                        double max_dist);

        /**
         * @brief Compute the nearest approximate IK solution.
         * @param cartesian_pose Robot TCP cartesian pose.
         * @param joint_seed Joint position close to desired cartesian target pose.
         * @return A pair containing the distance and the joint positions.
         */
        std::pair<double, Eigen::VectorXd>
        getNearestApproxIK(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd &joint_seed) ;
        

         /**
         * @brief Compute all IK solutions of given TCP cartesian pose.
         * @param cartesian_pose Robot TCP cartesian pose.
         * @param joint_seeds The given joint seeds for numerical methods.
         * @param max_dist The maximum acceptable distance from joint seed to potential IK solutions.
         * @return A vector of joint positions.
         */
        std::vector<Eigen::VectorXd>
        getAllIKSolutions(const Eigen::Isometry3d &cartesian_pose, const std::vector<Eigen::VectorXd> &joint_seeds,
                          double max_dist);
        
        /**
         * @brief Compute all IK solutions of given TCP cartesian pose with cost.
         * @param cartesian_pose Robot TCP cartesian pose.
         * @param joint_seeds The given joint seeds for numerical methods.
         * @param max_dist The maximum acceptable distance from joint seed to potential IK solutions.
         * @return A map of distances to joint positions.
         */
        std::map<double, Eigen::VectorXd> getAllIKSolutionsWithCost(const Eigen::Isometry3d &cartesian_pose,
                                                                    const std::vector<Eigen::VectorXd> &joint_seeds,
                                                                    double max_dist) ;

        /**
         * @brief Compute corresponding analytical jacobian for given joints.
         * @param joints The given joints.
         * @param jacob The output jacobian.
         * @return Error info indicating success or failure.
         */
        std::pair<int, std::string> 
        getAnalyticalJacobian(const Eigen::VectorXd &joints,
                                                    Eigen::Matrix<double, 6, -1> &jacob) ;

        /**
         * @brief Compute corresponding geometric jacobian for given joints.
         * @param joints The given joints.
         * @param jacob The output jacobian.
         * @param in_ee True for computing the jacobian w.r.t the end-effector frame.
         * @return Error info indicating success or failure.
         */
        std::pair<int, std::string>
        getGeometricJacobian(const Eigen::VectorXd &joints, Eigen::Matrix<double, 6, -1> &jacob,
                             bool in_ee) ;

        /**
         * @brief Compute all approximate IK solutions of given TCP cartesian pose.
         * @param cartesian_pose Robot TCP cartesian pose.
         * @param seed The given joint seed for numerical methods.
         * @return A map of distances to joint positions.
         */
        std::map<double, Eigen::VectorXd>
        getAllApproxIKSolutions(const Eigen::Isometry3d &cartesian_pose, const Eigen::VectorXd &seed) ;


        /**
          * @brief Compute all IK solutions of given elbow pose and wrist pose
          * @param elbow_pose robot elbow pose
          * @param wrist_pose robot elbow pose;
          * @return std::vector<JointPosition> the IK solutions
          */
        std::pair<double, Eigen::VectorXd>
        getIKPiecewise(const Eigen::Isometry3d &elbow_pose, const Eigen::Isometry3d &wrist_pose,
                       const Eigen::VectorXd &CurrentJoints);

        /**
         * @brief Check if the kinematics is human-like.
         * @return True if the kinematics is human-like.
         */
        bool IsHuman() const;

        /**
         * @brief Set the kinematics to be human-like.
         * @param human Flag indicating if the kinematics should be human-like.
         * @return True if the operation is successful.
         */
        bool setHuman(bool human);


};

#endif //IKSolver_INTERFACE_H
