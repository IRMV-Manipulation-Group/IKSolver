/**
 * Some functions are copied from <Modern Robotics: Mechanics, Planning, and Control> Code library,
 * see https://github.com/NxRLab/ModernRobotics for more information.
 * These functions may renamed, if so will be denoted
 * Common Interface of screw-based robot functions
 * Author: YX.E.Z
 * Date: 2.12.2022
 */

#ifndef GCOPTER_ROBOT_HPP
#define GCOPTER_ROBOT_HPP

#include "SE3.hpp"
#include "sdqp.hpp"
#include "piqp/piqp.hpp"
//#ifdef QIQP_FOUND


namespace robot {
    using SE3::TMat;
    typedef Eigen::Matrix<double, 6, -1> ScrewList;
    typedef Eigen::Matrix<double, 6, -1> Jacobian;
    typedef Eigen::Matrix<double, -1, 1> ThetaList;
    typedef Eigen::Matrix<double, 1, -1> RowThetaList;
    typedef std::vector<Eigen::Matrix<double, 6, -1>> Hessian; ///< Every element denotes that \f(dJ / dq_i\f)
    typedef std::vector<Eigen::Matrix<double, 3, 3>> RJacobian; ///< Every element denotes that \f(dR / dq_i\f)
    typedef std::vector<Eigen::Matrix<double, 4, 4>> TJacobian; ///< Every element denotes that \f(dT / dq_i\f)

    typedef TMat (*lfk_t)(const ThetaList &thetaList);

    typedef std::function<TMat(const ThetaList &thetaList)> lfk_func;

    typedef std::function<RowThetaList(double &H, const ThetaList &thetaList, const Jacobian &J)> gradient_func;

    template<typename T>
    struct matrix_hash : std::unary_function<T, size_t> {
        std::size_t operator()(T const &matrix) const {
            // Note that it is oblivious to the storage order of Eigen matrix (column- or
            // row-major). It will give you the same hash value for two different matrices if they
            // are the transpose of each other in different storage order.
            size_t seed = 0;
            for (long i = 0; i < matrix.size(); ++i) {
                auto elem = *(matrix.data() + i);
                seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };


    /**
     * # From Mr library with "FKinSpace"
     * Computes forward kinematics in the space frame for an open chain robot
     * @param M The home configuration (position and orientation) of the end-effector
     * @param SList The joint screw axes in the space frame when the manipulator is at the home position,
     * in the format of a matrix with axes as the columns
     * @param thetaList A list of joint coordinates
     * @return A homogeneous transformation matrix representing the end-
     *        effector frame when the joints are at the specified coordinates
     *        (i.t.o Space Frame)
     */
    inline static TMat fkInSpace(const TMat &M, const ScrewList &SList, const ThetaList &thetaList) {
        assert(thetaList.size() == SList.cols());
        TMat ret = M;
        for (int i = thetaList.size() - 1; i >= 0; --i) {
            ret = SE3::Exp(SList.col(i) * thetaList[i]) * ret;
        }
        return ret;
    }

    /**
     * # From MR library with "FKInBody"
     * Computes forward kinematics in the body frame for an open chain robot
     * @param M The home configuration (position and orientation) of the end-
              effector
     * @param BList The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
     * @param thetaList  A list of joint coordinates
     * @return  A homogeneous transformation matrix representing the end-
             effector frame when the joints are at the specified coordinates
             (i.t.o Body Frame)
     */
    inline static TMat fkInBody(const TMat &M, const ScrewList &BList, const ThetaList &thetaList) {
        assert(thetaList.size() == BList.cols());
        TMat ret = M;
        for (int i = 0; i < thetaList.size(); ++i) {
            ret *= SE3::Exp(BList.col(i) * thetaList[i]);
        }
        return ret;
    }

    /**
     * # From MR library
     * Computes the body Jacobian for an open chain robot
     * @param BList The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
     * @param thetaList A list of joint coordinates
     * @return The body Jacobian corresponding to the inputs (6xn real
             numbers)
     */
    inline static Jacobian jacobianBody(const ScrewList &BList, const ThetaList &thetaList) {
        assert(thetaList.size() == BList.cols());
        Jacobian ret(6, BList.cols());
        ret.rightCols(1) = BList.rightCols(1);
        TMat T = TMat::Identity();
        for (int i = thetaList.size() - 2; i > -1; --i) {
            T *= SE3::Exp(BList.col(i + 1) * -thetaList[i + 1]);
            ret.col(i) = SE3::adjoint(T, false) * BList.col(i);
        }
        return ret;
    }

    /**
     *
     * Computes the space Jacobian for an open chain robot
     * @param BList The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
     * @param thetaList A list of joint coordinates
     * @param ret The output jacobian
     * @return The space Jacobian corresponding to the inputs (6xn real
             numbers)
     */
    inline static Jacobian jacobianBodyInPlace(const ScrewList &BList, const ThetaList &thetaList, Jacobian &ret) {
        assert(thetaList.size() == BList.cols());
        ret.resize(6, BList.cols());
        ret.rightCols(1) = BList.rightCols(1);
        TMat T = TMat::Identity();
        for (int i = thetaList.size() - 2; i > -1; --i) {
            T *= SE3::Exp(BList.col(i + 1) * -thetaList[i + 1]);
            ret.col(i) = SE3::adjoint(T, false) * BList.col(i);
        }
        return ret;
    }

    /**
     * # From MR library
     * Computes the space Jacobian for an open chain robot
     * @param SList The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
     * @param thetaList A list of joint coordinates
     * @return The space Jacobian corresponding to the inputs (6xn real
             numbers)
     */
    inline static Jacobian jacobianSpace(const ScrewList &SList, const ThetaList &thetaList) {
        assert(thetaList.size() == SList.cols());
        Jacobian ret(6, SList.cols());
        ret.leftCols(1) = SList.leftCols(1);
        TMat T = TMat::Identity();
        for (int i = 1; i < SList.cols(); ++i) {
            T *= SE3::Exp(SList.col(i - 1) * thetaList[i - 1]);
            ret.col(i) = SE3::adjoint(T, false) * SList.col(i);
        }
        return ret;
    }

    /**
     *
     * Computes the space Jacobian for an open chain robot
     * @param SList The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
     * @param thetaList A list of joint coordinates
     * @param ret The output jacobian
     * @return The space Jacobian corresponding to the inputs (6xn real
             numbers)
     */
    inline static void jacobianSpaceInPlace(const ScrewList &SList, const ThetaList &thetaList, Jacobian &ret) {
        assert(thetaList.size() == SList.cols());
        ret.leftCols(1) = SList.leftCols(1);
        TMat T = TMat::Identity();
        for (int i = 1; i < SList.cols(); ++i) {
            T *= SE3::Exp(SList.col(i - 1) * thetaList[i - 1]);
            ret.col(i) = SE3::adjoint(T, false) * SList.col(i);
        }
    }

    /**
     * Convert a MR-form space jacobian to a norm form jacobian
     * @param T The pose of the jacobian defined link w.r.t. world
     * @param space_jacobian jacobian in MR form
     * @param in_ee True for change into the end frame
     * @return Norm form jacobian
     */
    inline static Jacobian fromMRSpaceJacobian(const TMat &T, const Jacobian &space_jacobian, bool in_ee = false) {
        Jacobian ret = SE3::adjoint(T, true) * space_jacobian;
        ret.topRows(3).swap(ret.bottomRows(3));
        if (!in_ee) {
            ret = SE3::changeJacobianMatrix(T, false) * ret;
        }
        return ret;
    }

    /**
     * Inplace version of \fromMRSpaceJacobian
     * @param T The pose of the jacobian defined link w.r.t. world
     * @param space_jacobian jacobian in MR form
     * @param in_ee True for change into the end frame
     */
    inline static void fromMRSpaceJacobianInPlace(const TMat &T, Jacobian &space_jacobian, bool in_ee = false) {
        space_jacobian = SE3::adjoint(T, true) * space_jacobian;
        space_jacobian.topRows(3).swap(space_jacobian.bottomRows(3));
        if (!in_ee) {
            space_jacobian = SE3::changeJacobianMatrix(T, false) * space_jacobian;
        }
    }

    /**
     * Convert a MR-form space jacobian to a norm form jacobian
     * @param T The pose of the jacobian defined link w.r.t. world
     * @param body_jacobian jacobian in MR form
     * @param in_ee True for change into the end frame
     * @return Norm form jacobian
     */
    inline static Jacobian fromMRBodyJacobian(const TMat &T, const Jacobian &body_jacobian, bool in_ee = true) {
        Jacobian ret = body_jacobian;
        ret.topRows(3).swap(ret.bottomRows(3));
        if (!in_ee) {
            ret = SE3::changeJacobianMatrix(T, false) * ret;
        }
        return ret;
    }

    /**
     * Inplace version of \fromMRBodyJacobian
     * @param T The pose of the jacobian defined link w.r.t. world
     * @param space_jacobian jacobian in MR form
     * @param in_ee True for change into the end frame
     */
    inline static void fromMRBodyJacobianInPlace(const TMat &T, Jacobian &body_jacobian, bool in_ee = true) {
        body_jacobian.topRows(3).swap(body_jacobian.bottomRows(3));
        if (!in_ee) {
            body_jacobian = SE3::changeJacobianMatrix(T, false) * body_jacobian;
        }
    }

    /**
     * Convert a Norm-form space jacobian to a MR form jacobian
     * @param T The pose of the jacobian defined link w.r.t. world
     * @param space_jacobian jacobian in MR form
     * @param in_ee True for change into the end frame
     * @return MR form jacobian
     */
    inline static Jacobian fromNormSpaceJacobian(const TMat &T, const Jacobian &space_jacobian, bool in_ee = false) {
        Jacobian ret = SE3::changeJacobianMatrix(T, true) * space_jacobian;
        ret.topRows(3).swap(ret.bottomRows(3));
        if (!in_ee) {
            ret = SE3::adjoint(T, false) * ret;
        }
        return ret;
    }

    /**
     * Inplace version of \fromNormSpaceJacobian
     * @param T The pose of the jacobian defined link w.r.t. world
     * @param space_jacobian jacobian in MR form
     * @param in_ee True for change into the end frame
     * @return MR form jacobian
     */
    inline static void fromNormSpaceJacobianInPlace(const TMat &T, Jacobian &space_jacobian, bool in_ee = false) {
        space_jacobian = SE3::changeJacobianMatrix(T, true) * space_jacobian;
        space_jacobian.topRows(3).swap(space_jacobian.bottomRows(3));
        if (!in_ee) {
            space_jacobian = SE3::adjoint(T, false) * space_jacobian;
        }
    }

    /**
     * Convert a Norm-form space jacobian to a MR form jacobian
     * @param T The pose of the jacobian defined link w.r.t. world
     * @param body_jacobian jacobian in MR form
     * @param in_ee True for change into the end frame
     * @return MR form jacobian
     */
    inline static Jacobian fromNormBodyJacobian(const TMat &T, const Jacobian &body_jacobian, bool in_ee = false) {
        Jacobian ret = body_jacobian;
        ret.topRows(3).swap(ret.bottomRows(3));
        if (!in_ee) {
            ret = SE3::adjoint(T, false) * ret;
        }
        return ret;
    }

    /**
     * Inplace version of \fromNormBodyJacobian
     * @param T The pose of the jacobian defined link w.r.t. world
     * @param body_jacobian jacobian in MR form
     * @param in_ee True for change into the end frame
    */
    inline static void fromNormBodyJacobianInPlace(const TMat &T, Jacobian &body_jacobian, bool in_ee = false) {
        body_jacobian.topRows(3).swap(body_jacobian.bottomRows(3));
        if (!in_ee) {
            body_jacobian = SE3::adjoint(T, false) * body_jacobian;
        }
    }

    /**
     * Compute Screw list through fk function
     * @param fk_function function handle of forward kinematics
     * @param SList Output Screw list
     * @param body True for computing screw w.r.t. body
     * @param joint_numbers the total joint numbers
     * @return Initial condition
     */
    inline static TMat
    computeScrewList(lfk_t fk_function, ScrewList &SList, bool body = false, const int joint_numbers = 7) {
        SList.resize(6, joint_numbers);
        ThetaList q(joint_numbers);
        q.setZero();
        TMat M = SE3::inv(fk_function(q));
        TMat Ti;
        for (int i = 0; i < joint_numbers; ++i) {
            q.setZero();
            q[i] = 1.;
            Ti = fk_function(q);
            SList.col(i) = SE3::Log(Ti * M);
        }
        M = SE3::inv(M);
        if (body) {
            SList = SE3::adjoint(M, true) * SList;
        }
        return M;
    }

    /**
     * Compute Screw list through fk function
     * @param fk_function function handle of forward kinematics
     * @param SList Output Screw list
     * @param body True for computing screw w.r.t. body
     * @param joint_numbers the total joint numbers
     * @return Initial condition
     */
    inline static TMat
    computeScrewList(lfk_func fk_function, ScrewList &SList, bool body = false, const int joint_numbers = 7) {
        SList.resize(6, joint_numbers);
        ThetaList q(joint_numbers);
        q.setZero();
        TMat M = SE3::inv(fk_function(q));
        TMat Ti;
        for (int i = 0; i < joint_numbers; ++i) {
            q.setZero();
            q[i] = 1.;
            Ti = fk_function(q);
            SList.col(i) = SE3::Log(Ti * M);
        }
        M = SE3::inv(M);
        if (body) {
            SList = SE3::adjoint(M, true) * SList;
        }
        return M;
    }

    /**
     * Compute analytical jacobian d([r, x])/d(q)
     * @param T The pose of the jacobian defined link w.r.t. world
     * @param BList The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
     * @param thetaList  A list of joint coordinates
     * @return analytical jacobian d([r, x])/d(q)
     */
    inline static Jacobian analyticalJacobianBody(const TMat &T, const ScrewList &BList, const ThetaList &thetaList) {
        Jacobian ret = jacobianBody(BList, thetaList);
        const auto &R = T.topLeftCorner<3, 3>();
        Eigen::Matrix3d LJI = SO3::leftJacobianInverse(SO3::Log(R));
        ret.topRows(3) = LJI * ret.topRows(3);
        ret.bottomRows(3) = R * ret.bottomRows(3);
        return ret;
    }

    /**
     * Compute analytical jacobian d([r, x])/d(q)
     * @param T The pose of the jacobian defined link w.r.t. world
     * @param BList The joint screw axes in the space frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
     * @param thetaList  A list of joint coordinates
     * @return analytical jacobian d([r, x])/d(q)
     */
    inline static Jacobian analyticalJacobianSpace(const TMat &T, const ScrewList &SList, const ThetaList &thetaList) {
        Jacobian ret = SE3::adjoint(T, true) * jacobianSpace(SList, thetaList);
        const auto &R = T.topLeftCorner<3, 3>();
        Eigen::Matrix3d LJI = SO3::leftJacobianInverse(SO3::Log(R));
        ret.topRows(3) = LJI * ret.topRows(3);
        ret.bottomRows(3) = R * ret.bottomRows(3);
        return ret;
    }

    /**
     * Compute numerical solution of inverse kinematics, Newton-Raphson method
     * @param T The query end-effector pose in SE3;
     * @param angles The joint angles
     * @param M The home configuration
     * @param SList The screw list in the body frame
     * @param eomg error thresh for omega
     * @param ev error thresh for v
     * @return The theta list whose fk result equals to T
     */
    inline static bool
    numericalIKInBody(const TMat &T, ThetaList &angles, const TMat &M, const ScrewList &BList, double eomg, double ev) {
        int i = 0;
        int max_iterations = 50;
        TMat Tfk = fkInBody(M, BList, angles);
        TMat Tdiff = SE3::inv(Tfk) * T;
        SE3::TVec Vb = SE3::Log(Tdiff);
        bool err = (Vb.head(3).norm() > eomg || Vb.tail(3).norm() > ev);
        Jacobian Jb;
        while (err && i++ < max_iterations) {
            Jb = jacobianBody(BList, angles);
            angles += Jb.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Vb);
            // iterate
            Tfk = fkInBody(M, BList, angles);
            Tdiff = SE3::inv(Tfk) * T;
            Vb = SE3::Log(Tdiff);
            err = (Vb.head(3).norm() > eomg || Vb.tail(3).norm() > ev);
        }
        for (int j = 0; j < angles.size(); ++j) {
            auto &angle = angles[j];
            angle = std::abs(angle) < SE3::eps ? 0. : angle;
        }
        return !err;
    }

    /**
     * Compute numerical solution of inverse kinematics, Levenberg-Marquardt method
     * @param T The query end-effector pose in SE3;
     * @param angles The joint angles
     * @param M The home configuration
     * @param SList The screw list in the body frame
     * @param eomg error thresh for omega
     * @param ev error thresh for v
     * @return The theta list whose fk result equals to T
     */
    inline static bool
    numericalIKLMInBody(const TMat &T, ThetaList &angles, const TMat &M, const ScrewList &BList, double eomg,
                        double ev, double* output_error = nullptr) {
        int i = 0;
        int max_iterations = 50;
        TMat Tfk = fkInBody(M, BList, angles);
        TMat Tdiff = SE3::inv(Tfk) * T;
        SE3::TVec Vb = SE3::Log(Tdiff);
        double omg = Vb.head(3).norm(), trans = Vb.tail(3).norm();
        double error = 0.5 * (omg * omg + trans * trans);
        bool err = (omg > eomg || trans > ev);
        Jacobian Jb;
        Eigen::Matrix<double, -1, 6> JbT;
        Eigen::MatrixXd H;
        Eigen::MatrixXd I(angles.size(), angles.size());
        I.setIdentity();
        Eigen::VectorXd g(angles.size());
        while (err && i++ < max_iterations) {
            Jb = jacobianBody(BList, angles);
            JbT = Jb.transpose();
            g = JbT * Vb;
            H = JbT * Jb + I * (error * 0.1);
            angles += H.inverse() * g;
            // iterate
            Tfk = fkInBody(M, BList, angles);
            Tdiff = SE3::inv(Tfk) * T;
            Vb = SE3::Log(Tdiff);

            omg = Vb.head(3).norm();
            trans = Vb.tail(3).norm();
            error = 0.5 * (omg * omg + trans * trans);
            err = (omg > eomg || trans > ev);
        }
        if(output_error)
            *output_error = error;
        for (int j = 0; j < angles.size(); ++j) {
            auto &angle = angles[j];
            angle = std::abs(angle) < SE3::eps ? 0. : angle;
        }
        return !err;
    }

    /**
     * Compute numerical solution of inverse kinematics Newton-Raphson method
     * @param T The query end-effector pose in SE3;
     * @param angles The joint angles
     * @param M The home configuration
     * @param SList The screw list in the space space
     * @param eomg error thresh for omega
     * @param ev error thresh for v
     * @return The theta list whose fk result equals to T
     */
    inline static bool
    numericalIKInSpace(const TMat &T, ThetaList &angles, const TMat &M, const ScrewList &SList, double eomg,
                       double ev) {
        int i = 0;
        int max_iterations = 50;
        TMat Tfk = fkInSpace(M, SList, angles);
        TMat Tdiff = T * SE3::inv(Tfk);
        SE3::TVec Vs = SE3::Log(Tdiff);
        bool err = (Vs.head(3).norm() > eomg || Vs.tail(3).norm() > ev);
        Jacobian Js;
        while (err && i++ < max_iterations) {
            Js = jacobianSpace(SList, angles);
            angles += Js.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Vs);
            // iterate
            Tfk = fkInSpace(M, SList, angles);
            Tdiff = T * SE3::inv(Tfk);
            Vs = SE3::Log(Tdiff);
            err = (Vs.head(3).norm() > eomg || Vs.tail(3).norm() > ev);
        }
        for (int j = 0; j < angles.size(); ++j) {
            auto &angle = angles[j];
            angle = std::abs(angle) < SE3::eps ? 0. : angle;
        }
        return !err;
    }

    inline static void pesudoInverse(const Eigen::MatrixXd &J, Eigen::MatrixXd &J_pinv) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::VectorXd sigma = svd.singularValues();
        Eigen::MatrixXd sigma_inv(J.cols(), J.rows());
        sigma_inv.setZero();
        for (int i = 0; i < sigma.size(); ++i) {
            sigma_inv(i, i) = sigma[i] > SE3::eps ? 1. / sigma[i] : 0.;
        }

        J_pinv = svd.matrixV() * sigma_inv * svd.matrixU().transpose();
    }

    inline static void getHessian(const Jacobian &J, Hessian &H) {
        const Eigen::Matrix<double, 3, -1> &Ja = J.topRows(3);
        const Eigen::Matrix<double, 3, -1> &Jw = J.bottomRows(3);
        const auto m = J.cols();

        H.resize(m);
        for(auto&Hi : H){
            Hi.resize(6, m);
            Hi.setZero();
        }

        for (int j = 0; j < m; ++j) {
            auto &Hj = H[j];
            const auto R = SO3::skew(Jw.col(j));
            for (int i = j; i < m; ++i) {
                Hj.col(i).head(3) = R * Ja.col(i);
                Hj.col(i).tail(3) = R * Jw.col(i);
                if (i != j)
                    H[i].col(j).head(3) = Hj.col(i).head(3);
            }
        }
    }
    /**
     * Compute human-liked numerical solution of inverse kinematics Levenberg-Marquardt method
     * @param getHumanoidGradient The function to compute gradient of humanoid index, this function must return the same size of theta list
     * @param T The query end-effector pose in SE3;
     * @param angles The joint angles
     * @param M The home configuration
     * @param SList The screw list in the space space
     * @param emog error thresh for omega
     * @param ev error thresh for v
     * @return The theta list whose fk result equals to T
     */
    inline static bool
    humanoidIKLMInSpace(const gradient_func &getHumanoidGradient, const TMat &T, ThetaList &angles, const TMat &M,
                        const ScrewList &SList, double emog, double ev, double* output_error = nullptr) {
        int i = 0;
        int max_iterations = 400;
        TMat Tfk = fkInSpace(M, SList, angles);
        TMat Tdiff = T * SE3::inv(Tfk);
        SE3::TVec Vs = SE3::Log(Tdiff);
        double omg = Vs.head(3).norm(), trans = Vs.tail(3).norm();
        double error = 0.5 * (omg * omg + trans * trans);
        bool err = (omg > emog || trans > ev);

        Jacobian Js;
        Eigen::Matrix<double, -1, 6> JsT;
        Eigen::RowVectorXd gH(angles.size());
        Eigen::MatrixXd JH_PINV, J_PINV;
        Eigen::MatrixXd H, I(angles.size(), angles.size()), N1;
        I.setIdentity();
        Eigen::VectorXd g(angles.size());
        double HI, HI_last;
        while (i++ < max_iterations) {
            Js = jacobianSpace(SList, angles);
            JsT = Js.transpose();

            HI_last = HI;
            gH = getHumanoidGradient(HI, angles, Js);  // 自定义函数，计算H_index的梯度
            if (!err && std::abs(HI - HI_last) / std::abs(HI) < 1e-4) {
                break;
            }
            // 计算误差项
            g = JsT * Vs;
            H = JsT * Js + I * (error * 0.1);

            pesudoInverse(Js, J_PINV);
            N1 = (I - J_PINV * Js);

            pesudoInverse(gH, JH_PINV);
            // 更新关节角度
            angles += H.inverse() * g - N1 * JH_PINV * HI;

            //wrap to -2pi, 2pi
            for (int j = 0; j < angles.size(); ++j) {
                angles[j] = fmod(angles[j], 2. * M_PI);
            }

            // 迭代
            Tfk = fkInSpace(M, SList, angles);
            Tdiff = T * SE3::inv(Tfk);
            Vs = SE3::Log(Tdiff);
            omg = Vs.head(3).norm();
            trans = Vs.tail(3).norm();
            error = 0.5 * (omg * omg + trans * trans);
            err = (omg > emog || trans > ev);
        }
        if(output_error)
            *output_error = error;
        for (int j = 0; j < angles.size(); ++j) {
            auto &angle = angles[j];
            angle = std::abs(angle) < SE3::eps ? 0. : angle;
        }
        return !err;
    }

    /**
     * Compute numerical solution of inverse kinematics Levenberg-Marquardt method
     * @param T The query end-effector pose in SE3;
     * @param angles The joint angles
     * @param M The home configuration
     * @param SList The screw list in the space space
     * @param emog error thresh for omega
     * @param ev error thresh for v
     * @return The theta list whose fk result equals to T
     */
    inline static bool
    numericalIKLMInSpace(const TMat &T, ThetaList &angles, const TMat &M, const ScrewList &SList, double emog,
                         double ev, double* output_error = nullptr) {
        int i = 0;
        int max_iterations = 50;
        TMat Tfk = fkInSpace(M, SList, angles);
        TMat Tdiff = T * SE3::inv(Tfk);
        SE3::TVec Vs = SE3::Log(Tdiff);
        double omg = Vs.head(3).norm(), trans = Vs.tail(3).norm();
        double error = 0.5 * Vs.norm();
        bool err = (omg > emog || trans > ev);
        Jacobian Js; Js.resize(6, angles.size());
        Eigen::Matrix<double, -1, 6> JsT;
        Eigen::MatrixXd H;
        Eigen::MatrixXd I(angles.size(), angles.size());
        I.setIdentity();
        Eigen::VectorXd g(angles.size());
        while (err && i++ < max_iterations) {
            jacobianSpaceInPlace(SList, angles, Js);
            JsT = Js.transpose();
            g = JsT * Vs;
            H = JsT * Js + I * (error * 0.1);
            angles.noalias() += H.inverse() * g;
            // iterate
            Tfk = fkInSpace(M, SList, angles);
            Tdiff = T * SE3::inv(Tfk);
            Vs = SE3::Log(Tdiff);
            omg = Vs.head(3).norm();
            trans = Vs.tail(3).norm();
            error = 0.5 * Vs.norm();
            err = (omg > emog || trans > ev);
        }
        if(output_error)
            *output_error = error;
        for (int j = 0; j < angles.size(); ++j) {
            auto &angle = angles[j];
            angle = std::abs(angle) < SE3::eps ? 0. : angle;
        }
        return !err;
    }

    inline static double singleGradient(const double &u, const double &l, const double &x, const double &d) {
        assert(d > 0. && d <= (u - l) * 0.5 && u > l);
        double temp1 = u - x;
        double temp2 = x - l;
        temp1 = temp1 < 0 ? 1e-3 : temp1;
        temp2 = temp2 < 0 ? 1e-3 : temp2;
        double temp3 = std::log(d);

        double scale = std::abs(x - (u + l) * 0.5);
        double zeta = (temp2 > d && temp1 > d) ? 0. : scale;

        double g = (std::log(temp2) - temp3) / temp1 - (std::log(temp1) - temp3) / temp2;
        return g * zeta;
    }

    inline static void enforceLimits(double &val, double min, double max) {
        val = fmod(val, 2 * M_PI);
        while (val > max) {
            val -= 2 * M_PI;
        }

        // If the joint_value is less than the min, add 2 * PI until it is more than the min
        while (val < min) {
            val += 2 * M_PI;
        }
    }

    inline static Eigen::VectorXd
    computeQNull(const Eigen::VectorXd &angles, const double &jd, const Eigen::Matrix<double, -1, 2> &limits) {
        Eigen::VectorXd q_null(angles.size());
        for (int j = 0; j < angles.size(); ++j) {
            double qj = angles[j];
            const double &uj = limits.col(0)[j];
            const double &lj = limits.col(1)[j];
            enforceLimits(qj, -M_PI, M_PI);
            if (qj - lj <= jd) {
                q_null[j] = ((qj - lj) - jd);
                q_null[j] = -q_null[j] * q_null[j];
                q_null[j] /= jd * jd;
            } else if (uj - qj <= jd) {
                q_null[j] = ((uj - qj) - jd);
                q_null[j] = q_null[j] * q_null[j];
                q_null[j] /= jd * jd;
            } else {
                q_null[j] = 0.;
            }
        }
        return -q_null;
    }

    /**
     * Compute numerical solution of inverse kinematics Levenberg-Marquardt method
     * @param T The query end-effector pose in SE3;
     * @param angles The joint angles
     * @param M The home configuration
     * @param SList The screw list in the space space
     * @param limits The given joint limits
     * @param eomg error thresh for omega
     * @param ev error thresh for v
     * @return The theta list whose fk result equals to T
     */
    inline static bool
    numericalIKLMWithLimitsInSpace(const TMat &T, ThetaList &angles, const TMat &M, const ScrewList &SList,
                                   const Eigen::Matrix<double, -1, 2> &limits, double eomg,
                                   double ev) {
        const int n = angles.size();
        assert(n == limits.rows() && angles.size() == SList.cols());
        int i = 0;
        int max_iterations = 50;
        TMat Tfk = fkInSpace(M, SList, angles);
        TMat Tdiff = T * SE3::inv(Tfk);
        SE3::TVec Vs = SE3::Log(Tdiff);
        double omg = Vs.head(3).norm(), trans = Vs.tail(3).norm();
        double error = 0.5 * Vs.norm();
        bool err = (omg > eomg || trans > ev);

        Jacobian Js;
        Eigen::Matrix<double, -1, 6> JsT;
        Eigen::MatrixXd J_PINV;
        Eigen::MatrixXd H;
        Eigen::MatrixXd I(n, n);
        I.setIdentity();
        Eigen::VectorXd g(n), q_null(n);
        while (err && i++ < max_iterations) {
            Js = jacobianSpace(SList, angles);
            JsT = Js.transpose();
            pesudoInverse(Js, J_PINV);
            H = JsT * Js + I * (error * 0.1);
            g = JsT * Vs;
            q_null = (I - J_PINV * Js) * computeQNull(angles, 0.3, limits);
            angles += H.inverse() * g + q_null;
            // iterate
            Tfk = fkInSpace(M, SList, angles);
            Tdiff = T * SE3::inv(Tfk);
            Vs = SE3::Log(Tdiff);
            omg = Vs.head(3).norm();
            trans = Vs.tail(3).norm();
            error = 0.5 * Vs.norm();
            err = (omg > eomg || trans > ev);
        }
        for (int j = 0; j < angles.size(); ++j) {
            auto &angle = angles[j];
            angle = std::abs(angle) < SE3::eps ? 0. : angle;
        }
        return !err;
    }

    inline static double solveQP(Eigen::VectorXd & g, const double error, const Eigen::VectorXd& q, const robot::Jacobian& J, const Eigen::VectorXd & v,
                                 const Eigen::Matrix<double, -1, 3> &limits, const TMat&Tfk, const Eigen::VectorXd* optional_term = nullptr){
        int n = q.size();
        int m = J.rows();
        Eigen::MatrixXd P(n, n);
        P.setIdentity();
        //P.block(n, n, m, m).diagonal().setConstant(60. / error);
        Eigen::VectorXd c(n);
        c.setZero();
        if(optional_term)
            c.head(n) = *optional_term;
        else{
            robot::Jacobian Jb = SE3::adjoint(Tfk, true) * J;
            Jb.topRows(3).swap(Jb.bottomRows(3));
            Eigen::MatrixXd R = Jb * Jb.transpose();
            Eigen::MatrixXd RI = R.inverse();
            Eigen::VectorXd RI_VEC = Eigen::Map<Eigen::VectorXd>(RI.data(), RI.size(), 1);
            double index = std::sqrt(R.determinant());
            if(index != 0.){
                robot::Hessian H_M;
                robot::getHessian(Jb, H_M);
                for (int i = 0; i < n; ++i) {
                    Eigen::MatrixXd H = H_M[i];
                    H = Jb * H.transpose();
                    Eigen::VectorXd H_VEC = Eigen::Map<Eigen::VectorXd>(H.data(), H.size(), 1);
                    c[i] = -index * RI_VEC.dot(H_VEC);
                }
            }
        }

        //equal constraint
        Eigen::MatrixXd A(m, n);
        A.setZero();
        A.leftCols(n) = J;
        //A.rightCols(m).diagonal().setConstant(-1);
        Eigen::VectorXd b(m);
        b = v;

        Eigen::MatrixXd G(2 * m, n);
        G.setZero();

        Eigen::VectorXd h(2 * m);
        h.setZero();

        Eigen::VectorXd x_lb(n);
        x_lb.head(n) = -limits.col(2) * 0.99;
        //x_lb.tail(m).setConstant(-std::numeric_limits<double>::infinity());
        Eigen::VectorXd x_ub(n);
        x_ub = -x_lb;
        //handle joint limits
        for (int i = 0; i < n; ++i) {
            const double &qu = limits.col(0)[i];
            const double &ql = limits.col(1)[i];

            x_ub[i] = std::min((qu - q[i]), x_ub[i]);
            x_lb[i] = std::max((ql - q[i]), x_lb[i]);
        }
        piqp::DenseSolver<double> solver;


        solver.setup(P, c, A, b, G, h, x_lb, x_ub);
        piqp::Status status = solver.solve();
        if (status != piqp::PIQP_SOLVED || solver.result().x.head(n).hasNaN())
            return INFINITY;
        g = solver.result().x.head(n);
        return 0;
    }

    /**
     * Compute numerical solution of inverse kinematics Levenberg-Marquardt method
     * @param T The query end-effector pose in SE3;
     * @param angles The joint angles
     * @param M The home configuration
     * @param SList The screw list in the space space
     * @param limits The given joint limits
     * @param eomg error thresh for omega
     * @param ev error thresh for v
     * @return The theta list whose fk result equals to T
     */
    inline static bool
    numericalIKQPInSpace(const TMat &T, ThetaList &angles, const TMat &M, const ScrewList &SList,
                         const Eigen::Matrix<double, -1, 3> &limits, double eomg,
                         double ev, double* output_error = nullptr) {
        const int n = angles.size();
        assert(n == limits.rows() && angles.size() == SList.cols());
        int i = 0;
        int max_iterations = 50;
        TMat Tfk = fkInSpace(M, SList, angles);
        TMat Tdiff = T * SE3::inv(Tfk);
        SE3::TVec Vs = SE3::Log(Tdiff);
        double omg = Vs.head(3).norm(), trans = Vs.tail(3).norm();
        bool err = (omg > eomg || trans > ev);
        double error = Vs.norm();
        Jacobian Js(6, n);
        Eigen::VectorXd g(n);
        while (err && i++ < max_iterations) {
            jacobianSpaceInPlace(SList, angles, Js);
            double ret = solveQP(g, error, angles, Js, Vs, limits, Tfk);
            if(ret == INFINITY)
                break;
            angles.noalias() += g;
            // iterate
            Tfk = fkInSpace(M, SList, angles);
            Tdiff = T * SE3::inv(Tfk);
            Vs = SE3::Log(Tdiff);
            omg = Vs.head(3).norm();
            trans = Vs.tail(3).norm();
            err = (omg > eomg || trans > ev);
            error = Vs.norm();
        }
        if(output_error)
            *output_error = error;
        for (int j = 0; j < angles.size(); ++j) {
            auto &angle = angles[j];
            angle = std::abs(angle) < SE3::eps ? 0. : angle;
        }
        return !err;
    }
    inline static bool
    humanoidIKQPInSpace( const gradient_func &getHumanoidGradient, const TMat &T, ThetaList &angles, const TMat &M, const ScrewList &SList,
                         const Eigen::Matrix<double, -1, 3> &limits, double eomg,
                         double ev, double* output_error = nullptr) {
        const int n = angles.size();
        assert(n == limits.rows() && angles.size() == SList.cols());
        int i = 0;
        int max_iterations = 100;
        TMat Tfk = fkInSpace(M, SList, angles);
        TMat Tdiff = T * SE3::inv(Tfk);
        SE3::TVec Vs = SE3::Log(Tdiff);
        double omg = Vs.head(3).norm(), trans = Vs.tail(3).norm();
        bool err = (omg > eomg || trans > ev);
        double error = Vs.norm();
        Jacobian Js(6, n);
        Eigen::VectorXd g(n);
        Eigen::VectorXd gH(n);
        double H;
        while (err && i++ < max_iterations) {
            jacobianSpaceInPlace(SList, angles, Js);
            if(getHumanoidGradient != nullptr)
                gH = getHumanoidGradient(H, angles, Js).transpose();// 自定义函数，计算H_index的梯度
            const Eigen::VectorXd* gHPtr = getHumanoidGradient == nullptr ? nullptr : &gH;
            double ret = solveQP(g, error, angles, Js, Vs, limits, Tfk, gHPtr);
            if(ret == INFINITY)
                break;
            angles.noalias() += g;
            // iterate
            Tfk = fkInSpace(M, SList, angles);
            Tdiff = T * SE3::inv(Tfk);
            Vs = SE3::Log(Tdiff);
            omg = Vs.head(3).norm();
            trans = Vs.tail(3).norm();
            err = (omg > eomg || trans > ev);
            error = Vs.norm();
        }
        if(output_error)
            *output_error = error;
        for (int j = 0; j < angles.size(); ++j) {
            auto &angle = angles[j];
            angle = std::abs(angle) < SE3::eps ? 0. : angle;
        }
        return !err;
    }
}


#endif //GCOPTER_ROBOT_HPP
