/**
 * Bridge between Eigen and std
 */

#ifndef DUAL_ARM_APP_UTILS_HPP
#define DUAL_ARM_APP_UTILS_HPP
#include <Eigen/Dense>
#include <vector>
#include <random>
namespace utils{
    typedef Eigen::Matrix<double, 7, 1> Vector7D;
    typedef Eigen::Matrix<double, 6, 1> Vector6D;

    static inline Eigen::Isometry3d convertToIsometry3d(const std::vector<double> &data_raw) {
        assert(data_raw.size() == 6 || data_raw.size() == 7);
        Eigen::Isometry3d result{Eigen::Isometry3d::Identity()};
        if (data_raw.size() == 6) {
            auto data = Eigen::Map<const Vector6D>(data_raw.data(), 6, 1);
            result.translation() = data.block<3, 1>(0, 0);
            auto euler = data.block<3, 1>(3, 0);
            result.linear() = Eigen::Matrix3d{Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ()) *
                                              Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) *
                                              Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX())};
            return result;
        }
        auto data = Eigen::Map<const Vector7D>(data_raw.data(), 7, 1);
        result.translation() = data.block<3, 1>(0, 0);
        result.linear() = Eigen::Quaterniond(data[6], data[3], data[4], data[5]).toRotationMatrix();
        return result;
    }

    static inline Eigen::Isometry3d convertToIsometry3d(const double* ptr, int n) {
        assert(n == 6 || n == 7);
        Eigen::Isometry3d result{Eigen::Isometry3d::Identity()};
        if (n == 6) {
            auto data = Eigen::Map<const Vector6D>(ptr, 6, 1);
            result.translation() = data.block<3, 1>(0, 0);
            auto euler = data.block<3, 1>(3, 0);
            result.linear() = Eigen::Matrix3d{Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ()) *
                                              Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY()) *
                                              Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX())};
            return result;
        }
        auto data = Eigen::Map<const Vector7D>(ptr, 7, 1);
        result.translation() = data.block<3, 1>(0, 0);
        result.linear() = Eigen::Quaterniond(data[6], data[3], data[4], data[5]).toRotationMatrix();
        return result;
    }

    static inline std::vector<double> covertToStdVector(const Eigen::VectorXd& data_raw){
        std::vector<double> ret(data_raw.size());
        memcpy(ret.data(), data_raw.data(), sizeof(double) * data_raw.size());
        return ret;
    }

    static inline void convertToStdVectorInplace(std::vector<double>& data_out, const Eigen::VectorXd& data_raw){
        data_out.resize(data_raw.size());
        memcpy(data_out.data(), data_raw.data(), sizeof(double) * data_raw.size());
    }

    static inline Eigen::VectorXd getUniformRandomVector(long size, double min, double max){
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<double> dis(min, max);
        return Eigen::VectorXd::NullaryExpr(size, 1, [&]() { return dis(gen); });
    }
}

#endif //DUAL_ARM_APP_UTILS_HPP
