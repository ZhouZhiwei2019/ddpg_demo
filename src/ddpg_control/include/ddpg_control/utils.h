#ifndef DDPG_CONTROL_UTILS_H
#define DDPG_CONTROL_UTILS_H
#include <Eigen/Dense>
namespace ddpg_control {
// 激活函数
inline Eigen::VectorXd tanh_activate(const Eigen::VectorXd& x) {
  return x.array().tanh();
}

inline Eigen::VectorXd relu(const Eigen::VectorXd& x) {
  return x.array().max(0.0);
}

}
#endif // DDPG_CONTROL_UTILS_H