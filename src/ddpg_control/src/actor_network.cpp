#include "ddpg_control/actor_network.h"
namespace ddpg_control {
ActorNetwork::ActorNetwork(int state_dim, int action_dim) {
  // 根据需求初始化权重
  W1_ = Eigen::MatrixXd::Random(400, state_dim);
  b1_ = Eigen::VectorXd::Zero(400);
  W2_ = Eigen::MatrixXd::Random(300, 400);
  b2_ = Eigen::VectorXd::Zero(300);
  W3_ = Eigen::MatrixXd::Random(action_dim, 300);
  b3_ = Eigen::VectorXd::Zero(action_dim);
}

Eigen::VectorXd ActorNetwork::forward(const Eigen::VectorXd& state) {
  Eigen::VectorXd x1 = relu(W1_ * state + b1_);
  Eigen::VectorXd x2 = relu(W2_ * x1 + b2_);
  Eigen::VectorXd x3 = W3_ * x2 + b3_;
  return tanh_activate(x3);
}
}