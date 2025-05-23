#include "ddpg_control/critic_network.h"
namespace ddpg_control {
CriticNetwork::CriticNetwork(int state_dim, int action_dim) {
  W1_ = Eigen::MatrixXd::Random(400, state_dim + action_dim);
  b1_ = Eigen::VectorXd::Zero(400);
  W2_ = Eigen::MatrixXd::Random(300, 400);
  b2_ = Eigen::VectorXd::Zero(300);
  W3_ = Eigen::MatrixXd::Random(1, 300);
  b3_ = Eigen::VectorXd::Zero(1);
}

double CriticNetwork::forward(const Eigen::VectorXd& state, const Eigen::VectorXd& action) {
  Eigen::VectorXd input(state.size() + action.size());
  input << state, action;
  Eigen::VectorXd x1 = relu(W1_ * input + b1_);
  Eigen::VectorXd x2 = relu(W2_ * x1 + b2_);
  Eigen::VectorXd x3 = W3_ * x2 + b3_;
  return x3(0);
}
}