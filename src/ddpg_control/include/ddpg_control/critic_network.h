#ifndef DDPG_CONTROL_CRITIC_NETWORK_H
#define DDPG_CONTROL_CRITIC_NETWORK_H

#include <Eigen/Dense>
#include <string>

namespace ddpg_control {

class CriticNetwork {
public:
  CriticNetwork(int state_dim, int action_dim);

  double forward(const Eigen::VectorXd& state, const Eigen::VectorXd& action);
  void backward(const Eigen::VectorXd& state, const Eigen::VectorXd& action, double td_error);
  void soft_update(CriticNetwork& target, double tau);
  void save(const std::string& path) const;
  void load(const std::string& path);

private:
  int state_dim_, action_dim_;
  Eigen::MatrixXd weights_;
  Eigen::VectorXd biases_;
};

} // namespace ddpg_control

#endif // DDPG_CONTROL_CRITIC_NETWORK_H
