#ifndef DDPG_CONTROL_ACTOR_NETWORK_H
#define DDPG_CONTROL_ACTOR_NETWORK_H

#include <Eigen/Dense>
#include <string>

namespace ddpg_control {

class ActorNetwork {
public:
  ActorNetwork(int input_dim, int output_dim);

  Eigen::VectorXd forward(const Eigen::VectorXd& state);
  void backward(const Eigen::VectorXd& state, double grad);
  void soft_update(ActorNetwork& target, double tau);
  void save(const std::string& path) const;
  void load(const std::string& path);

private:
  int input_dim_, output_dim_;
  Eigen::MatrixXd weights_;
  Eigen::VectorXd biases_;
};

} // namespace ddpg_control

#endif // DDPG_CONTROL_ACTOR_NETWORK_H
