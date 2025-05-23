#ifndef DDPG_CONTROL_CRITIC_NETWORK_H
#define DDPG_CONTROL_CRITIC_NETWORK_H
#include <Eigen/Dense>
#include "utils.h"
namespace ddpg_control {
class CriticNetwork {
public:
  CriticNetwork(int state_dim, int action_dim);
  double forward(const Eigen::VectorXd& state, const Eigen::VectorXd& action);
private:
  Eigen::MatrixXd W1_, W2_, W3_;
  Eigen::VectorXd b1_, b2_, b3_;
};
}
#endif