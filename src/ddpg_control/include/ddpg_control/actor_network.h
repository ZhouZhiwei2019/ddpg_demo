#ifndef DDPG_CONTROL_ACTOR_NETWORK_H
#define DDPG_CONTROL_ACTOR_NETWORK_H
#include <Eigen/Dense>
#include "utils.h"
namespace ddpg_control {
class ActorNetwork {
public:
  ActorNetwork(int state_dim, int action_dim);
  Eigen::VectorXd forward(const Eigen::VectorXd& state);
private:
  Eigen::MatrixXd W1_, W2_, W3_;
  Eigen::VectorXd b1_, b2_, b3_;
};
}
#endif