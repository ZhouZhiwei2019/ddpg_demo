#ifndef DDPG_CONTROL_REPLAY_BUFFER_H
#define DDPG_CONTROL_REPLAY_BUFFER_H
#include <deque>
#include <vector>
#include <Eigen/Dense>
namespace ddpg_control {
struct Transition {
  Eigen::VectorXd state;
  Eigen::VectorXd action;
  double reward;
  Eigen::VectorXd next_state;
};

class ReplayBuffer {
public:
  ReplayBuffer(size_t max_size);
  void add(const Transition& t);
  std::vector<Transition> sample(size_t batch_size);
private:
  std::deque<Transition> buffer_;
  size_t max_size_;
};
}
#endif