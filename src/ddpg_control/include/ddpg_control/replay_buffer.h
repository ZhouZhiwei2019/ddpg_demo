#ifndef DDPG_CONTROL_REPLAY_BUFFER_H
#define DDPG_CONTROL_REPLAY_BUFFER_H

#include <Eigen/Dense>
#include <deque>
#include <random>

namespace ddpg_control {

struct Transition {
  Eigen::VectorXd state;
  Eigen::VectorXd action;
  double reward;
  Eigen::VectorXd next_state;
};

class ReplayBuffer {
public:
  explicit ReplayBuffer(size_t capacity);
  void add(const Transition& transition);
  std::vector<Transition> sample(size_t batch_size) const;
  size_t size() const;

private:
  size_t capacity_;
  std::deque<Transition> buffer_;
  mutable std::default_random_engine generator_;
};

} // namespace ddpg_control

#endif // DDPG_CONTROL_REPLAY_BUFFER_H
