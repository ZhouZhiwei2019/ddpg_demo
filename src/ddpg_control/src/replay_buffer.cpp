#include "ddpg_control/replay_buffer.h"
#include <random>
namespace ddpg_control 
{

ReplayBuffer::ReplayBuffer(size_t max_size) : max_size_(max_size) {}

void ReplayBuffer::add(const Transition& t) 
{
  if (buffer_.size() >= max_size_) buffer_.pop_front();
  buffer_.push_back(t);
}

std::vector<Transition> ReplayBuffer::sample(size_t batch_size) 
{
  std::vector<Transition> batch;
  std::sample(buffer_.begin(), buffer_.end(), std::back_inserter(batch), batch_size, std::mt19937{std::random_device{}()});
  return batch;
}
}