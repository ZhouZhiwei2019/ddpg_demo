#include "ddpg_control/ddpg_agent.h"
#include <geometry_msgs/TwistStamped.h>
#include <fstream>

namespace ddpg_control {

DDPGAgent::DDPGAgent(int state_dim, int action_dim, bool train_mode)
  : state_dim_(state_dim), action_dim_(action_dim), train_mode_(train_mode),
    actor_(state_dim, action_dim), actor_target_(state_dim, action_dim),
    critic_(state_dim, action_dim), critic_target_(state_dim, action_dim),
    buffer_(100000),
    current_state_(Eigen::VectorXd::Zero(state_dim)),
    last_state_(Eigen::VectorXd::Zero(state_dim)),
    last_action_(Eigen::VectorXd::Zero(action_dim)),
    noise_dist_(0.0, 0.2) {

  actor_target_ = actor_;
  critic_target_ = critic_;

  px4_sub_ = nh_.subscribe("/uav/odom", 1, &DDPGAgent::px4_odom_callback, this);
  target_sub_ = nh_.subscribe("/target/odom", 1, &DDPGAgent::target_odom_callback, this);
  cmd_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("/uav/cmd_vel", 1);
  loop_timer_ = nh_.createTimer(ros::Duration(0.1), &DDPGAgent::loop_callback, this);
}

Eigen::VectorXd DDPGAgent::select_action(const Eigen::VectorXd& state) {
  Eigen::VectorXd action = actor_.forward(state);
  if (train_mode_) {
    for (int i = 0; i < action.size(); ++i) {
      action[i] += noise_dist_(generator_);
    }
  }
  return action;
}

void DDPGAgent::store_transition(const Eigen::VectorXd& state,
                                 const Eigen::VectorXd& action,
                                 double reward,
                                 const Eigen::VectorXd& next_state) {
  Transition t{state, action, reward, next_state};
  buffer_.add(t);
}

void DDPGAgent::train() {
  if (!train_mode_ || buffer_.size() < batch_size_) return;

  auto batch = buffer_.sample(batch_size_);

  for (const auto& t : batch) {
    Eigen::VectorXd next_action = actor_target_.forward(t.next_state);
    double target_q = t.reward + gamma_ * critic_target_.forward(t.next_state, next_action);
    double current_q = critic_.forward(t.state, t.action);
    double td_error = current_q - target_q;
    critic_.backward(t.state, t.action, td_error);

    Eigen::VectorXd action_pred = actor_.forward(t.state);
    double actor_q = critic_.forward(t.state, action_pred);
    actor_.backward(t.state, actor_q);
  }

  actor_.soft_update(actor_target_, tau_);
  critic_.soft_update(critic_target_, tau_);
}

void DDPGAgent::save_model(const std::string& dir) {
  actor_.save(dir + "/actor.txt");
  actor_target_.save(dir + "/actor_target.txt");
  critic_.save(dir + "/critic.txt");
  critic_target_.save(dir + "/critic_target.txt");
}

void DDPGAgent::load_model(const std::string& dir) {
  actor_.load(dir + "/actor.txt");
  actor_target_.load(dir + "/actor_target.txt");
  critic_.load(dir + "/critic.txt");
  critic_target_.load(dir + "/critic_target.txt");
}

void DDPGAgent::px4_odom_callback(const nav_msgs::Odometry::ConstPtr& msg) {
  current_state_.segment(0, 6) << msg->pose.pose.position.x, msg->pose.pose.position.y,
                                 msg->pose.pose.position.z,
                                 msg->twist.twist.linear.x, msg->twist.twist.linear.y,
                                 msg->twist.twist.linear.z;
  px4_received_ = true;
}

void DDPGAgent::target_odom_callback(const nav_msgs::Odometry::ConstPtr& msg) {
  current_state_.segment(6, 6) << msg->pose.pose.position.x, msg->pose.pose.position.y,
                                 msg->pose.pose.position.z,
                                 msg->twist.twist.linear.x, msg->twist.twist.linear.y,
                                 msg->twist.twist.linear.z;
  target_received_ = true;
}

void DDPGAgent::loop_callback(const ros::TimerEvent&) {
  if (!px4_received_ || !target_received_) return;

  Eigen::VectorXd action = select_action(current_state_);

  geometry_msgs::TwistStamped cmd;
  cmd.twist.linear.x = action[0];
  cmd.twist.linear.y = action[1];
  cmd.twist.linear.z = action[2];
  cmd.twist.angular.z = action[3];
  cmd_pub_.publish(cmd);

  double pos_error = (current_state_.segment(0, 3) - current_state_.segment(6, 3)).norm();
  double vel_error = (current_state_.segment(3, 3) - current_state_.segment(9, 3)).norm();
  double reward = - pos_error - 0.1 * vel_error;

  if (last_state_.size() > 0 && last_action_.size() > 0 && train_mode_) {
    store_transition(last_state_, last_action_, reward, current_state_);
    train();
  }

  last_state_ = current_state_;
  last_action_ = action;
}

} // namespace ddpg_control
