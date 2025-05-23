#ifndef DDPG_CONTROL_DDPG_AGENT_H
#define DDPG_CONTROL_DDPG_AGENT_H

#include "actor_network.h"
#include "critic_network.h"
#include "replay_buffer.h"
#include <Eigen/Dense>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <random>

namespace ddpg_control {

class DDPGAgent {
public:
  DDPGAgent(int state_dim, int action_dim, bool train_mode);

  Eigen::VectorXd select_action(const Eigen::VectorXd& state);
  void store_transition(const Eigen::VectorXd& state,
                        const Eigen::VectorXd& action,
                        double reward,
                        const Eigen::VectorXd& next_state);
  void train();

  void save_model(const std::string& dir);
  void load_model(const std::string& dir);

  void px4_odom_callback(const nav_msgs::Odometry::ConstPtr& msg);
  void target_odom_callback(const nav_msgs::Odometry::ConstPtr& msg);
  void loop_callback(const ros::TimerEvent&);

private:
  int state_dim_;
  int action_dim_;
  bool train_mode_;
  ActorNetwork actor_;
  ActorNetwork actor_target_;
  CriticNetwork critic_;
  CriticNetwork critic_target_;
  ReplayBuffer buffer_;

  Eigen::VectorXd current_state_;
  Eigen::VectorXd last_state_;
  Eigen::VectorXd last_action_;
  ros::NodeHandle nh_;
  ros::Subscriber px4_sub_, target_sub_;
  ros::Publisher cmd_pub_;
  ros::Timer loop_timer_;

  double gamma_ = 0.99;
  double tau_ = 0.005;
  int batch_size_ = 64;
  bool px4_received_ = false;
  bool target_received_ = false;

  std::default_random_engine generator_;
  std::normal_distribution<double> noise_dist_;
};

} // namespace ddpg_control

#endif // DDPG_CONTROL_DDPG_AGENT_H