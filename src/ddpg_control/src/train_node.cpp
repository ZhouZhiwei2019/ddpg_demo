#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include "ddpg_control/actor_network.h"
using namespace ddpg_control;
Eigen::VectorXd state(9);
ros::Publisher cmd_pub;
ActorNetwork* actor;
void state_callback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  // 填充state: 示例，仅x,y,z填入
  state(0) = 0.5; // depth
  state(1) = msg->pose.position.x;
  state(2) = msg->pose.position.y;
  state(3) = msg->pose.position.z;
  state(4) = 0.0; // yaw
  state(5) = 1.0; // target x
  state(6) = 1.0; // target y
  state(7) = 1.0; // target z
  state(8) = 0.0; // target yaw
  Eigen::VectorXd action = actor->forward(state);
  geometry_msgs::Twist cmd;
  cmd.linear.x = action(0);
  cmd.linear.y = action(1);
  cmd.linear.z = action(2);
  cmd.angular.z = action(3);
  cmd_pub.publish(cmd);
}
int main(int argc, char** argv) {
  ros::init(argc, argv, "ddpg_train_node");
  ros::NodeHandle nh;
  actor = new ActorNetwork(9, 4);
  cmd_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
  ros::Subscriber sub = nh.subscribe("/uav_pose", 10, state_callback);
  ros::spin();
  return 0;
}
