#ifndef RRT_PLANNER__RRT_PLANNER_HPP_
#define RRT_PLANNER__RRT_PLANNER_HPP_

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_core/global_planner.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "tf2_ros/buffer.h"
#include "visualization_msgs/msg/marker.hpp"

namespace rrt_planner
{

struct TreeNode
{
  double x;
  double y;
  int parent;  // index of the parent in the tree (-1 for the root)
};

class RRTPlanner : public nav2_core::GlobalPlanner
{
public:
  RRTPlanner() = default;
  ~RRTPlanner() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  nav_msgs::msg::Path createPlan(
    const geometry_msgs::msg::PoseStamped & start,
    const geometry_msgs::msg::PoseStamped & goal,
    std::function<bool()> cancel_checker) override;

private:
  bool isFree(double wx, double wy) const;
  bool segmentFree(double x0, double y0, double x1, double y1) const;
  void publishTree(const std::vector<TreeNode> & tree);

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Logger logger_{rclcpp::get_logger("RRTPlanner")};

  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_{nullptr};
  std::string global_frame_;
  std::string name_;

  rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::Marker>::SharedPtr tree_pub_;

  // Parameters
  double step_size_{0.3};        // [m] max extension per iteration
  double goal_tolerance_{0.2};   // [m] distance to goal to accept a solution
  double goal_bias_{0.05};       // probability of sampling the goal
  int max_iterations_{5000};
  int seed_{-1};                 // -1 -> random seed
  bool publish_tree_{true};

  std::mt19937 rng_;
};

}  // namespace rrt_planner

#endif  // RRT_PLANNER__RRT_PLANNER_HPP_
