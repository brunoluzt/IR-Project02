#include "rrt_planner/rrt_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "nav2_core/planner_exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace rrt_planner
{

void RRTPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  auto node = parent.lock();
  if (!node) {
    throw std::runtime_error("RRTPlanner: unable to lock parent node");
  }
  node_ = parent;
  clock_ = node->get_clock();
  logger_ = node->get_logger();
  tf_ = tf;
  name_ = name;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros->getCostmap();
  global_frame_ = costmap_ros->getGlobalFrameID();

  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".step_size", rclcpp::ParameterValue(0.3));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".goal_tolerance", rclcpp::ParameterValue(0.2));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".goal_bias", rclcpp::ParameterValue(0.05));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".max_iterations", rclcpp::ParameterValue(5000));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".seed", rclcpp::ParameterValue(-1));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".publish_tree", rclcpp::ParameterValue(true));

  node->get_parameter(name_ + ".step_size", step_size_);
  node->get_parameter(name_ + ".goal_tolerance", goal_tolerance_);
  node->get_parameter(name_ + ".goal_bias", goal_bias_);
  node->get_parameter(name_ + ".max_iterations", max_iterations_);
  node->get_parameter(name_ + ".seed", seed_);
  node->get_parameter(name_ + ".publish_tree", publish_tree_);

  if (seed_ >= 0) {
    rng_.seed(static_cast<unsigned int>(seed_));
  } else {
    std::random_device rd;
    rng_.seed(rd());
  }

  tree_pub_ = node->create_publisher<visualization_msgs::msg::Marker>(
    name_ + "/tree", rclcpp::QoS(1));

  RCLCPP_INFO(
    logger_, "Configured %s: step=%.2f tol=%.2f bias=%.2f max_it=%d",
    name_.c_str(), step_size_, goal_tolerance_, goal_bias_, max_iterations_);
}

void RRTPlanner::cleanup()
{
  RCLCPP_INFO(logger_, "Cleaning up %s", name_.c_str());
  tree_pub_.reset();
}

void RRTPlanner::activate()
{
  RCLCPP_INFO(logger_, "Activating %s", name_.c_str());
  if (tree_pub_) {
    tree_pub_->on_activate();
  }
}

void RRTPlanner::deactivate()
{
  RCLCPP_INFO(logger_, "Deactivating %s", name_.c_str());
  if (tree_pub_) {
    tree_pub_->on_deactivate();
  }
}

bool RRTPlanner::isFree(double wx, double wy) const
{
  unsigned int mx, my;
  if (!costmap_->worldToMap(wx, wy, mx, my)) {
    return false;  // outside the costmap
  }
  // Inscribed / lethal / unknown (253..255) are treated as collision.
  return costmap_->getCost(mx, my) < nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
}

bool RRTPlanner::segmentFree(double x0, double y0, double x1, double y1) const
{
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const double dist = std::hypot(dx, dy);
  const int n = std::max(1, static_cast<int>(std::ceil(dist / costmap_->getResolution())));
  for (int i = 0; i <= n; ++i) {
    const double t = static_cast<double>(i) / n;
    if (!isFree(x0 + t * dx, y0 + t * dy)) {
      return false;
    }
  }
  return true;
}

nav_msgs::msg::Path RRTPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal,
  std::function<bool()> cancel_checker)
{
  const auto t0 = std::chrono::steady_clock::now();

  nav_msgs::msg::Path path;
  path.header.stamp = clock_->now();
  path.header.frame_id = global_frame_;

  if (start.header.frame_id != global_frame_ || goal.header.frame_id != global_frame_) {
    RCLCPP_ERROR(
      logger_, "Start/goal must be in frame '%s' (got '%s' / '%s')",
      global_frame_.c_str(), start.header.frame_id.c_str(), goal.header.frame_id.c_str());
    throw nav2_core::InvalidPlanner("RRTPlanner: start/goal not in the global frame");
  }

  const double sx = start.pose.position.x, sy = start.pose.position.y;
  const double gx = goal.pose.position.x, gy = goal.pose.position.y;

  if (!isFree(gx, gy)) {
    throw nav2_core::GoalOccupied("RRTPlanner: goal is in collision or outside the costmap");
  }

  // Sampling bounds = whole costmap
  const double min_x = costmap_->getOriginX();
  const double min_y = costmap_->getOriginY();
  const double max_x = min_x + costmap_->getSizeInMetersX();
  const double max_y = min_y + costmap_->getSizeInMetersY();
  std::uniform_real_distribution<double> ux(min_x, max_x), uy(min_y, max_y), u01(0.0, 1.0);

  std::vector<TreeNode> tree;
  tree.reserve(max_iterations_ + 2);
  tree.push_back({sx, sy, -1});

  int goal_idx = -1;
  int it = 0;
  for (; it < max_iterations_; ++it) {
    if (cancel_checker()) {
      throw nav2_core::PlannerCancelled("RRTPlanner: planning cancelled");
    }

    // 1) Sample (with goal bias)
    double rx, ry;
    if (u01(rng_) < goal_bias_) {
      rx = gx; ry = gy;
    } else {
      rx = ux(rng_); ry = uy(rng_);
    }

    // 2) Nearest node (linear scan)
    int nearest = 0;
    double best = std::numeric_limits<double>::max();
    for (size_t i = 0; i < tree.size(); ++i) {
      const double d = std::hypot(tree[i].x - rx, tree[i].y - ry);
      if (d < best) { best = d; nearest = static_cast<int>(i); }
    }
    if (best < 1e-6) {continue;}

    // 3) Steer: move at most step_size_ towards the sample
    const double step = std::min(step_size_, best);
    const double nx = tree[nearest].x + step * (rx - tree[nearest].x) / best;
    const double ny = tree[nearest].y + step * (ry - tree[nearest].y) / best;

    // 4) Collision check of the new edge
    if (!segmentFree(tree[nearest].x, tree[nearest].y, nx, ny)) {continue;}

    tree.push_back({nx, ny, nearest});
    const int new_idx = static_cast<int>(tree.size()) - 1;

    // 5) Goal check: close enough and with a free straight connection
    const double dg = std::hypot(gx - nx, gy - ny);
    if (dg <= goal_tolerance_ ||
      (dg <= step_size_ && segmentFree(nx, ny, gx, gy)))
    {
      tree.push_back({gx, gy, new_idx});
      goal_idx = static_cast<int>(tree.size()) - 1;
      break;
    }
  }

  if (publish_tree_) {publishTree(tree);}

  if (goal_idx < 0) {
    RCLCPP_WARN(logger_, "RRT failed after %d iterations (%zu nodes)", it, tree.size());
    throw nav2_core::NoValidPathCouldBeFound("RRTPlanner: no path found");
  }

  // Backtrack from goal to root
  std::vector<int> idxs;
  for (int i = goal_idx; i != -1; i = tree[i].parent) {idxs.push_back(i);}
  std::reverse(idxs.begin(), idxs.end());

  double length = 0.0;
  for (size_t k = 0; k < idxs.size(); ++k) {
    geometry_msgs::msg::PoseStamped p;
    p.header = path.header;
    p.pose.position.x = tree[idxs[k]].x;
    p.pose.position.y = tree[idxs[k]].y;

    double yaw;
    if (k + 1 < idxs.size()) {
      yaw = std::atan2(tree[idxs[k + 1]].y - tree[idxs[k]].y,
                       tree[idxs[k + 1]].x - tree[idxs[k]].x);
      length += std::hypot(tree[idxs[k + 1]].x - tree[idxs[k]].x,
                           tree[idxs[k + 1]].y - tree[idxs[k]].y);
      p.pose.orientation.z = std::sin(yaw / 2.0);
      p.pose.orientation.w = std::cos(yaw / 2.0);
    } else {
      p.pose.orientation = goal.pose.orientation;  // keep requested final heading
    }
    path.poses.push_back(p);
  }

  const double ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count();
  // Estas linhas servem de base para as métricas do relatório
  RCLCPP_INFO(
    logger_, "RRT ok: iterations=%d nodes=%zu waypoints=%zu length=%.2f m time=%.1f ms",
    it + 1, tree.size(), path.poses.size(), length, ms);

  return path;
}

void RRTPlanner::publishTree(const std::vector<TreeNode> & tree)
{
  if (!tree_pub_ || !tree_pub_->is_activated()) {return;}
  visualization_msgs::msg::Marker m;
  m.header.frame_id = global_frame_;
  m.header.stamp = clock_->now();
  m.ns = "rrt_tree";
  m.id = 0;
  m.type = visualization_msgs::msg::Marker::LINE_LIST;
  m.action = visualization_msgs::msg::Marker::ADD;
  m.scale.x = 0.01;
  m.color.r = 0.1f; m.color.g = 0.6f; m.color.b = 1.0f; m.color.a = 0.8f;
  m.pose.orientation.w = 1.0;
  for (size_t i = 1; i < tree.size(); ++i) {
    geometry_msgs::msg::Point a, b;
    a.x = tree[tree[i].parent].x; a.y = tree[tree[i].parent].y;
    b.x = tree[i].x; b.y = tree[i].y;
    m.points.push_back(a);
    m.points.push_back(b);
  }
  tree_pub_->publish(m);
}

}  // namespace rrt_planner

PLUGINLIB_EXPORT_CLASS(rrt_planner::RRTPlanner, nav2_core::GlobalPlanner)
