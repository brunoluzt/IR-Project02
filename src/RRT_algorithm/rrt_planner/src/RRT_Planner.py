import math
import random
import time

import rclpy
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker


class RRTPlannerNode(Node):
    def __init__(self):
        super().__init__('rrt_planner_node')
        self.declare_parameter('step_size', 0.3)
        self.declare_parameter('goal_tolerance', 0.2)
        self.declare_parameter('goal_bias', 0.05)
        self.declare_parameter('max_iterations', 5000)
        self.declare_parameter('seed', -1)
        self.declare_parameter('occupied_threshold', 99)  # >=99 ou -1: colisão
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        seed = self.get_parameter('seed').value
        if seed >= 0:
            random.seed(seed)

        self.grid = None
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap',
                                 self.costmap_cb, qos)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.tree_pub = self.create_publisher(Marker, 'rrt/tree', 1)
        self.path_pub = self.create_publisher(Path, 'rrt/plan', 1)

        self.server = ActionServer(
            self, ComputePathToPose, 'rrt/compute_path_to_pose', self.execute)
        self.get_logger().info('RRT planner (Python) pronto.')

    # ---------- costmap ----------
    def costmap_cb(self, msg):
        self.grid = msg

    def is_free(self, x, y):
        g = self.grid
        res = g.info.resolution
        ix = int((x - g.info.origin.position.x) / res)
        iy = int((y - g.info.origin.position.y) / res)
        if ix < 0 or iy < 0 or ix >= g.info.width or iy >= g.info.height:
            return False
        v = g.data[iy * g.info.width + ix]
        return 0 <= v < self.get_parameter('occupied_threshold').value

    def segment_free(self, x0, y0, x1, y1):
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(d / self.grid.info.resolution)))
        for i in range(n + 1):
            t = i / n
            if not self.is_free(x0 + t * (x1 - x0), y0 + t * (y1 - y0)):
                return False
        return True

    # ---------- RRT ----------
    def rrt(self, start, goal):
        step = self.get_parameter('step_size').value
        tol = self.get_parameter('goal_tolerance').value
        bias = self.get_parameter('goal_bias').value
        max_it = self.get_parameter('max_iterations').value
        info = self.grid.info
        min_x, min_y = info.origin.position.x, info.origin.position.y
        max_x = min_x + info.width * info.resolution
        max_y = min_y + info.height * info.resolution

        nodes = [start]      # (x, y)
        parent = [-1]
        for it in range(max_it):
            if random.random() < bias:
                rx, ry = goal
            else:
                rx, ry = random.uniform(min_x, max_x), random.uniform(min_y, max_y)

            near = min(range(len(nodes)),
                       key=lambda i: math.hypot(nodes[i][0] - rx, nodes[i][1] - ry))
            nx0, ny0 = nodes[near]
            d = math.hypot(rx - nx0, ry - ny0)
            if d < 1e-6:
                continue
            s = min(step, d)
            nx, ny = nx0 + s * (rx - nx0) / d, ny0 + s * (ry - ny0) / d
            if not self.segment_free(nx0, ny0, nx, ny):
                continue

            nodes.append((nx, ny))
            parent.append(near)
            dg = math.hypot(goal[0] - nx, goal[1] - ny)
            if dg <= tol or (dg <= step and self.segment_free(nx, ny, goal[0], goal[1])):
                nodes.append(goal)
                parent.append(len(nodes) - 2)
                idxs, i = [], len(nodes) - 1
                while i != -1:
                    idxs.append(i)
                    i = parent[i]
                return [nodes[i] for i in reversed(idxs)], nodes, parent, it + 1
        return None, nodes, parent, max_it

    # ---------- action ----------
    def get_start(self, request):
        frame = self.get_parameter('global_frame').value
        if request.use_start:
            return request.start.pose.position.x, request.start.pose.position.y
        t = self.tf_buffer.lookup_transform(
            frame, self.get_parameter('base_frame').value, rclpy.time.Time())
        return t.transform.translation.x, t.transform.translation.y

    def execute(self, goal_handle):
        t0 = time.time()
        result = ComputePathToPose.Result()
        req = goal_handle.request
        frame = self.get_parameter('global_frame').value

        if self.grid is None:
            self.get_logger().error('Sem costmap em /global_costmap/costmap')
            goal_handle.abort()
            return result
        try:
            start = self.get_start(req)
        except Exception as e:  # erro de TF
            self.get_logger().error(f'TF: {e}')
            goal_handle.abort()
            return result

        goal = (req.goal.pose.position.x, req.goal.pose.position.y)
        if not self.is_free(*goal):
            self.get_logger().error('Goal em colisão ou fora do mapa')
            goal_handle.abort()
            return result

        pts, nodes, parent, iters = self.rrt(start, goal)
        self.publish_tree(nodes, parent, frame)
        if pts is None:
            self.get_logger().warn(f'RRT falhou ({iters} iterações)')
            goal_handle.abort()
            return result

        path = Path()
        path.header.frame_id = frame
        path.header.stamp = self.get_clock().now().to_msg()
        length = 0.0
        for k, (x, y) in enumerate(pts):
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x, p.pose.position.y = x, y
            if k + 1 < len(pts):
                yaw = math.atan2(pts[k + 1][1] - y, pts[k + 1][0] - x)
                length += math.hypot(pts[k + 1][0] - x, pts[k + 1][1] - y)
                p.pose.orientation.z = math.sin(yaw / 2)
                p.pose.orientation.w = math.cos(yaw / 2)
            else:
                p.pose.orientation = req.goal.pose.orientation
            path.poses.append(p)

        dt = time.time() - t0
        self.path_pub.publish(path)
        self.get_logger().info(
            f'RRT ok: iterations={iters} nodes={len(nodes)} '
            f'waypoints={len(pts)} length={length:.2f} m time={dt*1000:.1f} ms')
        result.path = path
        result.planning_time.sec = int(dt)
        result.planning_time.nanosec = int((dt % 1) * 1e9)
        goal_handle.succeed()
        return result

    def publish_tree(self, nodes, parent, frame):
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id = 'rrt_tree', 0
        m.type, m.action = Marker.LINE_LIST, Marker.ADD
        m.scale.x = 0.01
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.6, 1.0, 0.8
        m.pose.orientation.w = 1.0
        for i in range(1, len(nodes)):
            a, b = nodes[parent[i]], nodes[i]
            m.points.append(Point(x=a[0], y=a[1]))
            m.points.append(Point(x=b[0], y=b[1]))
        self.tree_pub.publish(m)


def main():
    rclpy.init()
    node = RRTPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()