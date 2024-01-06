import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Odometry


class TransformNode(Node):
    def __init__(self):
        super().__init__('transform_node')
        self.broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.broadcast_timer_callback)
        self.ground_truth_sub = self.create_subscription(
            Odometry,
            '/ground_truth',
            self.ground_truth_callback,
            10)
        self.ground_truth_odom_frame_pub = self.create_publisher(
            Odometry,
            '/ground_truth_odom_frame',
            10)

    def ground_truth_callback(self, msg):
        new_msg = Odometry()
        new_msg.header = msg.header
        new_msg.header.frame_id = 'odom'
        new_msg.child_frame_id = 'base_footprint'
        new_msg.pose.pose = msg.pose.pose
        new_msg.twist.twist = msg.twist.twist
        self.ground_truth_odom_frame_pub.publish(new_msg)

    def broadcast_timer_callback(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'odom'
        t.transform.translation.x = 0.0  # example translation
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0  # example rotation
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 0.0
        self.broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = TransformNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
