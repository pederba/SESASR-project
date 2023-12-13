import rclpy
import time
import numpy as np
import tf_transformations as tft

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from project_pkg.ekf import RobotEKF
from project_pkg.motion_models import eval_gux, eval_Gt, eval_Vt
from project_pkg.measurement_model import z_landmark, residual


class LocalizationNode(Node):
    def __init__(self):   
        super().__init__('localization_node')

        self.__ground_truth_sub = self.create_subscription(
            Odometry,
            'ground_truth',
            self.ground_truth_callback,
            10)
        self.__ground_truth_sub

        self.__odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        self.__odom_sub

        self.__ekf_pub = self.create_publisher(
            Odometry,
            'ekf',
            10)
        self.__ekf_pub

        self.ekf_period_s = 0.01
        self.initial_pose = np.array([-0.5, -0.5, 0.0])
        self.initial_covariance = np.array([])
        self.landmarks = np.array([[0,0], [1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]])

        self.std_rot1 = 
        self.std_transl = 
        self.std_rot2 = 


        self.std_lin_vel = 0.1  # [m/s]
        self.std_ang_vel = np.deg2rad(1.0)  # [rad/s]
        self.std_rng = 0.3  # [m]
        self.std_brg = np.deg2rad(1.0)  # [rad]

        self.max_range = 8.0  # [m]
        self.fov_deg = 45  # [deg]

    def ground_truth_callback(self, msg):
        pass

    def odom_callback(self, msg):
        pass


def main(args=None):
    rclpy.init(args=args)

    localization_node = LocalizationNode()

    rclpy.spin(localization_node)

    localization_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()