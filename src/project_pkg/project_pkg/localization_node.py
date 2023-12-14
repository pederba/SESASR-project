import rclpy
import numpy as np
import tf_transformations as tft

from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

from project_pkg.ekf import RobotEKF
from project_pkg.motion_models import eval_gux_odom, eval_Gt_odom, eval_Vt_odom, eval_hx, eval_Ht, get_odometry_input
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

        self.ekf_period_s = 0.1
        self.timer = self.create_timer(
            self.ekf_period_s,
            self.ekf_step
        )

        self.initial_pose = np.array([[-0.5, -0.5, 0.0]]).T
        self.initial_covariance = np.diag([0.1, 0.1, 0.1])
        self.landmarks = np.array([[0,0], [1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]])

        self.std_rot1 = np.deg2rad(1.0)
        self.std_transl = 0.3
        self.std_rot2 = np.deg2rad(1.0)

        # noise parameters
        self.std_lin_vel = 0.1  # [m/s]
        self.std_ang_vel = np.deg2rad(1.0)  # [rad/s]
        self.std_rng = 0.3  # [m]
        self.std_brg = np.deg2rad(1.0)  # [rad]

        # parameters for measurement model (z_landmarks())
        self.max_range = 8  # [m]
        self.fov_deg =  np.deg2rad(45) # [rad]

        self.ekf = RobotEKF(
            dim_x=3,
            dim_z=2,
            dim_u=3,
            eval_gux=eval_gux_odom,
            eval_Gt=eval_Gt_odom,
            eval_Vt=eval_Vt_odom,
            eval_hx=eval_hx,
            eval_Ht=eval_Ht
        )
        self.ekf.mu = self.initial_pose
        self.ekf.Sigma = self.initial_covariance
        self.ekf.Mt = np.diag([self.std_rot1**2, self.std_transl**2, self.std_rot2**2])
        self.ekf.Qt = np.diag([self.std_rng**2, self.std_brg**2])

        self.odom_pos = self.initial_pose
        self.prev_pose = self.initial_pose
        self.true_pose = self.initial_pose

    def ground_truth_callback(self, msg):
        x_pose_true = msg.pose.pose.position.x
        y_pose_true = msg.pose.pose.position.y
        _, _, yaw_true = tft.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.true_pose = np.array([[x_pose_true, y_pose_true, yaw_true]]).T

    def odom_callback(self, msg):
        x_pose = msg.pose.pose.position.x
        y_pose = msg.pose.pose.position.y
        _, _, yaw = tft.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.odom_pos = np.array([[x_pose, y_pose, yaw]]).T
        
    def ekf_step(self):
        u = get_odometry_input(self.odom_pos[:,0], self.prev_pose[:,0])
        self.ekf.predict(u=u)
        self.prev_pose = self.odom_pos

        # Simulate measuring landmarks
        for lmark in self.landmarks:
            z = z_landmark(self.true_pose, lmark, self.std_rng, self.std_brg, self.max_range, self.fov_deg)
            if z is not None:
                self.ekf.update(z[:,0], lmark, residual=residual)

        self.get_logger().info('Pose: ' + str(self.ekf.mu[:,0]))

        msg = Odometry()
        msg.pose.pose.position.x = self.ekf.mu[0,0]
        msg.pose.pose.position.y = self.ekf.mu[1,0]
        quaternion = Quaternion()
        quaternion.x, quaternion.y, quaternion.z, quaternion.w = tft.quaternion_from_euler(0, 0, self.ekf.mu[2,0])
        msg.pose.pose.orientation = quaternion
        # msg.pose.covariance = self.ekf.Sigma.flatten()
        self.__ekf_pub.publish(msg)


        


def main(args=None):
    rclpy.init(args=args)

    localization_node = LocalizationNode()

    rclpy.spin(localization_node)

    localization_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()