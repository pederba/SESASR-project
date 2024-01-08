import rclpy
import numpy as np
import tf_transformations as tft

from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

from project_pkg.ekf import RobotEKF
from project_pkg.motion_models import eval_gux_odom, eval_Gt_odom, eval_Vt_odom, get_odometry_input
from project_pkg.measurement_model import eval_Ht, eval_hx, z_landmark, residual

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

#Qt = np.diag([10, 10]) # measurement noise
#Mt = np.diag([1, 1, 1]) # motion noise

initial_pose = np.array([[-0.5, -0.5, 0.0]]).T


class LocalizationNode(Node):
    def __init__(self):   
        super().__init__('localization_node')

        self.__ground_truth_sub = self.create_subscription(
            Odometry,
            '/ground_truth_odom_frame',
            self.ground_truth_callback,
            10)
        self.__ground_truth_sub

        self.__odom_sub = self.create_subscription(
            Odometry,
            '/diff_drive_controller/odom',
            self.odom_callback,
            10)
        self.__odom_sub

        self.__ekf_pub = self.create_publisher(
            Odometry,
            '/ekf',
            10)
        self.__ekf_pub

        self.__landmark_debug_pub = self.create_publisher(
            Odometry,
            '/landmark_debug',
            10)

        self.__z_debug_pub = self.create_publisher(
            Odometry,
            '/z_debug',
            10)
        
        self.__z_hat_debug_pub = self.create_publisher(
            Odometry,
            '/z_hat_debug',
            10)
        
        self.__yaw_debug_pub = self.create_publisher(
            Odometry,
            '/yaw_debug',
            10)
        
        self.__res_debug_pub = self.create_publisher(
            Odometry,
            '/res_debug',
            10)
        
        self.declare_parameter('ekf_period_s', 0.005)
        self.declare_parameter('Qt_0', 0.01)
        self.declare_parameter('Qt_1', 0.01)
        self.declare_parameter('Mt_0', 0.01)
        self.declare_parameter('Mt_1', 0.01)
        self.declare_parameter('Mt_2', 0.01)
        self.declare_parameter('initial_pose', [-0.5, -0.5, 0.0])
        self.declare_parameter('initial_covariance', [0.1, 0.1, 0.1])
        self.declare_parameter('landmark1', [0,0])
        self.declare_parameter('landmark2', [1,0])
        self.declare_parameter('landmark3', [1,1])
        self.declare_parameter('landmark4', [0,1])
        self.declare_parameter('landmark5', [-1,1])
        self.declare_parameter('landmark6', [-1,0])
        self.declare_parameter('landmark7', [-1,-1])
        self.declare_parameter('landmark8', [0,-1])
        self.declare_parameter('landmark9', [1,-1])
        self.declare_parameter('std_rot1', np.deg2rad(1.0))
        self.declare_parameter('std_transl', 0.1)
        self.declare_parameter('std_rot2', np.deg2rad(1.0))
        self.declare_parameter('std_lin_vel', 0.1)
        self.declare_parameter('std_ang_vel', np.deg2rad(1.0))
        self.declare_parameter('std_rng', 0.3)
        self.declare_parameter('std_brg', np.deg2rad(1))
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('fov_deg', np.deg2rad(45))


        self.ekf_period_s = self.get_parameter('ekf_period_s').value
        Qt_0 = self.get_parameter('Qt_0').value
        Qt_1 = self.get_parameter('Qt_1').value
        Mt_0 = self.get_parameter('Mt_0').value
        Mt_1 = self.get_parameter('Mt_1').value
        Mt_2 = self.get_parameter('Mt_2').value
        self.initial_pose = self.get_parameter('initial_pose').value
        self.initial_covariance = self.get_parameter('initial_covariance').value
        # Get landmarks
        self.landmarks = np.array([
            self.get_parameter('landmark1').value,
            self.get_parameter('landmark2').value,
            self.get_parameter('landmark3').value,
            self.get_parameter('landmark4').value,
            self.get_parameter('landmark5').value,
            self.get_parameter('landmark6').value,
            self.get_parameter('landmark7').value,
            self.get_parameter('landmark8').value,
            self.get_parameter('landmark9').value
        ])
        # Standard deviation for odometry input
        self.std_rot1 = self.get_parameter('std_rot1').value
        self.std_transl = self.get_parameter('std_transl').value
        self.std_rot2 = self.get_parameter('std_rot2').value
        # Standard deviation for velocity
        self.std_lin_vel = self.get_parameter('std_lin_vel').value
        self.std_ang_vel = self.get_parameter('std_ang_vel').value
        # Standard deviation of range measurement
        self.std_rng = self.get_parameter('std_rng').value
        self.std_brg = self.get_parameter('std_brg').value
        # Maximum range of range sensor and field of view
        self.max_range = self.get_parameter('max_range').value
        self.fov_deg = self.get_parameter('fov_deg').value

        self.timer = self.create_timer(
            self.ekf_period_s,
            self.ekf_predict
        )
        Qt = np.diag([Qt_0, Qt_1]) # measurement noise
        Mt = np.diag([Mt_0, Mt_1, Mt_2]) # motion noise

        # self.initial_covariance = np.diag([0.1, 0.1, 0.1])
        # self.landmarks = np.array([[0,0], [1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]])

        # self.std_rot1 = np.deg2rad(1.0)
        # self.std_transl = 0.3
        # self.std_rot2 = np.deg2rad(1.0)

        # # noise parameters for measurement model (z_landmarks())
        # self.std_lin_vel = 0#0.1  # [m/s]
        # self.std_ang_vel = 0#np.deg2rad(1.0)  # [rad/s]
        # self.std_rng = 0#0.3  # [m]
        # self.std_brg = 0#np.deg2rad(1)  # [rad]

        # # parameters for measurement model (z_landmarks())
        # self.max_range = 8  # [m]
        # self.fov_deg =  np.deg2rad(45) # [rad]

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
        self.ekf.mu = initial_pose
        self.ekf.Sigma = self.initial_covariance
        self.ekf.Mt = Mt
        self.ekf.Qt = Qt

        self.odom_pose = initial_pose.copy()
        self.prev_pose = initial_pose.copy()
        self.true_pose = initial_pose.copy()


    def publish_ekf(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_footprint'
        msg.pose.pose.position.x = self.ekf.mu[0,0]
        msg.pose.pose.position.y = self.ekf.mu[1,0]
        quaternion = Quaternion()
        quaternion.x, quaternion.y, quaternion.z, quaternion.w = tft.quaternion_from_euler(0, 0, self.ekf.mu[2,0])
        msg.pose.pose.orientation = quaternion
        #msg.pose.covariance = self.ekf.Sigma.flatten()
        self.__ekf_pub.publish(msg)

    def publish_debug(self, z, z_hat, true_pose):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_footprint'
        msg.pose.pose.position.x = z[0][0]
        msg.pose.pose.position.y = z[1][0]
        quaternion = Quaternion()
        quaternion.x, quaternion.y, quaternion.z, quaternion.w = tft.quaternion_from_euler(0, 0, 0)
        msg.pose.pose.orientation = quaternion
        #msg.pose.covariance = self.ekf.Sigma.flatten()
        self.__landmark_debug_pub.publish(msg)

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_footprint'
        msg.pose.pose.position.x = z_hat[0][0]
        msg.pose.pose.position.y = z_hat[1][0]
        quaternion = Quaternion()
        quaternion.x, quaternion.y, quaternion.z, quaternion.w = tft.quaternion_from_euler(0, 0, 0)
        msg.pose.pose.orientation = quaternion
        self.__z_hat_debug_pub.publish(msg)

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_footprint'
        msg.pose.pose.position.x = float(true_pose[0])
        msg.pose.pose.position.y = float(true_pose[1])
        msg.pose.pose.position.z = float(true_pose[2])
        quaternion = Quaternion()
        quaternion.x, quaternion.y, quaternion.z, quaternion.w = tft.quaternion_from_euler(0, 0, 0)
        msg.pose.pose.orientation = quaternion
        self.__yaw_debug_pub.publish(msg)

        res = z[:, 0] - z_hat[:, 0]
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_footprint'
        msg.pose.pose.position.x = res[0]
        msg.pose.pose.position.y = res[1]
        quaternion = Quaternion()
        quaternion.x, quaternion.y, quaternion.z, quaternion.w = tft.quaternion_from_euler(0, 0, 0)
        msg.pose.pose.orientation = quaternion
        self.__res_debug_pub.publish(msg)

    def ground_truth_callback(self, msg):
        x_pose_true = msg.pose.pose.position.x
        y_pose_true = msg.pose.pose.position.y
        _, _, yaw_true = tft.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.true_pose = np.array([[x_pose_true, y_pose_true, yaw_true]]).T

    def odom_callback(self, msg):
        #TODO add noise to odom
        x_pose = msg.pose.pose.position.x + initial_pose[0,0] # Adjust for offset between the odom frame and the map frame
        y_pose = msg.pose.pose.position.y + initial_pose[1,0] #
        _, _, yaw = tft.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        
        self.odom_pose = np.array([[x_pose, y_pose, yaw]]).T

    def ekf_predict(self):
        # Predict
        # Simulate odometry
        
        u = get_odometry_input(self.odom_pose[:,0], self.prev_pose[:,0]) + np.array([[np.random.normal(0, self.std_rot1), np.random.normal(0, self.std_transl), np.random.normal(0, self.std_rot2)]]).T#TODO add noise here?
        self.ekf.predict(u=u)
        self.prev_pose = self.odom_pose.copy()

        # Update
        # Simulate measuring landmarks
        for lmark in self.landmarks:
            z = z_landmark(self.true_pose, lmark, self.std_rng, self.std_brg, self.max_range, self.fov_deg)
            if z is not None:
                # deleta v
                z_hat = eval_hx(*self.ekf.mu[:,0], *lmark)                
                self.publish_debug(z[:,0], z_hat, self.true_pose)
                #delete^
                self.ekf.update(z[:,0], lmark, residual=residual)

        self.publish_ekf()




def main(args=None):
    rclpy.init(args=args)

    localization_node = LocalizationNode()

    rclpy.spin(localization_node)

    localization_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()