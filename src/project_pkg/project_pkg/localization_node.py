import rclpy
import numpy as np
import tf_transformations as tft

from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Twist

from project_pkg.ekf import RobotEKF
from project_pkg.motion_models import eval_gux_odom, eval_Gt_odom, eval_Vt_odom, get_odometry_input, eval_gux_vel, eval_Gt_vel, eval_Vt_vel
from project_pkg.measurement_model import eval_Ht, eval_hx, z_landmark, residual, inverse_eval_hx

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf2_ros


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

        self.__vel_cmd_sub = self.create_subscription(
            Twist,
            '/diff_drive_controller/cmd_vel_unstamped',
            self.vel_command_callback,
            10)
        self.__vel_cmd_sub
        
        self.declare_parameter('motion_model', 'odometry')
        self.declare_parameter('ekf_period_s', 0.005)
        self.declare_parameter('Qt_0', 0.01)
        self.declare_parameter('Qt_1', 0.01)
        self.declare_parameter('Mt_0', 0.01)
        self.declare_parameter('Mt_1', 0.01)
        self.declare_parameter('Mt_2', 0.01)
        self.declare_parameter('initial_pose', [-2.0, -0.5, 0.0])
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

        self.motion_model = self.get_parameter('motion_model').value
        self.ekf_period_s = self.get_parameter('ekf_period_s').value
        Qt_0 = self.get_parameter('Qt_0').value
        Qt_1 = self.get_parameter('Qt_1').value
        Mt_0 = self.get_parameter('Mt_0').value
        Mt_1 = self.get_parameter('Mt_1').value
        Mt_2 = self.get_parameter('Mt_2').value
        self.initial_pose = np.array([self.get_parameter('initial_pose').value]).T
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

        if self.motion_model == 'odometry':
            Qt = np.diag([Qt_0, Qt_1]) # measurement noise
            Mt = np.diag([Mt_0, Mt_1, Mt_2]) # motion noise

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
            self.ekf.Mt = Mt
            self.ekf.Qt = Qt
            
            self.odom_pose = self.initial_pose.copy()
            self.prev_pose = self.initial_pose.copy()
            self.true_pose = self.initial_pose.copy()
            self.z = None
        elif self.motion_model == 'velocity':
            Qt = np.diag([Qt_0, Qt_1]) # measurement noise
            Mt = np.diag([Mt_0, Mt_1]) # motion noise

            self.ekf = RobotEKF(# TODO upate input for velocity
                dim_x=3,
                dim_z=2,
                dim_u=3,
                eval_gux=eval_gux_vel,
                eval_Gt=eval_Gt_vel,
                eval_Vt=eval_Vt_vel,
                eval_hx=eval_hx,
                eval_Ht=eval_Ht
            )
            self.ekf.mu = self.initial_pose
            self.ekf.Sigma = self.initial_covariance
            self.ekf.Mt = Mt
            self.ekf.Qt = Qt
            
            self.odom_pose = self.initial_pose.copy()
            self.prev_pose = self.initial_pose.copy()
            self.true_pose = self.initial_pose.copy()
            self.z = None
            self.vel_cmd = np.array([[0,0]]).T


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


    def ground_truth_callback(self, msg):
        x_pose_true = msg.pose.pose.position.x
        y_pose_true = msg.pose.pose.position.y
        _, _, yaw_true = tft.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.true_pose = np.array([[x_pose_true, y_pose_true, yaw_true]]).T
        # Simulate measuring landmarks
        

    def odom_callback(self, msg):
        x_pose = msg.pose.pose.position.x + self.initial_pose[0][0] # Adjust for offset between the odom frame and the map frame
        y_pose = msg.pose.pose.position.y + self.initial_pose[1][0] #
        _, _, yaw = tft.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.odom_pose = np.array([[x_pose, y_pose, yaw]]).T
    
    def vel_command_callback(self, msg):
        self.vel_cmd = np.array([[msg.linear.x, msg.angular.z]]).T
    

    def ekf_predict(self):
        if self.motion_model == 'odometry':
            u = get_odometry_input(self.odom_pose[:,0], self.prev_pose[:,0]) + np.array([[np.random.normal(0, self.std_rot1), np.random.normal(0, self.std_transl), np.random.normal(0, self.std_rot2)]]).T
            self.ekf.predict_odom(u=u)
        elif self.motion_model == 'velocity':
            self.ekf.predict_vel(u=self.vel_cmd, g_extra_args=(self.ekf_period_s,))
        else:
            Exception("Unknown motion model")

        self.prev_pose = self.odom_pose.copy()# Update
        for lmark in self.landmarks:
            z = z_landmark(self.true_pose, lmark, self.std_rng, self.std_brg, self.max_range, self.fov_deg)
            if z is not None:
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