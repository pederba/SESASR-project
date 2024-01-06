import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import  tf_transformations

class RecorderNode(Node):
   def __init__(self):
      super().__init__("my_node")
      self.get_logger().info("Initializing my_node")

      self.__save_data_timer = self.create_timer(1.0, self.save_data)

      self.__ground_truth_sub = self.create_subscription(
         Odometry,
         '/ground_truth',
         self.ground_truth_callback,
         10)
      self.__ground_truth_sub

      self.__odom_sub = self.create_subscription(
         Odometry,
         '/diff_drive_controller/odom',
         self.odom_callback,
         10)
      self.__odom_sub
       
      self.__ekf_sub = self.create_subscription(
         Odometry,
         '/ekf',
         self.ekf_callback,
         10)
      self.__ekf_sub

      self.odom = []
      self.filter = []
      self.ground_truth = []

   def odom_callback(self, odom):
      _, _, yaw = tf_transformations.euler_from_quaternion([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])
      self.odom.append([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
      
   
   def ekf_callback(self, filter):
      _, _, yaw = tf_transformations.euler_from_quaternion([filter.pose.pose.orientation.x, filter.pose.pose.orientation.y, filter.pose.pose.orientation.z, filter.pose.pose.orientation.w])
      self.filter.append([filter.pose.pose.position.x, filter.pose.pose.position.y, yaw])
      
   
   def ground_truth_callback(self, ground_truth):
      _, _, yaw = tf_transformations.euler_from_quaternion([ground_truth.pose.pose.position.x, ground_truth.pose.pose.position.y, ground_truth.pose.pose.orientation.z, ground_truth.pose.pose.orientation.w])
      self.ground_truth.append([ground_truth.pose.pose.position.x, ground_truth.pose.pose.position.y, yaw])

   def save_data(self):
      np.save("odom.npy", np.array(self.odom))
      np.save("filter.npy", np.array(self.filter))
      np.save("ground_truth.npy", np.array(self.ground_truth))
    

def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass    
    finally:
        node.save_data()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()