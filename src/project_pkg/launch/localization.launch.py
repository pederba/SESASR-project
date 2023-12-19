import os
import launch
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(get_package_share_directory('project_pkg'), 'config', 'localization_params.yaml')
    return launch.LaunchDescription([

        Node(
            package='project_pkg',
            executable='localization_node',
            name='localization_node',
            parameters=[config]
        ),

        Node(
            package='project_pkg',
            executable='recorder',
            name='recorder',
            output='screen'
        ),

        ExecuteProcess(
            cmd=['xterm', '-e', 'ros2 run turtlebot3_teleop teleop_keyboard'], # remember to export TURTLEBOT3_MODEL=burger
            output='screen'
        )
    ])