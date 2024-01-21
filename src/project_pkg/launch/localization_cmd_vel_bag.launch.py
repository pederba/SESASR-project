import os
import launch
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import Shutdown, TimerAction




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

        Node(
            package='project_pkg',
            executable='transform_node',
            name='transform_node',
            parameters=[config]
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('turtlebot3_bringup'), 
                    'launch', 
                    'gazebo.launch.py'
                ])
            )
        ),

        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', '/workspaces/SESASR-project/bag_with_commands_turning3'],
            output='screen'
        ),
    ])