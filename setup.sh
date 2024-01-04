#!/bin/bash
set -e

vcs import < src/ros2.repos src
sudo apt-get update
rosdep update
rosdep install --from-paths src --ignore-src -y
sudo pip3 install transforms3d
