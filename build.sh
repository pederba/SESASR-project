#!/bin/bash
set -e

export MAKEFLAGS="-j 6" # 6 is the number of cores the build system is allowed to use

# Set the default build type
BUILD_TYPE=RelWithDebInfo
colcon build \
--symlink-install \
--parallel-workers 3 \ 
--cmake-args "-DCMAKE_BUILD_TYPE=$BUILD_TYPE" "-DCMAKE_EXPORT_COMPILE_COMMANDS=On" \
-Wall -Wextra -Wpedantic
