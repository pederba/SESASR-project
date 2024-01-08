import numpy as np
import matplotlib.pyplot as plt
import math
import json

from project_pkg.localization_node import initial_pose #, Mt, Qt, 

import yaml

# Path to your YAML file
file_path = '/workspaces/SESASR-project/src/project_pkg/config/localization_params.yaml'

# Reading the YAML file
with open(file_path, 'r') as file:
    data = yaml.safe_load(file)

# Accessing parameters
localization_node_params = data['localization_node']['ros__parameters']
ekf_period_s = localization_node_params['ekf_period_s']
Qt_0 = localization_node_params['Qt_0']
Qt_1 = localization_node_params['Qt_1']
Mt_0 = localization_node_params['Mt_0']
Mt_1 = localization_node_params['Mt_1']
Mt_2 = localization_node_params['Mt_2']

Qt = np.diag([Qt_0, Qt_1]) # measurement noise
Mt = np.diag([Mt_0, Mt_1, Mt_2]) # motion noise


filter = np.load("filter.npy")
odom = np.load("odom.npy")
ground_truth = np.load("ground_truth.npy")

# Plot odom data
plt.scatter(odom[:, 0], odom[:, 1], label="odom")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Odom Data")
plt.show()

# Plot ground truth data
plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label="ground truth")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ground Truth Data")
plt.show()

# Plot filter data
plt.scatter(filter[:, 0], filter[:, 1], label="filter")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Filter Data")
plt.show()


# adjust data size to fit each other
max_allowed_length = min(len(filter), len(odom), len(ground_truth))

filter_indices = np.linspace(0, len(filter)-1, max_allowed_length, dtype=int)
filter = filter[filter_indices]

odom_indices = np.linspace(0, len(odom)-1, max_allowed_length, dtype=int)
odom = odom[odom_indices]

ground_truth_indices = np.linspace(0, len(ground_truth)-1, max_allowed_length, dtype=int)
ground_truth = ground_truth[ground_truth_indices]

# compute errors
position_rmse = math.sqrt(np.mean((filter[:,0] - ground_truth[:,0])**2 + (filter[:,1] - ground_truth[:,1])**2))
position_max_abs_error = np.max(np.abs(filter[:,0] - ground_truth[:,0]) + np.abs(filter[:,1] - ground_truth[:,1]))
position_total_cumulative_error = np.sum(np.abs(filter[:,0] - ground_truth[:,0]) + np.abs(filter[:,1] - ground_truth[:,1]))
position_error_percentage = (np.abs(filter[-1,0] - ground_truth[-1,0]) + np.abs(filter[-1,1] - ground_truth[-1,1])) / position_total_cumulative_error

orientation_rmse = math.sqrt(np.mean((filter[:,2] - ground_truth[:,2])**2))
orientation_max_abs_error = np.max(np.abs(filter[:,2] - ground_truth[:,2]))
orientation_total_cumulative_error = np.sum(np.abs(filter[:,2] - ground_truth[:,2]))

# write the values to a .json file
filename = f"results_Qt_{Qt.diagonal()}_Mt_{Mt.diagonal()}.json"
with open(filename, 'a') as f:
    json.dump({
        "Qt diagonal": Qt.diagonal().tolist(),
        "Mt diagonal": Mt.diagonal().tolist(),
        
        "position_rmse": position_rmse,
        "position_max_abs_error": position_max_abs_error,
        "position_total_cumulative_error": position_total_cumulative_error,
        "position_error_percentage": position_error_percentage,
        "orientation_rmse": orientation_rmse,
        "orientation_max_abs_error": orientation_max_abs_error,
        "orientation_total_cumulative_error": orientation_total_cumulative_error
    }, 
    f,
    indent=4
    )

# landmarks
landmarks = np.array([[0,0], [1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]])


odom_offset = np.copy(odom)
odom_offset[:, 0] -= 2
odom_offset[:, 1] -= 0.5


# plot 
plt.scatter(filter[:,0], filter[:,1], label="filter")
plt.scatter(odom_offset[:,0], odom_offset[:,1], label="odom")
plt.scatter(ground_truth[:,0], ground_truth[:,1], label="ground truth")
plt.scatter(landmarks[:,0], landmarks[:,1], color='red', label="landmarks")  # Add this line to plot the landmarks
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Position")
filename = f"position_Qt_{Qt.diagonal()}_Mt_{Mt.diagonal()}.png"
plt.savefig("plots/" + filename)
plt.close()

plt.plot(np.sqrt((filter[:,0] - ground_truth[:,0])**2 + (filter[:,1] - ground_truth[:,1])**2))
plt.title("Position error")
plt.xlabel("time")
plt.ylabel("error")
filename = f"position_error_Qt_{Qt.diagonal()}_Mt_{Mt.diagonal()}.png"
plt.savefig("plots/" + filename)
plt.close()

plt.plot(np.abs(filter[:,2] - ground_truth[:,2]))
plt.title("Orientation error")
plt.xlabel("time")
plt.ylabel("error")
filename = f"orientation_error_Qt_{Qt.diagonal()}_Mt_{Mt.diagonal()}.png"
plt.savefig("plots/" + filename)