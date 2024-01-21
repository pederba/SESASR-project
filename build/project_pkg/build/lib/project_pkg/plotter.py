import os
import numpy as np
import matplotlib.pyplot as plt
import math
import json 

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
initial_pose = localization_node_params['initial_pose']
std_rot1 = localization_node_params["std_rot1"] 
std_transl = localization_node_params["std_transl"]
std_rot2 = localization_node_params["std_rot2"]
std_lin_vel = localization_node_params["std_lin_vel"] 
std_ang_vel = localization_node_params["std_ang_vel"]
std_rng = localization_node_params["std_rng"]
std_brg = localization_node_params["std_brg"]
max_range = localization_node_params["max_range"]
fov_deg = localization_node_params["fov_deg"]



Qt = np.diag([Qt_0, Qt_1]) # measurement noise
Mt = np.diag([Mt_0, Mt_1, Mt_2]) # motion noise


filter = np.loadtxt("filter.csv")
odom = np.loadtxt("odom.csv")
ground_truth = np.loadtxt("ground_truth.csv")

# # Plot odom data
# plt.scatter(odom[:, 0], odom[:, 1], label="odom")
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Odom Data")
# plt.show()

# # Plot ground truth data
# plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label="ground truth")
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Ground Truth Data")
# plt.show()

# # Plot filter data
# plt.scatter(filter[:, 0], filter[:, 1], label="filter")
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Filter Data")
# plt.show()


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

# where to put results
subfolder = "std_lin_vel: " + str(std_lin_vel) + "  std_ang_vel:" + str(std_ang_vel) + "  std_rng:" + str(std_rng) + "  std_brg:" + str(std_brg) + "  max_range:" + str(max_range) + "  fov_deg:" + str(fov_deg) + "/"

directory = "plots/" + subfolder
if not os.path.exists(directory):
    os.makedirs(directory)

# write the values to a .json file
filename = directory + f"results_Qt_{Qt.diagonal()}_Mt_{Mt.diagonal()}.json"
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
        "orientation_total_cumulative_error": orientation_total_cumulative_error,
        "std_rot1": std_rot1, 
        "std_transl": std_transl,
        "std_rot2": std_rot2,
        "std_lin_vel": std_lin_vel, 
        "std_ang_vel": std_ang_vel,
        "std_rng": std_rng,
        "std_brg": std_brg,
        "max_range": max_range,
        "fov_deg": fov_deg

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
plt.scatter(filter[:,0], filter[:,1], label="filter", color="blue")
plt.plot(odom_offset[:,0], odom_offset[:,1], label="odom", color="orange")
plt.plot(ground_truth[:,0], ground_truth[:,1], label="ground truth", color="green")
plt.scatter(landmarks[:,0], landmarks[:,1], color='red', label="landmarks")  # Add this line to plot the landmarks
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Position")
filename = f"position_Qt_{Qt.diagonal()}_Mt_{Mt.diagonal()}.png"
plt.savefig(directory + filename)
plt.close()

plt.plot(np.sqrt((filter[:,0] - ground_truth[:,0])**2 + (filter[:,1] - ground_truth[:,1])**2))
plt.title("Position error")
plt.xlabel("time")
plt.ylabel("error")
filename = f"position_error_Qt_{Qt.diagonal()}_Mt_{Mt.diagonal()}.png"
plt.savefig(directory + filename)
plt.close()

plt.plot(np.abs(filter[:,2] - ground_truth[:,2]))
plt.title("Orientation error")
plt.xlabel("time")
plt.ylabel("error")
filename = f"orientation_error_Qt_{Qt.diagonal()}_Mt_{Mt.diagonal()}.png"
plt.savefig(directory + filename)
