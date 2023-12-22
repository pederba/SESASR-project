import numpy as np
import matplotlib.pyplot as plt
import math
import json

from project_pkg.localization_node import localization_node.ekf.Qt, localization_node.ekf.Mt

filter = np.load("filter.npy")
odom = np.load("odom.npy")
ground_truth = np.load("ground_truth.npy")

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
with open('results.json', 'w') as f:
    json.dump({
        "Qt": localization_node.ekf.Qt,
        "Mt": localization_node.ekf.Mt,

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


# plot 
plt.plot(filter[:,0], filter[:,1], label="filter")
plt.plot(odom[:,0], odom[:,1], label="odom")
plt.plot(ground_truth[:,0], ground_truth[:,1], label="ground truth")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Position")
plt.savefig("position.png")
plt.close()

plt.plot(np.sqrt((filter[:,0] - ground_truth[:,0])**2 + (filter[:,1] - ground_truth[:,1])**2))
plt.title("Position error")
plt.xlabel("time")
plt.ylabel("error")
plt.savefig("position_error.png")
plt.close()

plt.plot(np.abs(filter[:,2] - ground_truth[:,2]))
plt.title("Orientation error")
plt.xlabel("time")
plt.ylabel("error")
plt.savefig("orientation_error.png")
