import numpy as np
import matplotlib.pyplot as plt
import math

filter = np.load("filter.npy")
odom = np.load("odom.npy")
ground_truth = np.load("ground_truth.npy")

max_allowed_size = min(len(filter), len(odom), len(ground_truth))

filter_indexes = np.arange(0, max_allowed_size, int(len(filter)/max_allowed_size))
filter = filter[filter_indexes]

odom_indexes = np.arange(0, max_allowed_size, int(len(odom)/max_allowed_size))
odom = odom[odom_indexes]

ground_truth_indexes = np.arange(0, max_allowed_size, int(len(ground_truth)/max_allowed_size))
ground_truth = ground_truth[ground_truth_indexes]

position_rmse = math.sqrt(np.mean((filter[:,0] - ground_truth[:,0])**2 + (filter[:,1] - ground_truth[:,1])**2))
position_max_abs_error = np.max(np.abs(filter[:,0] - ground_truth[:,0]) + np.abs(filter[:,1] - ground_truth[:,1]))
position_total_cumulative_error = np.sum(np.abs(filter[:,0] - ground_truth[:,0]) + np.abs(filter[:,1] - ground_truth[:,1]))
position_error_percentage = (np.abs(filter[-1,0] - ground_truth[-1,0]) + np.abs(filter[-1,1] - ground_truth[-1,1])) / position_total_cumulative_error

orientation_rmse = math.sqrt(np.mean((filter[:,2] - ground_truth[:,2])**2))
orientation_max_abs_error = np.max(np.abs(filter[:,2] - ground_truth[:,2]))
orientation_total_cumulative_error = np.sum(np.abs(filter[:,2] - ground_truth[:,2]))

# write the values to a file
with open("errors.txt", "w") as f:
    f.write("Position RMSE: " + str(position_rmse) + "\n")
    f.write("Position max abs error: " + str(position_max_abs_error) + "\n")
    f.write("Position total cumulative error: " + str(position_total_cumulative_error) + "\n")
    f.write("Position error percentage: " + str(position_error_percentage) + "\n")
    f.write("Orientation RMSE: " + str(orientation_rmse) + "\n")
    f.write("Orientation max abs error: " + str(orientation_max_abs_error) + "\n")
    f.write("Orientation total cumulative error: " + str(orientation_total_cumulative_error) + "\n")

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
