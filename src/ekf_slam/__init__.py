import numpy as np

dt = 0.1  # time tick [s].
POSE_DIMS = 3  # [x,y,yaw].
LM_DIMS = 2  # [x,y].
N_LANDMARKS = 4

# Simulated velocity command noise params. stdev of velocity and angular rate noise.
R = np.array([1.0, np.deg2rad(10.0)])
