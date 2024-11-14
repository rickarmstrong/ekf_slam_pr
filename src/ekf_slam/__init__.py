import numpy as np

DELTA_T = 0.1  # time tick [s].
POSE_DIMS = 3  # [x,y,yaw].
LM_DIMS = 2  # [x,y].
N_LANDMARKS = 4
STATE_DIMS = POSE_DIMS + LM_DIMS * N_LANDMARKS

# Simulated velocity command noise params. stdev of velocity and angular rate noise.
R_sim = np.array([1.0, np.deg2rad(10.0)])
