import numpy as np

dt = 0.1  # time tick [s].
POSE_DIMS = 3  # [x,y,yaw].
LM_DIMS = 2  # [x,y].

# Initial robot pose and landmark ground truth: EKF SLAM can start from uninitialized landmark locations,
# but we start with a fixed number of known locations for simplicity.
INITIAL_POSE = np.zeros((POSE_DIMS,))
LANDMARKS = np.array([
    [10.0, -2.0],
    [15.0, 10.0],
    [3.0, 15.0],
    [-5.0, 20.0]])

# Simulated noise params.
R = np.diag([1.0, np.deg2rad(10.0)]) ** 2
