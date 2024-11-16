import numpy as np

DELTA_T = 0.1  # time tick [s].
POSE_DIMS = 3  # [x,y,yaw].
LM_DIMS = 2  # [x,y].
N_LANDMARKS = 4
STATE_DIMS = POSE_DIMS + LM_DIMS * N_LANDMARKS

# Simulated velocity command noise params. stdev of velocity and angular rate noise.
R_sim = np.array([1.0, np.deg2rad(10.0)])

# Process noise covariance: we handle it in the dumbest way possible, by just adding a
# constant matrix pulled out of our behind. A couple of reasonable ways to do it are
# discussed here:
# https://github.com/cra-ros-pkg/robot_localization/blob/ef0a27352962c56a970f2bbeb8687313b9e54a9a/src/filter_base.cpp#L132.
# If this were a real robot, it might matter, but for the purpose of simulation, we just cheat.
R_t = np.diag([0.1, 0.1, 0.01])
