import numpy as np

DELTA_T = 0.1  # time tick [s].
POSE_DIMS = 3  # [x,y,yaw].
LM_DIMS = 2  # [x,y].
N_LANDMARKS = 4
STATE_DIMS = POSE_DIMS + LM_DIMS * N_LANDMARKS

# Landmark indexing helper.
def jj(j):
    return POSE_DIMS + 2 * j
