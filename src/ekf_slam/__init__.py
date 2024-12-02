import numpy as np


LANDMARKS = np.array([
    [10.0, -2.0],
    [15.0, 2.0],
])

DELTA_T = 0.1  # time tick [s].
POSE_DIMS = 3  # [x,y,yaw].
LM_DIMS = 2  # [x,y].
STATE_DIMS = POSE_DIMS + LM_DIMS * len(LANDMARKS)


# Landmark indexing helpers.
def jj(j):
    """Return the index of the first element of the jth landmark
     in a full state vector."""
    return POSE_DIMS + 2 * j


def get_landmark(mu_t, j):
    return mu_t[jj(j): jj(j) + LM_DIMS]


def set_landmark(mu_t, j, lm):
    mu_t[jj(j): jj(j) + LM_DIMS] = lm


def new_cov_matrix():
    C = np.zeros((STATE_DIMS, STATE_DIMS))

    # Set landmark covariances to inf (unknown).
    lm_covs = np.diag(np.full(len(LANDMARKS) * LM_DIMS, np.finfo(np.float64).max))
    C[3:, 3:] = lm_covs
    return C
