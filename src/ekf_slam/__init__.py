import numpy as np

LANDMARKS = np.array([
    [10., 5.],
    [10., 20.],
    [-5., 16.]
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


def get_landmark_cov(sigma_t, j):
    """
    Given the full covariance matrix, fetch the submatrix that corresponds to
    the covariance of a single landmark.
    Args:
        sigma_t :  np.array, shape (STATE_DIMS, STATE_DIMS)
            Full covariance matrix of our state estimate.
        j : int
            Index of the landmark of interest.
    Returns:
        2x2 covariance matrix of landmark j, or empty if j is out of bounds.
    """
    lm_cov = sigma_t[jj(j): jj(j) + LM_DIMS, jj(j): jj(j) + LM_DIMS]
    return lm_cov


def set_landmark(mu_t, j, lm):
    mu_t[jj(j): jj(j) + LM_DIMS] = lm


def get_landmark_count(mu_t):
    return int((len(mu_t) - POSE_DIMS) / LM_DIMS)
