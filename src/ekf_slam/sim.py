import numpy as np

from ekf_slam import LM_DIMS, N_LANDMARKS

SIM_TIME = 40.0  # simulation time [s].
MAX_RANGE = 10.0  # Maximum observation range.


# Simulated velocity command noise params. stdev of velocity and angular rate noise.
R_sim = np.array([1.0, np.deg2rad(10.0)])

# Process noise covariance: we handle it in the dumbest way possible, by just adding a
# constant matrix pulled out of our behind. A couple of reasonable ways to do it are
# discussed here:
# https://github.com/cra-ros-pkg/robot_localization/blob/ef0a27352962c56a970f2bbeb8687313b9e54a9a/src/filter_base.cpp#L132.
# If this were a real robot, it might matter, but for the purpose of simulation, we just cheat.
R_t = np.diag([0.1, 0.1, 0.01])


def get_vel_cmd(R=np.array([0., 0.])):
    rng = np.random.default_rng()

    v = 1.0  # [m/s]
    omega = 0.1  # [rad/s]

    u = np.array([v, omega])
    u_noisy =  rng.normal([v, omega], scale=R_sim)

    return u, u_noisy

def in_range(pose_xy, landmarks, max_range=MAX_RANGE):
    """
    Return a list of landmark indices corresponding to landmarks that are
    within range of pose.
    Args:
        x : array_like
            2D pose, where pose[0] is x, and pose[1] is y.
        landmarks : np.array
            landmarks.shape == (n, 2), where n is the number of 2D landmarks.
        max_range : float
            Distance threshold.
    Returns: a list of indices from the incoming array of landmarks that are in range.
    """
    idx = []
    for j, lm in enumerate(landmarks):
        if np.linalg.norm(lm - pose_xy) <= max_range:
            idx.append(j)
    return idx


def measure(u_t, landmarks, max_range):
    z_t = np.zeros(LM_DIMS * N_LANDMARKS)
    return z_t


def validate_landmarks(landmarks):
    # Check assumptions about our fake landmarks.
    assert len(landmarks) == N_LANDMARKS
    for landmark in landmarks:
        # Landmarks are initialized to (0, 0), in the state vector, so
        # no landmarks at the origin.
        assert np.all(np.not_equal(landmark, np.array([0., 0.])))
