import numpy as np

from ekf_slam import LM_DIMS, N_LANDMARKS

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
