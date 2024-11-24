import numpy as np

from ekf_slam import LM_DIMS, N_LANDMARKS, jj

SIM_TIME = 40.0  # simulation time [s].
MAX_RANGE = 10.0  # Maximum observation range.

# Simulated measurement noise params. stdev of range and bearing measurements noise.
Q_sim = np.array([0.01, np.deg2rad(1.0)])


# Simulated velocity command noise params. stdev of velocity and angular rate noise.
R_sim = np.array([1.0, np.deg2rad(10.0)])

# Process noise covariance: we handle it in the dumbest way possible, by just adding a
# constant matrix pulled out of our behind. A couple of reasonable ways to do it are
# discussed here:
# https://github.com/cra-ros-pkg/robot_localization/blob/ef0a27352962c56a970f2bbeb8687313b9e54a9a/src/filter_base.cpp#L132.
# If this were a real robot, it might matter, but for the purpose of simulation, we just cheat.
# Note: we assume a Unicycle model, which means there's never a y-component to our velocity
# in the robot frame. We make noise in y much smaller than in x.
R_t = np.diag([0.1, 0.1, 0.01])


def get_vel_cmd(R=R_sim):
    rng = np.random.default_rng()

    v = 1.0  # [m/s]
    omega = 0.1  # [rad/s]

    u = np.array([v, omega])
    u_noisy =  rng.normal([v, omega], scale=R)

    return u, u_noisy

def in_range(x_t, landmarks, max_range=MAX_RANGE):
    """
    Return a list of landmark indices corresponding to landmarks that are
    within range of pose.
    Args:
        x_t : np.array
            2D pose: (x, y). shape == (2,).
        landmarks : np.array
            landmarks.shape == (n, 2), where n is the number of 2D landmarks.
        max_range : float
            Distance threshold.
    Returns: a list of indices from the incoming array of landmarks that are in range.
    """
    assert x_t.shape == (2,)  # 2D pose.
    idx = []
    for j, lm in enumerate(landmarks):
        if np.linalg.norm(lm - x_t) <= max_range:
            idx.append(j)
    return idx


def get_measurements(x_t, landmarks, max_range, Q=Q_sim):
    """
    Return a list of simulated landmark observations.
    Args:
        x_t : array-like
            2D pose: (x, y, theta).
        landmarks :
            Ground-truth landmarks. shape == (n, 2), where n is the number of 2D landmarks.
        max_range :
        Q : array-like
            Noise params for range, bearing.

    Returns:
        j_i: Indices of landmarks in-range.
        z_i: An (optionally noise-corrupted) range-bearing measurement (r, phi)
            of each landmark that is within range, or None if no landmarks are in range.
            phi is in the range [-pi, pi]. Measurement is expressed in the robot frame.
    """
    rng = np.random.default_rng()
    z_i = []
    j_i = in_range(x_t[:2], landmarks, max_range)
    for j in j_i:
        lm = landmarks[j]
        v_sensor_lm = lm - x_t[:2]  # Vector from sensor to landmark.

        # Calculate range, bearing, add sim noise.
        r = np.linalg.norm(v_sensor_lm) + rng.normal(scale=Q[0])
        phi = np.atan2(v_sensor_lm[1], v_sensor_lm[0]) + rng.normal(scale=Q[1])
        theta = x_t[2]  # Account for the rotation of the sensor.
        z_i.append(np.array([r, phi - theta]))
    return j_i, z_i


def validate_landmarks(landmarks):
    # Check assumptions about our fake landmarks.
    assert len(landmarks) == N_LANDMARKS
    for landmark in landmarks:
        # Landmarks are initialized to (0, 0), in the state vector, so
        # no landmarks at the origin.
        assert np.all(np.not_equal(landmark, np.array([0., 0.])))
