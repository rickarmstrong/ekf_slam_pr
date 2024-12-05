import numpy as np

from ekf_slam import LM_DIMS, jj

SIM_TIME = 40.0  # simulation time [s].
MAX_RANGE = 15.0  # Maximum observation range.

# Simulated measurement noise params. stdev of range and bearing measurements noise.
Q_sim = np.array([0.1, np.deg2rad(0.1)])
Q_t = np.diag([Q_sim[0] ** 2,  Q_sim[1] ** 2])

# Simulated process noise covariance.
R_sim = np.array([0.4, 0.4, np.deg2rad(0.1)])
R_t = np.diag([R_sim[0] ** 2, R_sim[1] ** 2, R_sim[2] ** 2])


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


def get_measurements(x_t, landmarks, max_range, Q=Q_t):
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
        r = np.linalg.norm(v_sensor_lm) + rng.normal(scale=np.sqrt(Q[0][0]))
        phi = np.atan2(v_sensor_lm[1], v_sensor_lm[0]) + rng.normal(scale=np.sqrt(Q[1][1]))
        theta = x_t[2]  # Account for the rotation of the sensor.
        z_i.append(np.array([r, phi - theta]))
    return j_i, z_i
