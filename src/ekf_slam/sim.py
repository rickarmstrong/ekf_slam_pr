import numpy as np

from ekf_slam.ekf import g, range_bearing

SIM_TIME = 60.0  # simulation time [s].
MAX_RANGE = 10.0 # Maximum observation range.

# Simulated control noise params.
M_sim = np.array([1.0, np.deg2rad(0.1)])
M_t = np.diag([M_sim[0] ** 2, M_sim[1] ** 2])

# Simulated measurement noise params. stdev of range and bearing measurements noise.
Q_sim = np.array([0.1, np.deg2rad(0.5)])
Q_t = np.diag([Q_sim[0] ** 2,  Q_sim[1] ** 2])


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


def generate_trajectory(u_t, initial_state, duration, time_step, M=np.diag([0.0, 0.0, 0.0])):
    t = 0.
    mu_t_h = [np.array(initial_state)]
    while duration >= t:
        mu_t_h.append(g(u_t, mu_t_h[-1], delta_t=time_step, M=M))
        t += time_step
    return mu_t_h


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
        z = range_bearing(x_t, landmarks[j])
        z[0] += rng.normal(scale=np.sqrt(Q[0][0]))
        z[1] += rng.normal(scale=np.sqrt(Q[1][1]))
        z_i.append(z)
    return j_i, z_i
