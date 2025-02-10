import numpy as np

from ekf_slam.ekf import g

SIM_TIME = 60.0  # simulation time [s].
MAX_RANGE = 10.0 # Maximum observation range.

# Simulated control noise params (linear and angular velocity).
M_sim = np.array([1.0, np.deg2rad(0.1)])
M_t = np.diag([M_sim[0] ** 2, M_sim[1] ** 2])

# Simulated measurement noise params. stdev of cartesian (x, y) measurement noise.
Q_sim = np.array([0.1, 0.1])
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

    Assume our simulated sensor has a 360-degree view and range limited to max_range.
    Args:
        x_t : array-like
            3D pose: (x, y, theta).
        landmarks :
            Ground-truth landmarks. shape == (n, 2), where n is the number of 2D landmarks.
        max_range :
        Q : array-like
            Noise params for x, y.

    Returns:
        j_i: Indices of landmarks in-range.
        z_i: An (optionally noise-corrupted) cartesian measurement (x, y)
            of each landmark that is within range, or None if no landmarks are in range.
            Measurement is expressed in the robot (sensor) frame.
    """
    # First, calculate the homogeneous map->sensor transform:
    # The sensor->map frame transformation is given by the block matrix
    # [ R t ]
    # [ 0 1 ],
    # where R is the 2x2 rotation and t is the translation (x, y).T of the sensor in the map frame.
    # Then, the inverse transform is
    # [ inv(R) -inv(R)@t ]
    # [     0       1    ]
    theta = x_t[2]
    ct = np.cos(theta)
    st = np.sin(theta)
    b_T_m = np.array([
        [ct,    st,     -x_t[0] * ct - x_t[1] * st],
        [-st,   ct,     x_t[0] * st - x_t[1] * ct ],
        [0.,    0.,                 1.            ]
    ])

    # Generate simulated sensor noise.
    rng = np.random.default_rng()
    noise = np.array([
            rng.normal(scale=np.sqrt(Q[0][0])),
            rng.normal(scale=np.sqrt(Q[1][1])),
            0.
    ])

    # Transform the (known) landmark positions to the sensor frame, and add noise.
    z_i = []
    j_i = in_range(x_t[:2], landmarks, max_range)
    for j in j_i:
        z = b_T_m @ np.append(landmarks[j], 1.) + noise
        z_i.append(z[:2])
    return j_i, z_i
