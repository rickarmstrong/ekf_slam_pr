from math import cos, sin

import numpy as np

from ekf_slam import get_landmark, get_landmark_count, DELTA_T, LM_DIMS, POSE_DIMS, jj
from ekf_slam.frames import map_to_sensor


def F_x(n_landmarks):
    """Return a matrix that maps from 3D pose space [x  y  theta].T to the full EKF
    state space [x_R m].T, shape == (2N+3,)"""
    return np.hstack((np.eye(POSE_DIMS), np.zeros((POSE_DIMS, LM_DIMS * n_landmarks))))


def F_x_j(j, n_landmarks):
    """Build a matrix that maps the 2x5 jacobian of the measurement function to the
    full EKF covariance space (2N+3 x 2N+3).

    See http://ais.informatik.uni-freiburg.de/teaching/ws13/mapping/pdf/slam05-ekf-slam.pdf. Note: Stachniss' description
    of EKF SLAM drops the "signature" of measurements, for clarity. Consequently, the dimensions of this matrix
    (and others) differ from Table 10.1 in the Thrun book.
    Args:
        j : int
            Zero-based landmark index.
        n_landmarks : int
            Number of landmarks. We use this to pad the matrix to the full state dimensions.
    Returns:
        F : np.array.shape == (2N+3, 2N+3).
    """
    # We use zero-based indices everywhere, but Thrun et al use one-based landmark indices, like normal people.
    # For this purpose, it's useful to use one-based landmark indices, to match the literature.
    jn = j + 1

    # Build left-to-right.
    F = np.block([
        [np.eye(3)],
        [np.zeros((2, 3))]
    ])

    # Add first padding block, if needed.
    if jn > 1:
        pad_1 = np.zeros((5, 2*jn - 2))
        F = np.hstack((F, pad_1))

    # These columns select the landmark of interest.
    selector = np.vstack((np.zeros((3, 2)), np.eye(2)))
    F = np.hstack((F, selector))

    # Add the final padding.
    pad_2 = np.zeros((5, 2*n_landmarks - 2*jn))
    F = np.hstack((F, pad_2))
    return F


def g(u_t, mu, delta_t=DELTA_T, M=np.diag([0.0, 0.0])):
    """
    Noise-free velocity motion model, with the option to add Gaussian process noise.
    Args:
        u_t : np.array
            Current control command: (v, theta). u_t.shape==(2,).
        mu : np.array
            Current (full) state vector. mu.shape==(STATE_DIMS,).
        delta_t : float, optional
            Time step for the prediction, in seconds.
        M : np.array shape==(2, 2).
            Control noise params. We use only the velocity angular velocity variances, from the diagonal.
    Returns:
        Predicted state based on the current state, time step, and velocity command.
        Shape == (STATE_DIMS,).
    """
    v_t = u_t[0]
    omega_t = u_t[1]
    theta = mu[2]

    # Add control noise.
    rng = np.random.default_rng()
    v_t += rng.normal(scale=np.sqrt(M[0][0]))
    omega_t +=  rng.normal(scale=np.sqrt(M[1][1]))

    # The control command u_t represents a circular trajectory, whose radius
    # is abs(v_t / omega_t). To reduce clutter we'll rename the signed ratio v/omega.
    r_signed = v_t / omega_t

    # Pose delta.
    delta_x = np.array([
        -r_signed * sin(theta) + r_signed * sin(theta + (omega_t * delta_t)),
        r_signed * cos(theta) - r_signed * cos(theta + (omega_t * delta_t)),
        omega_t * delta_t])

    # Current (full) state + pose delta.
    mu_next = mu + F_x(get_landmark_count(mu)).T @ delta_x
    mu_next[2] = np.atan2(np.sin(mu_next[2]), np.cos(mu_next[2]))  # Normalize to [-pi, pi].
    return mu_next


def get_expected_measurement(mu_t_bar, j):
    """Given the current state vector, return the expected measurement of landmark j
    which is expressed in the map frame, in the sensor frame."""
    return map_to_sensor(get_landmark(mu_t_bar, j), mu_t_bar[:3])


def G_t_x(u_t, mu, delta_t=DELTA_T):
    """Return the 3x3 Jacobian of the motion model function g()."""
    v_t = u_t[0]
    omega_t = u_t[1]
    theta = mu[2]

    # The control command u_t represents a circular trajectory, whose radius
    # is abs(v_t / omega_t). To reduce clutter we'll rename the signed ratio v/omega.
    r_signed = v_t / omega_t

    return np.array([
        [0., 0., -r_signed * cos(theta) + r_signed * cos(theta + omega_t * delta_t )],
        [0., 0., -r_signed * sin(theta) + r_signed * sin(theta + omega_t * delta_t)],
        [0., 0., 0.]])


def H_i_t(mu_t, j, n_landmarks):
    """Return the high-dimensional Jacobian of the sensor model.
    """
    mu_j = get_landmark(mu_t, j)
    theta = mu_t[2]
    H_0_0 = -np.cos(theta)
    H_0_1 = -np.sin(theta)
    H_0_2 = -mu_j[0] * np.sin(theta) + mu_j[1] * np.cos(theta) + mu_t[0] * np.sin(theta) + mu_t[1] * np.cos(theta)
    H_0_3 = np.cos(theta)
    H_0_4 = np.sin(theta)
    H_1_0 = np.sin(theta)
    H_1_1 = -np.cos(theta)
    H_1_2 = -mu_j[0] * np.cos(theta) - mu_j[1] * np.sin(theta) + mu_t[0] * np.cos(theta) + mu_t[1] * np.sin(theta)
    H_1_3 = -np.sin(theta)
    H_1_4 = np.cos(theta)
    H_low = np.array([
        [H_0_0, H_0_1, H_0_2, H_0_3, H_0_4],
        [H_1_0, H_1_1, H_1_2, H_1_3, H_1_4]
    ])
    return H_low @ F_x_j(j, n_landmarks)


def init_landmark(mu_t, j, z):
    """
    Set the map-frame position of landmark j in mu_t to match the
    range-bearing measurement z.
    Args:
        mu_t : np.array
            State vector.
        j : int
            Index of the landmark we wish to update.
        z : np.array
            Range, bearing from the robot frame to the landmark. shape == (2,).

    Returns: None. Mutates the jth landmark in mu_t with the map-frame location of
    the observed landmark, based on the current robot pose.
    """
    x, y, theta = mu_t[:POSE_DIMS]
    r, phi = z
    mu_t[jj(j): jj(j) + LM_DIMS] = np.array([
        x + r * cos(phi + theta),
        y + r * sin(phi + theta)])


def V_t_x(u_t, mu, delta_t=DELTA_T):
    v_t = u_t[0]
    w_t = u_t[1]  # "omega_t"
    theta = mu[2]

    s_t = np.sin(theta)
    c_t = np.cos(theta)
    s_w_t = np.sin(theta + w_t * delta_t)
    c_w_t = np.cos(theta + w_t * delta_t)

    V_0_0 = (1. / w_t) * (-s_t + s_w_t)
    V_0_1 = (v_t / w_t ** 2) * s_t  - s_w_t + (v_t / w_t) * c_w_t * delta_t
    V_1_0 = (1. / w_t) * (c_t - c_w_t)
    V_1_1 = -(v_t / w_t ** 2) * c_t - c_w_t + (v_t / w_t) * s_w_t * delta_t

    return np.array([
        [V_0_0, V_0_1],
        [V_1_0, V_1_1],
        [0., delta_t]])
