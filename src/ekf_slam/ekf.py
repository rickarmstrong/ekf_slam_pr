from math import cos, sin

import numpy as np

from ekf_slam import DELTA_T, LM_DIMS, POSE_DIMS, N_LANDMARKS, jj

# Maps from 3D pose space [x  y  theta].T to the full EKF state space [x_R m].T, shape == (2N+3,)
F_x = np.hstack((np.eye(POSE_DIMS), np.zeros((POSE_DIMS, LM_DIMS * N_LANDMARKS))))


def F_x_j(j):
    """Build a matrix that maps the 2x5 jacobian of the measurement function to the
    full EKF covariance space (2N+3 x 2N+3).

    See http://ais.informatik.uni-freiburg.de/teaching/ws13/mapping/pdf/slam05-ekf-slam.pdf. Note: Stachniss' description
    of EKF SLAM drops the "signature" of measurements, for clarity. Consequently, the dimensions of this matrix
    (and others) differ from Table 10.1 in the Thrun book.
    Args:
        j : int
            Zero-based landmark index.
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
    pad_2 = np.zeros((5, 2*N_LANDMARKS - 2*jn))
    F = np.hstack((F, pad_2))
    return F


def g(u_t, mu, delta_t=DELTA_T):
    """
    Noise-free velocity motion model.
    Args:
        u_t : np.array
            Current control command: (v, theta). u_t.shape==(2,).
        mu : np.array
            Current (full) state vector. mu.shape==(STATE_DIMS,).
        delta_t : float, optional
            Time step for the prediction, in seconds.
    Returns:
        Predicted state based on the current state, time step, and velocity command.
        Shape == (STATE_DIMS,).
    """
    v_t = u_t[0]
    omega_t = u_t[1]
    theta = mu[2]

    # The control command u_t represents a circular trajectory, whose radius
    # is abs(v_t / omega_t). To reduce clutter we'll rename the signed ratio v/omega.
    r_signed = v_t / omega_t

    # Pose delta.
    delta_x = np.array([
        -r_signed * sin(theta) + r_signed * sin(theta + (omega_t * delta_t)),
        r_signed * cos(theta) - r_signed * cos(theta + (omega_t * delta_t)),
        omega_t * delta_t])

    # Current (full) state + pose delta.
    return mu + F_x.T @ delta_x


def get_expected_measurement(mu_t, j):
    """
    Return the expected measurement (range, bearing) for estimated landmark j,
    and current position estimate, taken from the current full state vector.
    Args:
        mu_t : np.array
            shape == (STATE_DIMS,).
        j : int
            The landmark index.
    Returns:
        The expected range/bearing of the landmark.
    """
    x_t = mu_t[:2]
    lm_bar = mu_t[jj(j): jj(j) + LM_DIMS]
    v = lm_bar - x_t

    # r, phi.
    return np.array([np.linalg.norm(v), np.atan2(v[1], v[0])])


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


def H_i_t(d, q, j):
    d_x = d[0]
    d_y = d[1]
    sqrt_q = np.sqrt(q)
    H_low = 1. / q * np.array([
        [-sqrt_q * d_x, -sqrt_q * d_y,  0,  sqrt_q * d_x,   sqrt_q * d_y],
        [d_y,           -d_x,           -q, -d_y,           d_x]
    ])
    return H_low @ F_x_j(j)


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
