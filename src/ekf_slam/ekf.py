from math import cos, sin

import numpy as np

from ekf_slam import DELTA_T, LM_DIMS, POSE_DIMS, N_LANDMARKS, jj

# Maps from 3D pose space [x  y  theta].T to the full EKF state space [x_R m].T.
F_x = np.hstack((np.eye(POSE_DIMS), np.zeros((POSE_DIMS, LM_DIMS * N_LANDMARKS))))

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
