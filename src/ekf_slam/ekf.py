from math import cos, sin

import numpy as np

from ekf_slam import dt, LM_DIMS, POSE_DIMS

# Maps 3D pose space x_R [x  y  theta].T to the full EKF state space [x_R m].T.
F_x = np.hstack((np.eye(POSE_DIMS), np.zeros((POSE_DIMS, POSE_DIMS + LM_DIMS))))

def g(u_t, mu):
    """
    Noise-free velocity motion model.
    Args:
        u_t : np.array
            Current control command, v, theta. u_t.shape==(2,).
        mu : np.array
            Current (full) state vector.
    Returns:
        Predicted state based on the current state and velocity command.
    """
    v_t = u_t[0]
    omega_t = u_t[1]
    theta = mu[2]

    # Pose delta.
    delta_x = np.array([
        -(v_t / omega_t) * sin(theta) + sin(theta + (omega_t * dt)),
        (v_t / omega_t) * cos(theta) - (v_t / omega_t) * cos(theta + (omega_t * dt)),
        omega_t * dt])

    # Current (full) state + pose delta.
    return mu + F_x.T @ delta_x
