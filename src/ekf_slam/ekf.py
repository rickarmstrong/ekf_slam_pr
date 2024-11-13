from math import cos, sin

import numpy as np

from ekf_slam import dt, LM_DIMS, POSE_DIMS, N_LANDMARKS

# Maps 3D pose space x_R [x  y  theta].T to the full EKF state space [x_R m].T.
F_x = np.hstack((np.eye(POSE_DIMS), np.zeros((POSE_DIMS, LM_DIMS * N_LANDMARKS))))

def g(u_t, mu, R=np.array([0., 0.])):
    """
    Velocity motion model with optional additive zero-mean Gaussian noise.
    Args:
        u_t : np.array
            Current control command, v, theta. u_t.shape==(2,).
        mu : np.array
            Current (full) state vector.
        R : np.array, optional
            xy_velocity variance, theta variance, R.shape==(2,).
            Added to incoming translational and angular velocities.
            Default is zero noise.
    Returns:
        Predicted state based on the current state and velocity command.
    """
    rng = np.random.default_rng()
    v_bar_t = u_t[0] + rng.normal(scale=R[0])
    omega_bar_t = u_t[1] + rng.normal(scale=R[1])
    theta = mu[2]

    # Pose delta.
    delta_x = np.array([
        -(v_bar_t / omega_bar_t) * sin(theta) + sin(theta + (omega_bar_t * dt)),
        (v_bar_t / omega_bar_t) * cos(theta) - (v_bar_t / omega_bar_t) * cos(theta + (omega_bar_t * dt)),
        omega_bar_t * dt])

    # Current (full) state + pose delta.
    return mu + F_x.T @ delta_x


def get_vel_cmd():
    # Constantly driving around in a big circle.
    v = 1.0  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u
