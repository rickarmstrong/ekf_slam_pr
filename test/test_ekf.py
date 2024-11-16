from math import cos, sin


import numpy as np

from ekf_slam import LM_DIMS, N_LANDMARKS, POSE_DIMS, R_sim, STATE_DIMS
from ekf_slam.ekf import g

def test_g():
    """Minimal smoke test."""
    rng = np.random.default_rng()
    u_t = np.array([1.0, 0.1])  # Velocity command: v, theta.
    mu_current = rng.normal(size=(POSE_DIMS + LM_DIMS * N_LANDMARKS))
    mu_next = g(u_t, mu_current)
    assert mu_current.shape == mu_next.shape

def test_g_one_sec():
    # After a delta_t of one second, we know where we should be.
    v_t = 1.0  # m/s
    omega_t = 1.0
    delta_t = 1.0  # 1.0 s

    u_t = np.array([v_t, omega_t])
    x_0 = np.zeros(STATE_DIMS)
    x_1 = g(u_t, x_0, delta_t)
    assert np.isclose(x_1[0], sin(1.0))
    assert np.isclose(x_1[1], 1.0 - cos(1.0))
