import numpy as np

from ekf_slam import LM_DIMS, POSE_DIMS, N_LANDMARKS
from ekf_slam.ekf import g

def test_g():
    """Minimal smoke test."""
    rng = np.random.default_rng()
    u_t = np.array([1.0, 0.1])  # v, theta.
    mu_current = rng.normal(size=(POSE_DIMS + LM_DIMS * N_LANDMARKS))
    mu_next = g(u_t, mu_current)
    assert mu_current.shape == mu_next.shape
