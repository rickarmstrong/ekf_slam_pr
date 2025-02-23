from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ekf_slam import get_landmark_cov, LANDMARKS, LM_DIMS, POSE_DIMS, STATE_DIMS, jj, range_bearing
from ekf_slam.ekf import F_x_j, g, init_landmark
from ekf_slam.sim import in_range, get_measurements


def test_F_x_j():
    n_landmarks = 4
    Fxj = F_x_j(0, n_landmarks)
    Fxj_expected = np.array([
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    ])
    assert np.allclose(Fxj, Fxj_expected)

    Fxj = F_x_j(1, n_landmarks)
    Fxj_expected = np.array([
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    ])
    assert np.allclose(Fxj, Fxj_expected)

    Fxj = F_x_j(3, n_landmarks)
    Fxj_expected = np.array([
        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    ])
    assert np.allclose(Fxj, Fxj_expected)


def test_g():
    """Minimal smoke test."""
    rng = np.random.default_rng()
    u_t = np.array([1.0, 0.1])  # Velocity command: v, theta.
    mu_current = rng.normal(size=(POSE_DIMS + LM_DIMS * len(LANDMARKS)))
    mu_next = g(u_t, mu_current, len(LANDMARKS))
    assert mu_current.shape == mu_next.shape


def test_get_landmark_cov():
    # Example covariance matrix representing pose and two landmarks.
    sigma = np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 2., 0.],
        [0., 0., 0., 0., 0., 0., 2.],
    ])

    lm_0_cov_expected = np.array([
        [1., 0.],
        [0., 1.]
    ])
    lm_0_cov = get_landmark_cov(sigma, 0)
    assert np.allclose(lm_0_cov_expected, lm_0_cov)

    lm_1_cov_expected = np.array([
        [2., 0.],
        [0., 2.]
    ])
    lm_1_cov = get_landmark_cov(sigma, 1)
    assert np.allclose(lm_1_cov_expected, lm_1_cov)

    # Ask for a covariance sub-matrix that doesn't exist.
    lm_2_cov = get_landmark_cov(sigma, 2)
    assert len(lm_2_cov) == 0


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


def test_init_landmark():
    # State vector representing a robot and two landmarks.
    mu_t = np.zeros(POSE_DIMS + 2 * LM_DIMS)

    # Robot at (1, 0), looking down the x-axis.
    x = np.array([1., 0., 0.])  # x, y, theta.
    mu_t[:3] = x

    # Landmarks: one at "nine o'clock", relative to the robot,
    # another straight behind the robot.
    landmarks = np.array([
        [1., 1.],
        [-1., 0.]
    ])

    # Robot-frame range-bearing measurements corresponding to the landmarks.
    z_t = np.array([
        [1., np.pi / 2.],
        [2., np.pi]
    ])

    for j, z in enumerate(z_t):
        init_landmark(mu_t, j, z)
        assert np.allclose(mu_t[jj(j): jj(j) + LM_DIMS], landmarks[j])


def test_in_range():
    landmarks  = np.array([
        [1., 0.],
        [2., 2.],  # Out of range.
        [0., 1.]
    ])
    x = np.array([0., 0.])
    j = in_range(x, landmarks, 1.0)
    assert np.allclose(landmarks[j], np.array([
        [1., 0.],
        [0., 1.]
    ]))


def test_get_measurements_zero_noise():
    """Call get_measurements() with the sensor rotated to several orientations.
    """
    # x, y, theta: at origin, looking down the x-axis.
    x_t = np.array([0., 0., 0.])

    # Two landmarks in-range, one out.
    landmarks  = np.array([
        [1., 0.],
        [2., 2.],  # Out of range.
        [0., 1.]
    ])

    expected = np.array([
        [1., 0.],
        [0., 1.]
    ])
    j_i, z_i_t = get_measurements(x_t, landmarks, max_range=1.0, Q=np.zeros((2, 2)))
    assert np.allclose(np.array(z_i_t), expected)

    # All landmarks out-of-range.
    j_i, z_i_t = get_measurements(x_t, landmarks, max_range=0.1, Q=np.zeros((2, 2)))
    assert len(j_i) == 0.
    assert len(z_i_t) == 0.

    # Rotate the sensor to point up the y-axis.
    x_t = np.array([0., 0., np.pi / 2.])
    j_i, z_i_t = get_measurements(x_t, landmarks, max_range=1.0, Q=np.zeros((2, 2)))
    expected = np.array([
        [0., -1.],
        [1., 0.]
    ])
    assert np.allclose(np.array(z_i_t), expected)

    # Rotate the sensor to point down the negative x-axis.
    x_t = np.array([0., 0., -np.pi])
    j_i, z_i_t = get_measurements(x_t, landmarks, max_range=1.0, Q=np.zeros((2, 2)))
    expected = np.array([
        [-1., 0.],
        [0., -1.]
    ])
    assert np.allclose(np.array(z_i_t), expected)


def test_measure_noisy(plot_observations=False):

    # x, y, theta: at origin, looking up the y-axis.
    x_t = np.array([0., 0., np.pi / 2.])

    # Two landmarks in-range, one out.
    max_range = 2.0
    landmarks  = np.array([
        [1., 0.],
        [2., 2.],  # Out of range.
        [0., 1.]
    ])

    # Take a bunch of measurements and verify that the statistics make sense.
    N = 100
    z_t = {}  # Entries look like {landmark_id: [measurements]}.
    for k in range(N):
        jj, zz = get_measurements(x_t, landmarks, max_range)
        for j, z in zip(jj, zz):
            try:
                z_t[j].append(z)
            except KeyError:
                z_t[j] = [z]  # First sighting.

    # Plot the observations.
    if plot_observations:
        for j in z_t.keys():
            for obs in np.array(z_t[j]):
                r = obs[0]
                theta = obs[1]
                x = r * cos(theta)
                y = r * sin(theta)
                plt.plot(x, y, '.b')
        plt.show()

# @pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
@pytest.mark.parametrize("sensor_pose,landmark,z_expected", [
    # Sensor at the origin, landmark on the x-axis at (1., 0). Positive rotations
    # through the 'cardinal' directions, i.e. x, y, -x, -y.
    ([0., 0., 0.], [1., 0], [1., 0.]),
    ([0., 0., np.pi / 2.], [1., 0], [1., -np.pi / 2.]),
    ([0., 0., 2. * np.pi / 2.], [1., 0], [1., -np.pi]),
    ([0., 0., 3. * np.pi / 2.], [1., 0], [1., np.pi / 2.]),

    # Sensor at the origin, landmark on the x-axis at (1., 0). Negative rotations
    # through the 'cardinal' directions, i.e. -y, -x, y.
    ([0., 0., -np.pi / 2.], [1., 0], [1., np.pi / 2.]),
    ([0., 0., 2. * -np.pi / 2.], [1., 0], [1., np.pi]),
    ([0., 0., 3. * -np.pi / 2.], [1., 0], [1., -np.pi / 2.]),

    # Sensor away from the origin.
    ([1., 1., 0.], [2., 2.], [np.sqrt(2.), np.pi / 4.]),
    ([1., 1., np.pi / 4.], [2., 2.], [np.sqrt(2.), 0.]),
    ([1., 1., np.pi / 4.], [0., 0.], [np.sqrt(2.), -np.pi]),
    ([1., 1., -np.pi / 4.], [0., 0.], [np.sqrt(2.), -np.pi / 2.]),

])
def test_range_bearing(sensor_pose, landmark, z_expected):
    z = range_bearing(np.array(sensor_pose), np.array(landmark))
    assert np.allclose(z, z_expected)
