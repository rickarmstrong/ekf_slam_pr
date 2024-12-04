from math import cos, sin

import matplotlib.pyplot as plt
import numpy as np

from ekf_slam import LANDMARKS, LM_DIMS, POSE_DIMS, STATE_DIMS, jj, new_cov_matrix
from ekf_slam.ekf import F_x_j, g, get_expected_measurement, init_landmark
from ekf_slam.sim import in_range, get_measurements


def test_new_cov_matrix():
    # Smoke test.
    C = new_cov_matrix()


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


def test_get_expected_measurement():
    # State vector representing a robot and two landmarks.
    n_landmarks = 2
    mu_t = np.zeros(POSE_DIMS + 2 * LM_DIMS)

    # Landmarks: one at "nine o'clock", relative to the robot,
    # another straight behind the robot.
    landmarks = np.array([
        [1., 1.],
        [-1., 0.]
    ])
    mu_t[jj(0): jj(0) + LM_DIMS] = landmarks[0]
    mu_t[jj(1): jj(1) + LM_DIMS] = landmarks[1]

    # Robot at (1, 0), looking down the x-axis.
    mu_t[:3] = np.array([1., 0., 0.])  # x, y, theta.

    expected_measurements = np.array([
        [1., np.pi / 2.],
        [2., np.pi]
    ])
    for j in range(n_landmarks):
        z_hat = get_expected_measurement(mu_t, j)
        assert np.allclose(z_hat, expected_measurements[j])


    # Robot at (1, 0), looking parallel to the positive y-axis.
    mu_t[:3] = np.array([1., 0., np.pi / 2.])  # x, y, theta.

    expected_measurements = np.array([
        [1., 0.],
        [2., np.pi / 2.]
    ])
    for j in range(n_landmarks):
        z_hat = get_expected_measurement(mu_t, j)
        assert np.allclose(z_hat, expected_measurements[j])


def test_g_one_sec():
    # After a delta_t of one second, we know where we should be.
    v_t = 1.0  # m/s
    omega_t = 1.0
    delta_t = 1.0  # 1.0 s

    u_t = np.array([v_t, omega_t])
    x_0 = np.zeros(STATE_DIMS)
    x_1 = g(u_t, x_0, len(LANDMARKS), delta_t)
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
    # x, y, theta: at origin, looking down the x-axis.
    x_t = np.array([0., 0., 0.])

    # Two landmarks in-range, one out.
    landmarks  = np.array([
        [1., 0.],
        [2., 2.],  # Out of range.
        [0., 1.]
    ])

    j_i, z_i_t = get_measurements(x_t, landmarks, max_range=1.0, Q=np.zeros((2, 2)))
    expected = np.array([
        [1., 0.],
        [1., np.pi / 2.]
    ])
    assert np.allclose(np.array(z_i_t), expected)

    # All landmarks out-of-range.
    j_i, z_i_t = get_measurements(x_t, landmarks, max_range=0.1, Q=np.zeros((2, 2)))
    assert len(j_i) == 0.
    assert len(z_i_t) == 0.

    # Rotate the sensor to point up the y-axis.
    x_t = np.array([0., 0., np.pi / 2.])
    j_i, z_i_t = get_measurements(x_t, landmarks, max_range=1.0, Q=np.zeros((2, 2)))
    expected = np.array([
        [1., -np.pi / 2.],
        [1., 0.]
    ])
    assert np.allclose(np.array(z_i_t), expected)

    # Rotate the sensor to point down the negative x-axis.
    x_t = np.array([0., 0., -np.pi / 2.])
    j_i, z_i_t = get_measurements(x_t, landmarks, max_range=1.0, Q=np.zeros((2, 2)))
    expected = np.array([
        [1., np.pi / 2.],
        [1., np.pi]
    ])
    assert np.allclose(np.array(z_i_t), expected)


def test_measure_noisy():

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
    if True:
        for j in z_t.keys():
            for obs in np.array(z_t[j]):
                r = obs[0]
                theta = obs[1]
                x = r * cos(theta)
                y = r * sin(theta)
                plt.plot(x, y, '.b')
        plt.show()
