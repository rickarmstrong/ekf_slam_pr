"""
Extended Kalman Filter SLAM example.

Plotting and ground truth generation code borrowed from
https://github.com/AtsushiSakai/PythonRobotics/tree/master/SLAM/EKFSLAM
"""
import matplotlib.pyplot as plt
import numpy as np

from ekf_slam import DELTA_T, N_LANDMARKS, POSE_DIMS, STATE_DIMS, new_cov_matrix, get_landmark
from ekf_slam.ekf import F_x, g, G_t_x, H_i_t, init_landmark
from ekf_slam.sim import MAX_RANGE, get_measurements, Q_t, R_t, SIM_TIME, validate_landmarks

INITIAL_POSE = np.zeros((POSE_DIMS, 1))
LANDMARKS = np.array([
    [10.0, -2.0],
    [15.0, 20.0],
    [3.0, 15.0],
    [-5.0, 20.0],
    [30., 5.]
])
validate_landmarks(LANDMARKS)

SHOW_PLOT = False


def main():
    t = 0.0

    # Full state column vector,length 3+2*N, where N is the number of landmarks.
    mu_bar = np.zeros(STATE_DIMS)  # Model-based state prediction.
    mu_bar_prev = np.zeros(STATE_DIMS)  # Previous prediction, i.e. at t-1.
    mu_gt_prev = np.zeros(STATE_DIMS)  # Ground truth.

    S_bar = np.eye(STATE_DIMS)
    S_bar_prev = np.eye(STATE_DIMS)

    # Init history.
    mu_gt_h = mu_bar
    mu_bar_h = mu_bar
    S_bar_h = S_bar

    while SIM_TIME >= t:
        ### Predict. ###
        u_t = np.array([1.0, 0.000000000001])
        mu_gt = g(u_t, mu_gt_prev)  # Noise-free prediction of next state, keeping only the pose for ground truth.
        mu_bar = g(u_t, mu_bar_prev, R=R_t)  # Prediction of next state with some additive noise.

        # Update predicted covariance.
        Fx = F_x(N_LANDMARKS)
        G_t = np.eye(STATE_DIMS) + Fx.T @ G_t_x(u_t, mu_bar_prev) @ Fx
        S_bar = G_t @ S_bar_prev @ G_t.T + Fx.T @ R_t @ Fx

        ### Observe. ###
        j_i, z_i = get_measurements(mu_gt, LANDMARKS, MAX_RANGE, Q=np.diag([0., 0.]))

        ### Correct. ###
        for j, z in zip(j_i, z_i):
            if np.allclose(get_landmark(mu_bar, j), np.zeros(2)):
                init_landmark(mu_bar, j, z)

            d = get_landmark(mu_bar, j) - mu_bar[:2]
            q = np.inner(d.T, d)
            z_hat = np.array([
                np.sqrt(q),
                np.atan2(d[1], d[0] - mu_bar[2])])
            H_i_t_j = H_i_t(d, q, j)

            # Kalman gain.
            try:
                K_i_t = (S_bar @ H_i_t_j.T) @ np.linalg.inv((H_i_t_j @ S_bar @ H_i_t_j.T) + Q_t)
            except np.linalg.LinAlgError as e:
                print(f"Exception: {e}")
                continue

            # Update our state.
            mu_bar = mu_bar + K_i_t @ (z - z_hat)
            S_bar = (np.eye(STATE_DIMS) - K_i_t @ H_i_t_j) @ S_bar

        # Store history for plotting.
        mu_gt_h = np.vstack((mu_gt_h, mu_gt))
        mu_bar_h = np.vstack((mu_bar_h, mu_bar))
        S_bar_h = np.vstack((S_bar_h, S_bar))

        t += DELTA_T
        mu_gt_prev = mu_gt
        mu_bar_prev = mu_bar
        S_bar_prev = S_bar

    # Ground-truth robot positions.
    plt.plot(mu_gt_h[:, 0], mu_gt_h[:, 1], '-b')

    # Robot position estimates.
    plt.plot(mu_bar_h[:, 0], mu_bar_h[:, 1], '+r')

    # Ground-truth landmark positions.
    plt.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], 'xb')

    # Landmark estimates.
    for j in range(N_LANDMARKS):
        lm = get_landmark(mu_bar, j)
        plt.plot(lm[0], lm[1], '*r')

    plt.axis('equal')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
