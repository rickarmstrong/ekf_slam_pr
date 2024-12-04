"""
Extended Kalman Filter SLAM example.

Plotting and ground truth generation code borrowed from
https://github.com/AtsushiSakai/PythonRobotics/tree/master/SLAM/EKFSLAM
"""
import matplotlib.pyplot as plt
import numpy as np

from ekf_slam import DELTA_T, LANDMARKS, POSE_DIMS, STATE_DIMS, get_landmark
from ekf_slam.ekf import F_x, g, get_expected_measurement, G_t_x, H_i_t, init_landmark
from ekf_slam.sim import MAX_RANGE, get_measurements, Q_t, R_t, SIM_TIME

INITIAL_POSE = np.array([0., 0., 0.])
SHOW_PLOT = False


def main():
    t = 0.0

    # Full state column vector,length 3+2*N, where N is the number of landmarks.
    mu_t_bar = np.zeros(STATE_DIMS)  # Motion model-based state prediction. LaTeX: \bar \mu_t
    mu_t_prev = np.zeros(STATE_DIMS) # LaTeX: \bar\mu_{t-1}
    mu_t_prev_gt = np.zeros(STATE_DIMS)  # Ground truth.

    S_t_prev = np.eye(STATE_DIMS)  # LaTeX: \Sigma_{t-1}
    S_t_bar = np.eye(STATE_DIMS)  # \bar\Sigma_t

    # Set initial pose.
    mu_t_bar[:3] = INITIAL_POSE
    mu_t_prev[:3] = INITIAL_POSE
    mu_t_prev_gt[:3] = INITIAL_POSE

    # Init history.
    mu_t_bar_gt_h = mu_t_bar
    mu_t_bar_h = mu_t_bar
    S_t_bar_h = S_t_bar

    while SIM_TIME >= t:
        ### Predict. ###
        u_t = np.array([1.0, 0.00000000001])
        mu_t_bar_gt = g(u_t, mu_t_prev_gt, len(LANDMARKS))  # Noise-free prediction of next state, keeping only the pose for ground truth.
        mu_t_bar = g(u_t, mu_t_prev, len(LANDMARKS), R=R_t)  # Prediction of next state with some additive noise.

        # Update predicted covariance.
        Fx = F_x(len(LANDMARKS))
        G_t = np.eye(STATE_DIMS) + Fx.T @ G_t_x(u_t, mu_t_prev) @ Fx
        S_t_bar = G_t @ S_t_prev @ G_t.T + Fx.T @ R_t @ Fx

        ### Observe. ###
        j_i, z_i = get_measurements(mu_t_bar_gt, LANDMARKS, MAX_RANGE, Q=Q_t)

        # Correct. ###
        for j, z in zip(j_i, z_i):
            if np.allclose(get_landmark(mu_t_bar, j), np.zeros(2)):
                init_landmark(mu_t_bar, j, z)

            z_hat, H_i_t_j = get_expected_measurement(mu_t_bar, j)

            # Kalman gain.
            try:
                # (2N+3, 2) = (2N+3,2N+3) @ (2N+3, 2) @ ((2, 2N+3) @ (2N+3, 2N+3) @ (2N+3, 2) + (2, 2))^-1
                K_i_t = (S_t_bar @ H_i_t_j.T) @ np.linalg.inv((H_i_t_j @ S_t_bar @ H_i_t_j.T) + Q_t)
            except np.linalg.LinAlgError as e:
                print(f"Exception: {e}")
                continue

            # Update our state.
            mu_t_bar = mu_t_bar + K_i_t @ (z - z_hat)
            mu_t_bar[2] = np.atan2(np.sin(mu_t_bar[2]),np.cos(mu_t_bar[2]))  # Normalize theta.
            S_bar = (np.eye(STATE_DIMS) - K_i_t @ H_i_t_j) @ S_t_bar

        # Store history for plotting.
        mu_t_bar_gt_h = np.vstack((mu_t_bar_gt_h, mu_t_bar_gt))
        mu_t_bar_h = np.vstack((mu_t_bar_h, mu_t_bar))
        S_t_bar_h = np.vstack((S_t_bar_h, S_t_bar))

        t += DELTA_T
        mu_t_prev_gt = mu_t_bar_gt
        mu_t_prev = mu_t_bar
        S_t_prev = S_t_bar

    # Ground-truth robot positions.
    plt.plot(mu_t_bar_gt_h[:, 0], mu_t_bar_gt_h[:, 1], '.b')

    # Robot position estimates.
    plt.plot(mu_t_bar_h[:, 0], mu_t_bar_h[:, 1], '+r')

    # Ground-truth landmark positions.
    plt.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], 'xb')

    # Final landmark position estimates.
    for j in range(len(LANDMARKS)):
        lm = get_landmark(mu_t_bar, j)
        plt.plot(lm[0], lm[1], '*r')

    plt.axis('equal')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
