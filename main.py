"""
Extended Kalman Filter SLAM example.

Plotting and ground truth generation code borrowed from
https://github.com/AtsushiSakai/PythonRobotics/tree/master/SLAM/EKFSLAM
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np

from ekf_slam import DELTA_T, LANDMARKS, STATE_DIMS, get_landmark, get_landmark_cov
from ekf_slam.ekf import F_x, g, get_expected_measurement, G_t_x, init_landmark
from ekf_slam.sim import confidence_ellipse, MAX_RANGE, generate_trajectory, get_measurements, Q_t, R_t, SIM_TIME

INITIAL_POSE = np.array([0., 0., 0.])
SHOW_PLOT = False


def main():
    t = 0.0

    # Full state column vector,length 3+2*N, where N is the number of landmarks.
    mu_t_0 = np.zeros(STATE_DIMS)  # Motion model-based state prediction. LaTeX: \bar \mu_t
    S_t_0 = np.eye(STATE_DIMS) * 1000000 # LaTeX: \Sigma_t

    # Init pose and pose covariance.
    mu_t_0[:3] = INITIAL_POSE
    S_t_0[:3, :3] = np.zeros((3, 3))

    # Constant control input.
    u_t = np.array([1.0, 0.1])

    # Init history. We pre-generate ground-truth and dead-reckoning.
    mu_t_h = [mu_t_0]
    mu_t_bar_gt_h = generate_trajectory(u_t, mu_t_0, SIM_TIME, DELTA_T)  # Ground-truth.
    mu_t_bar_dr_h = generate_trajectory(u_t, mu_t_0, SIM_TIME, DELTA_T, R_t)  # Dead-reckoning.
    S_t_h = [S_t_0]

    while SIM_TIME >= t:
        ## Predict. ###
        mu_t_bar = g(u_t, mu_t_h[-1], R=R_t)  # Prediction of next state with some additive noise.

        # Update predicted covariance.
        Fx = F_x(len(LANDMARKS))
        G_t = np.eye(STATE_DIMS) + Fx.T @ G_t_x(u_t, mu_t_h[-1]) @ Fx
        S_t_bar = G_t @ S_t_h[-1] @ G_t.T + Fx.T @ R_t @ Fx

        # Observe.
        j_i, z_i = get_measurements(mu_t_bar_gt_h[int(t / DELTA_T)], LANDMARKS, MAX_RANGE, Q=Q_t)

        # Correct, based on available measurements.
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

            # Update state and covariance estimates for this observation.
            mu_t_bar = mu_t_bar + K_i_t @ (z - z_hat)
            S_t_bar = (np.eye(STATE_DIMS) - K_i_t @ H_i_t_j) @ S_t_bar

        # Store history, for access to last state, and for plotting.
        mu_t_h.append(np.array(mu_t_bar))  # mu_t = mu_t_bar.
        S_t_h.append(np.array(S_t_bar))  # S_t = S_t_bar.

        t += DELTA_T

    fig, ax = plt.subplots()

    # Ground-truth robot positions.
    gt = np.vstack(mu_t_bar_gt_h)
    ax.plot(gt[:, 0], gt[:, 1], '.b')

    # Dead reckoning motion estimates.
    dr = np.vstack(mu_t_bar_dr_h)
    ax.plot(dr[:, 0], dr[:, 1], '.r')

    # Ground-truth landmark positions.
    ax.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], 'xb')

    # Robot position estimates.
    mu = np.vstack(mu_t_h)
    ax.plot(mu[:, 0], mu[:, 1], '+g')

    # Final landmark position estimates.
    # for j in range(len(LANDMARKS)):
    #     lm = get_landmark(mu_t_h[-1], j)
    #     ax.plot(lm[0], lm[1], '*r')
    # confidence_ellipse(lm[0], lm[1], get_landmark_cov(S_t_h[-1], j), ax, n_std=3, edgecolor='red')

    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # Plot theta.
    plt.figure(1)
    plt.plot(mu[:, 2])
    plt.show()

if __name__ == '__main__':
    main()
