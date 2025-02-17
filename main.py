"""
Extended Kalman Filter SLAM example.

Plotting and ground truth generation code inspired by
https://github.com/AtsushiSakai/PythonRobotics/tree/master/SLAM/EKFSLAM
"""
import time

import numpy as np

from ekf_slam import DELTA_T, LANDMARKS, STATE_DIMS, get_landmark, get_landmark_count
from ekf_slam.ekf import F_x, g, get_expected_measurement, G_t_x, H_i_t, init_landmark, V_t_x
from ekf_slam.frames import sensor_to_map
from ekf_slam.vis import animate, plot_all
from ekf_slam.sim import MAX_RANGE, generate_trajectory, get_measurements, M_t, Q_t, SIM_TIME

INITIAL_POSE = np.array([0., 0., 0.])
INITIAL_LM_COV = 1e6


ANIMATE_PLOT = False
# Set this to the path to which we should save a new animated .gif of the run.
# E.g. '/home/rick/src/ekf_slam/EKF_SLAM.gif'. Set to '' to skip saving.
SAVE_ANIMATED_PLOT_TO = ''


def main():
    t_sim_start = time.time()
    t = 0.0

    # Full state column vector,length 3+2*N, where N is the number of landmarks.
    mu_t_0 = np.zeros(STATE_DIMS)  # Motion model-based state prediction. LaTeX: \bar \mu_t
    S_t_0 = np.eye(STATE_DIMS) * INITIAL_LM_COV # LaTeX: \Sigma_t

    # Init pose and pose covariance.
    mu_t_0[:3] = INITIAL_POSE
    S_t_0[:3, :3] = np.zeros((3, 3))

    # Constant control input.
    u_t = np.array([1.0, 0.1])

    # Init history. We pre-generate ground-truth and dead-reckoning.
    mu_t_h = [mu_t_0]
    mu_t_bar_gt_h = generate_trajectory(u_t, mu_t_0, SIM_TIME, DELTA_T)  # Ground-truth.
    mu_t_bar_dr_h = generate_trajectory(u_t, mu_t_0, SIM_TIME, DELTA_T, M_t)  # Dead-reckoning.
    S_t_h = [S_t_0]
    z_h = [[]]

    # Matrix that maps from 3D pose space [x  y  theta].T to the full EKF
    # state space [x_R m].T, shape == (2N+3,).
    Fx = F_x(len(LANDMARKS))

    while SIM_TIME >= t:
        # Predict motion.
        mu_t_bar = g(u_t, mu_t_h[-1], M=M_t)  # Prediction of next state with some additive noise.

        # Predict covariance of the predicted motion.
        G_t = np.eye(STATE_DIMS) + Fx.T @ G_t_x(u_t, mu_t_h[-1]) @ Fx

        # V_t: jacobian of the function that maps control space noise (v_t, omega_t)
        # to state space (x, y, theta).
        V_t = V_t_x(u_t, mu_t_h[-1])
        R_t = V_t @ M_t @ V_t.T
        S_t_bar = G_t @ S_t_h[-1] @ G_t.T + Fx.T @ R_t @ Fx

        # Observe. Measurements are expressed in the sensor frame.
        j_i, z_i = get_measurements(mu_t_bar_gt_h[int(t / DELTA_T)], LANDMARKS, MAX_RANGE, Q=Q_t)
        z_map = [sensor_to_map(z, mu_t_bar[:3]) for z in z_i]  # Save these for later visualization.

        # Correct, based on available measurements.
        for j, z in zip(j_i, z_i):
            # If we have not yet observed this landmark,
            # use our measurement as our initial estimate.
            mu_t_j = get_landmark(mu_t_bar, j)
            if np.allclose(mu_t_j, np.zeros(2)):
                init_landmark(mu_t_bar, j, z)

            # Get the Jacobian of the expected measurement.
            H_i_t_j = H_i_t(mu_t_bar, j, get_landmark_count(mu_t_bar))

            # Kalman gain.
            try:
                # (2N+3, 2) = (2N+3,2N+3) @ (2N+3, 2) @ ((2, 2N+3) @ (2N+3, 2N+3) @ (2N+3, 2) + (2, 2))^-1
                K_i_t = (S_t_bar @ H_i_t_j.T) @ np.linalg.inv((H_i_t_j @ S_t_bar @ H_i_t_j.T) + Q_t)
            except np.linalg.LinAlgError as e:
                print(f"Exception: {e}")
                continue

            # Update state and covariance estimates for this observation.
            z_hat = get_expected_measurement(mu_t_bar, j)
            mu_t_bar = mu_t_bar + K_i_t @ (z - z_hat)
            mu_t_bar[2] = np.atan2(np.sin(mu_t_bar[2]), np.cos(mu_t_bar[2]))
            S_t_bar = (np.eye(STATE_DIMS) - K_i_t @ H_i_t_j) @ S_t_bar

        # Store history, for access to last state, and for plotting.
        mu_t_h.append(np.array(mu_t_bar))  # mu_t = mu_t_bar.
        S_t_h.append(np.array(S_t_bar))  # S_t = S_t_bar.
        z_h.append(zip(j_i, z_map))

        t += DELTA_T

    total_seconds = time.time() - t_sim_start
    iterations = int(SIM_TIME / DELTA_T)
    print(f"Ran {iterations} iterations in {total_seconds} seconds, rate: {SIM_TIME / DELTA_T / total_seconds} Hz")

    if ANIMATE_PLOT:
        animate(
            mu_t_bar_gt_h=mu_t_bar_gt_h,
            mu_t_bar_dr_h=mu_t_bar_dr_h,
            mu_t_h=mu_t_h,
            S_t_h=S_t_h,
            z_h=z_h,
            save_plot_to=SAVE_ANIMATED_PLOT_TO)
    else:
        plot_all(
            mu_t_bar_gt_h=mu_t_bar_gt_h,
            mu_t_bar_dr_h=mu_t_bar_dr_h,
            landmarks_gt=LANDMARKS,
            mu_t_h=mu_t_h,
            S_t_h=S_t_h)


if __name__ == '__main__':
    main()
