"""
Extended Kalman Filter SLAM example.

Plotting and ground truth generation code inspired by
https://github.com/AtsushiSakai/PythonRobotics/tree/master/SLAM/EKFSLAM
"""
import numpy as np

from ekf_slam import DELTA_T, LANDMARKS, STATE_DIMS, get_landmark, get_landmark_count, get_landmark_cov, range_bearing
from ekf_slam.ekf import F_x, g, G_t_x, H_i_t, init_landmark, V_t_x
from ekf_slam.vis import animate, plot_all
from ekf_slam.sim import MAX_RANGE, generate_trajectory, get_measurements,M_t, Q_t, SIM_TIME

INITIAL_POSE = np.array([0., 0., 0.])
INITIAL_LM_COV = 1e6
ANIMATE_PLOT = True


def main():
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

    while SIM_TIME >= t:
        # Predict motion.
        mu_t_bar = g(u_t, mu_t_h[-1], M=M_t)  # Prediction of next state with some additive noise.

        # Predict covariance of the predicted motion.
        Fx = F_x(len(LANDMARKS))
        G_t = np.eye(STATE_DIMS) + Fx.T @ G_t_x(u_t, mu_t_h[-1]) @ Fx

        # V_t: jacobian of the function that maps control space noise (v_t, omega_t)
        # to state space (x, y, theta).
        V_t = V_t_x(u_t, mu_t_h[-1])
        R_t = V_t @ M_t @ V_t.T
        S_t_bar = G_t @ S_t_h[-1] @ G_t.T + Fx.T @ R_t @ Fx

        # Observe.
        j_i, z_i = get_measurements(mu_t_bar_gt_h[int(t / DELTA_T)], LANDMARKS, MAX_RANGE, Q=Q_t)

        # Correct, based on available measurements.
        for j, z in zip(j_i, z_i):
            lm = get_landmark(mu_t_bar, j)
            if np.allclose(lm, np.zeros(2)):
                init_landmark(mu_t_bar, j, z)

            # Get the expected measurement.
            z_hat = range_bearing(mu_t_bar[:3], lm)

            # Get the Jacobian of the expected measurement.
            H_i_t_j = H_i_t(lm - mu_t_bar[:2], z_hat[0] ** 2, j, get_landmark_count(mu_t_bar))

            # Kalman gain.
            try:
                # (2N+3, 2) = (2N+3,2N+3) @ (2N+3, 2) @ ((2, 2N+3) @ (2N+3, 2N+3) @ (2N+3, 2) + (2, 2))^-1
                K_i_t = (S_t_bar @ H_i_t_j.T) @ np.linalg.inv((H_i_t_j @ S_t_bar @ H_i_t_j.T) + Q_t)
            except np.linalg.LinAlgError as e:
                print(f"Exception: {e}")
                continue

            # Update state and covariance estimates for this observation.
            mu_t_bar = mu_t_bar + K_i_t @ (z - z_hat)
            mu_t_bar[2] = np.atan2(np.sin(mu_t_bar[2]), np.cos(mu_t_bar[2]))
            S_t_bar = (np.eye(STATE_DIMS) - K_i_t @ H_i_t_j) @ S_t_bar

        # Store history, for access to last state, and for plotting.
        mu_t_h.append(np.array(mu_t_bar))  # mu_t = mu_t_bar.
        S_t_h.append(np.array(S_t_bar))  # S_t = S_t_bar.
        z_h.append(zip(j_i, z_i))

        t += DELTA_T

    if ANIMATE_PLOT:
        animate(
            mu_t_bar_gt_h=mu_t_bar_gt_h,
            mu_t_bar_dr_h=mu_t_bar_dr_h,
            mu_t_h=mu_t_h,
            S_t_h=S_t_h,
            z_h=z_h)
    else:
        plot_all(
            mu_t_bar_gt_h=mu_t_bar_gt_h,
            mu_t_bar_dr_h=mu_t_bar_dr_h,
            landmarks_gt=LANDMARKS,
            mu_t_h=mu_t_h,
            S_t_h=S_t_h)


if __name__ == '__main__':
    main()
