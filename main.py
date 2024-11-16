"""
Extended Kalman Filter SLAM example.

Plotting and ground truth generation code borrowed from
https://github.com/AtsushiSakai/PythonRobotics/tree/master/SLAM/EKFSLAM
"""
import matplotlib.pyplot as plt
import numpy as np

from ekf_slam import DELTA_T, N_LANDMARKS, POSE_DIMS, R_t, STATE_DIMS
from ekf_slam.ekf import F_x, g, get_vel_cmd, G

SIM_TIME = 40.0  # simulation time [s].
MAX_RANGE = 20.0  # Maximum observation range.
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.

# Initial robot pose and landmark ground truth: EKF SLAM can start from uninitialized landmark locations,
# but we start with a fixed number of known locations for simplicity.
INITIAL_POSE = np.zeros((POSE_DIMS, 1))
LANDMARKS = np.array([
    [10.0, -2.0],
    [15.0, 10.0],
    [3.0, 15.0],
    [-5.0, 20.0]])
assert len(LANDMARKS) == N_LANDMARKS

SHOW_PLOT = False

def main():
    t = 0.0

    # Full state column vector,length 3+2*N, where N is the number of landmarks.
    mu_bar = np.zeros(STATE_DIMS)  # Model-based state prediction.
    mu_bar_prev = np.zeros(STATE_DIMS)  # Previous prediction, i.e. at t-1.
    mu_gt_prev = np.zeros(STATE_DIMS)  # Ground truth.

    S_bar = np.zeros((STATE_DIMS, STATE_DIMS))
    S_bar_prev = np.zeros((STATE_DIMS, STATE_DIMS))

    # Init history.
    mu_gt_h = mu_bar
    mu_bar_h = mu_bar
    S_bar_h = S_bar

    while SIM_TIME >= t:
        u_t, u_t_noisy = get_vel_cmd()
        mu_gt = g(u_t, mu_gt_prev)  # Noise-free prediction of next state, keeping only the pose for ground truth.
        mu_bar = g(u_t_noisy, mu_bar_prev)  # Prediction of next state with some additive noise.

        # Update predicted covariance.
        G_t = F_x.T @ G(u_t_noisy, mu_bar_prev) @ F_x
        S_bar = G_t @ S_bar_prev @ G_t.T + F_x.T @ R_t @ F_x

        # Store history for plotting.
        mu_gt_h = np.vstack((mu_gt_h, mu_gt))
        mu_bar_h = np.vstack((mu_bar_h, mu_bar))
        S_bar_h = np.vstack((S_bar_h, S_bar))

        t += DELTA_T
        mu_gt_prev = mu_gt
        mu_bar_prev = mu_bar

    plt.plot(mu_gt_h[:, 0], mu_gt_h[:, 1], '-b')
    plt.plot(mu_bar_h[:, 0], mu_bar_h[:, 1], '-k')
    plt.axis('square')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
