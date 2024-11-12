"""
Extended Kalman Filter SLAM example.

Plotting and ground truth generation code borrowed from
https://github.com/AtsushiSakai/PythonRobotics/tree/master/SLAM/EKFSLAM
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from ekf_slam import dt, LM_DIMS, POSE_DIMS

SIM_TIME = 50.0  # simulation time [s].
MAX_RANGE = 20.0  # Maximum observation range.
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.

# Initial robot pose and landmark ground truth: EKF SLAM can start from uninitialized landmark locations,
# but we start with a fixed number of known locations for simplicity.
INITIAL_POSE = np.zeros((POSE_DIMS,))
LANDMARKS = np.array([
    [10.0, -2.0],
    [15.0, 10.0],
    [3.0, 15.0],
    [-5.0, 20.0]])

# Simulated noise params.
Q = np.diag([0.2, np.deg2rad(1.0)]) ** 2
R = np.diag([1.0, np.deg2rad(10.0)]) ** 2


SHOW_PLOT = False
def plot(hxDR, hxEst, hxTrue, landmarks, xEst):
    plt.cla()
    plt.plot(landmarks[:, 0], landmarks[:, 1], "*k")
    plt.plot(xEst[0], xEst[1], ".r")

    # plot landmarks
    for i in range(calc_n_lm(xEst)):
        plt.plot(xEst[POSE_DIMS + i * 2],
                 xEst[POSE_DIMS + i * 2 + 1], "xg")
    plt.plot(hxTrue[0, :],
             hxTrue[1, :], "-b")
    plt.plot(hxDR[0, :],
             hxDR[1, :], "-k")
    plt.plot(hxEst[0, :],
             hxEst[1, :], "-r")
    plt.axis("equal")
    plt.grid(True)


def get_vel_cmd():
    # Constantly driving around in a big circle.
    v = 1.0  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u


def calc_n_lm(x):
    n = int((len(x) - POSE_DIMS) / LM_DIMS)
    return n


def main():
    t = 0.0

    # Track ground truth robot pose (for display).
    x_true = np.zeros((POSE_DIMS, 1))

    # Full state column vector,length 3+2*N, where N is the number of landmarks.
    mu = np.zeros((POSE_DIMS + LM_DIMS * len(LANDMARKS), 1))
    S = np.zeros((len(mu), len(mu)))  # Sigma, full covariance matrix.

    # Init history.
    x_true_acc = np.array(INITIAL_POSE)
    m_acc = LANDMARKS
    mu_acc = mu
    S_acc = S

    while SIM_TIME >= t:
        u = get_vel_cmd()

        # store data history

        t += dt
        time.sleep(0.1)


if __name__ == '__main__':
    main()
