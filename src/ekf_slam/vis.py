import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from ekf_slam import get_landmark, get_landmark_cov, DELTA_T, LANDMARKS
from ekf_slam.sim import confidence_ellipse, MAX_RANGE, SIM_TIME

def animate(**kwargs):
    fig, ax = plt.subplots()
    ax.set_title(f"Duration: {SIM_TIME}s, Sensor Range: {MAX_RANGE}")

    # Ground-truth robot positions.
    gt = np.vstack(kwargs['mu_t_bar_gt_h'])
    gt_plot = ax.plot(gt[0, 0], gt[0, 1], '.b', label="Ground-truth")[0]

    # Use ground-truth bounds plus a little more, for the plot bounds.
    xmin = np.min(gt[:, 0])
    xmax = np.max(gt[:, 0])
    ymin = np.min(gt[:, 1])
    ymax = np.max(gt[:, 1])
    extra = 0.5
    xextra = np.abs(xmax * extra)
    yextra = np.abs(ymax * extra)
    ax.set(xlim=[xmin - xextra, xmax + xextra], ylim=[ymin - yextra, ymax + yextra], aspect='equal')

    # Dead reckoning motion estimates.
    dr = np.vstack(kwargs['mu_t_bar_dr_h'])
    dr_plot = ax.plot(dr[0, 0], dr[0, 1], '.r', label="Dead-reckoning")[0]

    # Ground-truth landmark positions.
    ax.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], 'xb')

    # Robot position estimates.
    mu = np.vstack(kwargs['mu_t_h'])
    mu_plot = ax.plot(mu[:, 0], mu[:, 1], '+g', label="EKF estimate")[0]

    # Landmark measurements.
    zt_plot = ax.plot(0, 0, '*g', label="Landmark measurement")[0]

    # Annotations.
    k_text = ax.text(0., 10., f"")

    ax.legend()

    def update(k):
        # Ground-truth.
        gt_plot.set_xdata(gt[:k, 0])
        gt_plot.set_ydata(gt[:k, 1])

        # Dead-reckoning.
        dr_plot.set_xdata(dr[:k, 0])
        dr_plot.set_ydata(dr[:k, 1])

        # Robot position estimates.
        mu_plot.set_xdata(mu[:k, 0])
        mu_plot.set_ydata(mu[:k, 1])

        # Landmark observations.
        zx = []
        zy = []
        for _, zh in kwargs['z_h'][k]:
            theta = mu[k, 2]
            zx.append(mu[k, 0] + zh[0] * np.cos(zh[1] + theta))
            zy.append(mu[k, 1] + zh[0] * np.sin(zh[1] + theta))
        if len(zx) > 0:
            zt_plot.set_xdata(zx)
            zt_plot.set_ydata(zy)

        k_text.set_text(f"t = {(k * DELTA_T):.2f}")

        return gt_plot

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(gt), interval=1)
    plt.show()
    # ani.save(filename="/home/rick/src/ekf_slam/EKF_SLAM.gif", writer="pillow")


def plot_all(**kwargs):
    fig, ax = plt.subplots()

    # Ground-truth robot positions.
    gt = np.vstack(kwargs['mu_t_bar_gt_h'])
    ax.plot(gt[:, 0], gt[:, 1], '.b')

    # Dead reckoning motion estimates.
    dr = np.vstack(kwargs['mu_t_bar_dr_h'])
    ax.plot(dr[:, 0], dr[:, 1], '.r')

    # Ground-truth landmark positions.
    ax.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], 'xb')

    # Robot position estimates.
    mu = np.vstack(kwargs['mu_t_h'])
    ax.plot(mu[:, 0], mu[:, 1], '+g')

    # Final landmark position estimates.
    for j in range(len(LANDMARKS)):
        lm = get_landmark(mu[-1], j)
        ax.plot(lm[0], lm[1], '*r')
        confidence_ellipse(lm[0], lm[1], get_landmark_cov(kwargs['S_t_h'][-1], j), ax, n_std=3, edgecolor='red')

    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # Plot theta, as a sanity check.
    plt.figure(1)
    plt.plot(mu[:, 2])
    plt.show()
