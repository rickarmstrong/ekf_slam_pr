import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

import numpy as np

from ekf_slam import get_landmark, get_landmark_count, get_landmark_cov, DELTA_T, LANDMARKS
from ekf_slam.sim import MAX_RANGE, SIM_TIME

def animate(save_plot_to='', **kwargs):
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

    # Robot position confidence ellipses.
    pos_cov_ellipse = Ellipse((0, 0), width=1., height=1., facecolor='none', edgecolor='red')
    ax.add_patch(pos_cov_ellipse)

    # Landmark measurements.
    zt_plot = ax.plot(0, 0, '*g', label="Landmark measurement")[0]

    # Landmark position confidence ellipses.
    lm_cov_ellipses = []
    for j in range(get_landmark_count(mu[0])):
        lm_cov = get_landmark_cov(kwargs['S_t_h'][0], j)
        ell = Ellipse((0, 0), width=lm_cov[0, 0], height=lm_cov[1, 1], facecolor='none', edgecolor='red')
        lm_cov_ellipses.append(ax.add_patch(ell))

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

        # Robot position estimate confidence ellipse.
        rx, ry, tf = get_ellipse_params(mu[k][0], mu[k][1], kwargs['S_t_h'][k][:2, :2], 3.0)
        pos_cov_ellipse.width = rx
        pos_cov_ellipse.height = ry
        pos_cov_ellipse.set_transform(tf + ax.transData)

        # Landmark observations.
        zx = []
        zy = []
        for idx, zh in kwargs['z_h'][k]:
            lm_x = zh[0]
            lm_y = zh[1]
            zx.append(lm_x)
            zy.append(lm_y)

            # Update landmark confidence ellipses.
            rx, ry, tf = get_ellipse_params(lm_x, lm_y, get_landmark_cov(kwargs['S_t_h'][k], idx), 3.0)
            lm_cov_ellipses[idx].width = rx
            lm_cov_ellipses[idx].height = ry
            lm_cov_ellipses[idx].set_transform(tf + ax.transData)
        if len(zx) > 0:
            zt_plot.set_xdata(zx)
            zt_plot.set_ydata(zy)

        k_text.set_text(f"t = {(k * DELTA_T):.2f}")

        return gt_plot

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(gt), interval=1)
    plt.show()
    if save_plot_to != '':
        ani.save(filename=save_plot_to, writer="pillow")


def confidence_ellipse(x, y, cov, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse cov at location (x, y).

    Modified version of https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    x, y : float
        The location of center of the ellipse.

    cov : np.array, shape (2, 2)
        The 2x2 covariance matrix we with to represent.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radii.

    facecolor : color used to fill the ellipse, forwarded to `~matplotlib.patches.Ellipse`.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`. In particular, 'edgecolor' is passed through.

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    # cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    tf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x, y)

    ellipse.set_transform(tf + ax.transData)
    return ax.add_patch(ellipse)


def get_ellipse_params(x, y, cov, n_std=1.0):
    # cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    tf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(x, y)

    return ell_radius_x, ell_radius_y, tf


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
