from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import numpy as np

from ekf_slam.ekf import g

SIM_TIME = 40.0  # simulation time [s].
MAX_RANGE = 10.0 # Maximum observation range.

# Simulated measurement noise params. stdev of range and bearing measurements noise.
Q_sim = np.array([0., np.deg2rad(0.1)])
Q_t = np.diag([Q_sim[0] ** 2,  Q_sim[1] ** 2])

# Simulated process noise covariance.
R_sim = np.array([0.1, 0.05, np.deg2rad(0.1)])
R_t = np.diag([R_sim[0] ** 2, R_sim[1] ** 2, R_sim[2] ** 2])


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

def in_range(x_t, landmarks, max_range=MAX_RANGE):
    """
    Return a list of landmark indices corresponding to landmarks that are
    within range of pose.
    Args:
        x_t : np.array
            2D pose: (x, y). shape == (2,).
        landmarks : np.array
            landmarks.shape == (n, 2), where n is the number of 2D landmarks.
        max_range : float
            Distance threshold.
    Returns: a list of indices from the incoming array of landmarks that are in range.
    """
    assert x_t.shape == (2,)  # 2D pose.
    idx = []
    for j, lm in enumerate(landmarks):
        if np.linalg.norm(lm - x_t) <= max_range:
            idx.append(j)
    return idx


def generate_trajectory(u_t, initial_state, duration, time_step, R=np.diag([0.0, 0.0, 0.0])):
    t = 0.
    mu_t_h = [np.array(initial_state)]
    while duration >= t:
        mu_t_h.append(g(u_t, mu_t_h[-1], delta_t=time_step, R=R))
        t += time_step
    return mu_t_h


def get_measurements(x_t, landmarks, max_range, Q=Q_t):
    """
    Return a list of simulated landmark observations.
    Args:
        x_t : array-like
            2D pose: (x, y, theta).
        landmarks :
            Ground-truth landmarks. shape == (n, 2), where n is the number of 2D landmarks.
        max_range :
        Q : array-like
            Noise params for range, bearing.

    Returns:
        j_i: Indices of landmarks in-range.
        z_i: An (optionally noise-corrupted) range-bearing measurement (r, phi)
            of each landmark that is within range, or None if no landmarks are in range.
            phi is in the range [-pi, pi]. Measurement is expressed in the robot frame.
    """
    rng = np.random.default_rng()
    z_i = []
    j_i = in_range(x_t[:2], landmarks, max_range)
    for j in j_i:
        lm = landmarks[j]
        v_sensor_lm = lm - x_t[:2]  # Vector from sensor to landmark.

        # Calculate range, bearing, add sim noise.
        r = np.linalg.norm(v_sensor_lm) + rng.normal(scale=np.sqrt(Q[0][0]))
        phi = np.atan2(v_sensor_lm[1], v_sensor_lm[0]) + rng.normal(scale=np.sqrt(Q[1][1]))
        theta = x_t[2]  # Account for the rotation of the sensor.
        z_i.append(np.array([r, phi - theta]))
    return j_i, z_i
