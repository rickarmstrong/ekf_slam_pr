import numpy as np


def map_to_sensor(p, x_t):
    """Map a point p, expressed in the map frame, to the sensor frame at pose x_t.
    Args:
        p : array_like
            point (x, y) expressed in the map frame.
        x_t : array_like
            pose (x, y, theta) of the sensor expressed in the map frame.

    Returns:
        The (x, y) point p, expressed in the sensor frame at pose x.
    """
    # Calculate the homogeneous map->sensor transform:
    # The sensor->map frame transformation is given by the block matrix
    # [ R t ]
    # [ 0 1 ],
    # where R is the 2x2 rotation and t is the translation (x, y).T of the sensor in the map frame.
    # Then, the inverse transform is
    # [ inv(R) -inv(R)@t ]
    # [     0       1    ]
    theta = x_t[2]
    ct = np.cos(theta)
    st = np.sin(theta)
    b_T_m = np.array([
        [ct,    st,     -x_t[0] * ct - x_t[1] * st],
        [-st,   ct,     x_t[0] * st - x_t[1] * ct ],
        [0.,    0.,                 1.            ]
    ])

    # Temporarily promote x_t to a homogeneous point, transform, then back to 2D.
    return (b_T_m @ np.append(p, 1.))[:2]


def sensor_to_map(p, x_t):
    """Map a point p, expressed in the sensor frame at pose x_t, to the map frame
    Args:
        p : array_like
            point (x, y) expressed in the sensor frame.
        x_t : array_like
            pose (x, y, theta) of the sensor, expressed in the map frame.

    Returns:
        The (x, y) point p, expressed in the sensor frame at pose x.
    """
    # The homogeneous sensor->map frame transformation is given by the block matrix
    # [ R t ]
    # [ 0 1 ],
    # where R is the 2x2 rotation and t is the translation (x, y).T of the sensor in the map frame.
    theta = x_t[2]
    ct = np.cos(theta)
    st = np.sin(theta)
    m_T_b = np.array([
        [ct, -st, x_t[0]],
        [st, ct, x_t[1]],
        [0., 0., 1.]
    ])
    # Temporarily promote x_t to a homogeneous point, transform, then back to 2D.
    return (m_T_b @ np.append(p, 1.))[:2]
