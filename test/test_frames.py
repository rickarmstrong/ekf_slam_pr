import numpy as np

from ekf_slam.frames import  map_to_sensor, sensor_to_map

def test_map_to_sensor():
    """Check this transformation assuming the sensor looks along the platform's x-axis."""

    # A point on the map y-axis.
    p_map = np.array([0., 1.])

    # Sensor at origin, looking down the x-axis.
    x_t = np.array([0., 0., 0.])  # x, y, theta.
    expected = np.array([0., 1.])
    actual = map_to_sensor(p_map, x_t)
    assert np.allclose(actual, expected)

    # Sensor at (1, 1, pi), looking at the point down the x-axis.
    x_t = np.array([1., 1., np.pi])
    expected = np.array([1., 0.])
    actual = map_to_sensor(p_map, x_t)
    assert np.allclose(actual, expected)

    # Sensor at (1, 0, 0), no rotation.
    x_t = np.array([1., 0., 0])
    expected = np.array([-1., 1.])
    actual = map_to_sensor(p_map, x_t)
    assert np.allclose(actual, expected)

    # Sensor at (-1, 0, pi/4), looking at the point.
    x_t = np.array([-1., 0., np.pi / 4.])
    expected = np.array([np.sqrt(2), 0.])
    actual = map_to_sensor(p_map, x_t)
    assert np.allclose(actual, expected)


def test_sensor_to_map():
    # Sensor at (1, 1), no rotation.
    x_t = np.array([1., 1., 0.])

    # Point is directly underneath the sensor.
    p_sensor = np.array([0., 0.])
    expected = np.array([1., 1.])
    actual = sensor_to_map(p_sensor, x_t)
    assert np.allclose(actual, expected)

    # Point at the origin.
    p_sensor = np.array([-1., -1.])
    expected = np.array([0., 0.])
    actual = sensor_to_map(p_sensor, x_t)
    assert np.allclose(actual, expected)


    # Same, with sensor pointed at the origin.
    x_t[2] = 5. * np.pi / 4.
    p_sensor = np.array([np.sqrt(2.), 0.])
    expected = np.array([0., 0.])
    actual = sensor_to_map(p_sensor, x_t)
    assert np.allclose(actual, expected)
