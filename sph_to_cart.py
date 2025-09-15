import numpy as np

def mod_sph_to_cart(alt, azi):
    """ Unit vector representing a solar ray defined by alt(itude) and azi(muth) angles.
        Input is in degrees. Azimuth counts CW from +y (N)

        Returns: numpy array representing a cartesian vector.
    """
    alt_rads = np.deg2rad(alt)
    azi_rads = np.deg2rad(azi)
    x = - np.cos(alt_rads) * np.sin(azi_rads)
    y = - np.cos(alt_rads) * np.cos(azi_rads)
    z = - np.sin(alt_rads)

    return np.array([x, y, z])