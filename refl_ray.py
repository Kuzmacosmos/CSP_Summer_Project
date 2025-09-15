import numpy as np

def refl_ray(ray_in, pl_norm):
    """ Computes the reflected ray on a surface based on the vectorial form of Law of Reflection.

        Referring to eq. (2.162), p. 69 of the textbook Springer handbook of lasers and optics; 2nd ed.

        https://doi.org/10.1007/978-3-642-19409-2
    """
    a1 = ray_in / np.linalg.norm(ray_in)
    n = pl_norm / np.linalg.norm(pl_norm)
    if np.dot(a1, n) > 0:
        n = -n
    d = np.dot(a1, n)
    return a1 - 2 * d * n

