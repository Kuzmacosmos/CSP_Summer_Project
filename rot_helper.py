import numpy as np

def rodrigues(v, k, angle):
    """Rodrigues' formula for rotation of point v around k axis about an angle."""
    k = k/np.linalg.norm(k)
    return (v * np.cos(angle) + np.cross(k,v) * np.sin(angle)
            + k * np.dot(k,v) * (1 - np.cos(angle))
            )