import numpy as np
from scipy.spatial import ConvexHull
from .utils import config

# Trapezoidal weights
def trapezoidal(x):
    """Trapezoidal weights for trapezoidal rule integration."""
    diff = np.diff(x, axis = 0)
    weights = np.zeros(x.shape, dtype=config.real(np))
    weights[1:-1] = diff[1:] + diff[:-1]
    weights[0] = diff[0]
    weights[-1] = diff[-1]
    weights = weights / 2
    return weights

# Uniform weights
def uniform(x):
    """Uniform weights for Monte-Carlo integration."""
    Nx = x.shape[0]
    # Dimension 1
    if x.shape[1] < 2:
        volume = np.max(x) - np.min(x)
    # Approximate area using convex hull
    else:
        hull = ConvexHull(x)
        volume = hull.volume
    weights = volume*np.ones((Nx,1), dtype=config.real(np)) / Nx
    return weights

def get_weights(identifier, x):
    """Get the type of quadrature weights associated to the numpy array x."""
    
    if isinstance(identifier, str):
        return {
                "trapezoidal": trapezoidal(x),
                "uniform": uniform(x),
                }[identifier]