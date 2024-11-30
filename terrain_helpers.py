import numpy as np
import scipy as sp

def discretize_terrain(x, y, z, xy_resolution=5, num_z_bins=20):
    """
    Discretizes a continuous terrain into a 3D grid state space.
    
    Parameters:
    - x: 2D array of x coordinates.
    - y: 2D array of y coordinates.
    - z: 2D array of elevation values (same shape as x and y).
    - xy_resolution: Resolution for discretizing x and y (units per grid cell).
    - num_z_bins: Number of discrete bins for z (elevation) values.

    Returns:
    - state_map: 2D array where each entry is a tuple (x_index, y_index, z_index).
    - z_bins: The elevation bin edges used for discretization.
    """

    # Determine the grid size based on the xy resolution
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Create discrete indices for x and y based on the specified resolution
    x_indices = ((x - x_min) / xy_resolution).astype(int)
    y_indices = ((y - y_min) / xy_resolution).astype(int)

    # Discretize the z (elevation) values into bins
    z_flat = z.flatten()
    z_bins = np.linspace(z_flat.min(), z_flat.max(), num_z_bins + 1)
    z_indices = np.digitize(z.flatten(), bins=z_bins) - 1
    z_indices = z_indices.reshape(z.shape)
    print(z_indices)

    # Combine x, y, and z indices into a state map
    state_map = np.stack([x_indices, y_indices, z_indices], axis=-1)
    
    true_x = x_indices * xy_resolution + x_min
    true_y = y_indices * xy_resolution + y_min
    true_z = z_indices * (z_flat.max() - z_flat.min()) / num_z_bins + z_flat.min()
    
    # Create the inverse map, which contains the original (x, y, z) values
    inverse_map = np.stack([true_x, true_y, true_z], axis=-1)  # Storing the original values in the same shape
    
    return state_map, z_bins, inverse_map