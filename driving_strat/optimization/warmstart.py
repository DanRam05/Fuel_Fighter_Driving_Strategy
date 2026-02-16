import numpy as np

def generate_warmstart(s_grid, v_init=10.0):
    """
    Generates an initial guess for the OCP variables.
    
    Args:
        s_grid (array): The discretization points along the track.
        v_init (float): The starting constant velocity guess (m/s).
        
    Returns:
        dict: Initial values for states and controls.
    """
    N = len(s_grid) - 1
    
    # n (lateral deviation): Start on the centerline (0)
    n_guess = np.zeros(N + 1)
    
    # alpha (heading error): Start aligned with track (0)
    alpha_guess = np.zeros(N + 1)
    
    # v (velocity): Start at a constant speed to avoid 1/v singularities
    v_guess = np.full(N + 1, v_init)
    
    # F_wheels (Force): Small positive force to overcome baseline drag
    f_guess = np.full(N, 5.0) 
    
    return {
        'n': n_guess,
        'alpha': alpha_guess,
        'v': v_guess,
        'F': f_guess
    }