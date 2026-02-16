import casadi as ca
import numpy as np
from optimization.warmstart import generate_warmstart

def solve_energy_ocp(s_grid, f_kappa, track_radius=6.0):
    """
    Solves a spatial-domain OCP to find the minimum energy trajectory.
    """
    opti = ca.Opti()
    N = len(s_grid) - 1
    ds = s_grid[1] - s_grid[0]

    # --- 1. Variables ---
    # States
    n = opti.variable(N+1)      # Lateral deviation from centerline [m]
    alpha = opti.variable(N+1)  # Heading error relative to track [rad]
    v = opti.variable(N+1)      # Velocity [m/s]
    
    # Controls
    F_wheels = opti.variable(N) # Traction force [N]

    # --- 2. Parameters (Typically from model/race_car.py) ---
    m = 80.0      # Vehicle mass [kg]
    C_rr = 0.005   # Rolling resistance coefficient
    C_da = 0.2     # Drag area (Cd * A)
    rho = 1.225    # Air density [kg/m^3]
    g = 9.81       # Gravity [m/s^2]

    # --- 3. Warmstart (Initial Guess) ---
    # Prevents 'Maximum_Iterations_Exceeded' by giving a valid starting point
    guess = generate_warmstart(s_grid, v_init=8.0)
    opti.set_initial(n, guess['n'])
    opti.set_initial(alpha, guess['alpha'])
    opti.set_initial(v, guess['v'])
    opti.set_initial(F_wheels, guess['F'])

    # --- 4. Objective Function ---
    # Minimize energy: Integral of (Force * distance)
    # We use sum of squares for Force to penalize aggressive acceleration
    energy_loss = ca.sum1(F_wheels**2 * ds) 
    
    # Optional: Add a small time penalty to prevent the car from going too slow
    time_penalty = ca.sum1(ds / (v + 1e-3))
    
    opti.minimize(energy_loss + 0.1 * time_penalty)

    # --- 5. Dynamics (Spatial Transformation) ---
    for k in range(N):
        # Local track curvature
        kappa_k = f_kappa(s_grid[k])
        
        # Geometry: How n and alpha change over distance s
        # d_n/ds = (1 - n*kappa) * tan(alpha)
        opti.subject_to(n[k+1] == n[k] + (1 - n[k]*kappa_k) * ca.tan(alpha[k]) * ds)
        
        # Physics: Energy conservation (v_next^2 = v_curr^2 + 2*a*ds)
        # F_net = F_wheels - F_drag - F_rolling_resistance
        F_drag = 0.5 * rho * C_da * v[k]**2
        F_rr = m * g * C_rr
        acceleration = (F_wheels[k] - F_drag - F_rr) / m
        
        opti.subject_to(v[k+1]**2 == v[k]**2 + 2 * acceleration * ds)

    # --- 6. Constraints ---
    # Stay within track boundaries
    opti.subject_to(opti.bounded(-track_radius, n, track_radius))
    
    # Keep velocity within safe/physical limits
    opti.subject_to(opti.bounded(1.0, v, 25.0)) 
    
    # Force limits (Motor/Engine limits)
    opti.subject_to(opti.bounded(-200, F_wheels, 200))

    # Boundary Conditions (Lap consistency)
    # Start and end with the same state for a closed loop
    opti.subject_to(n[0] == n[N])
    opti.subject_to(v[0] == v[N])
    opti.subject_to(alpha[0] == alpha[N])

    # --- 7. Solver Setup ---
    opts = {
        'ipopt.max_iter': 2000,
        'ipopt.print_level': 5,
        'ipopt.tol': 1e-6,
        'ipopt.mu_strategy': 'adaptive'
    }
    opti.solver('ipopt', opts)

    # Solve and return variables
    sol = opti.solve()
    variables = {'n': n, 'v': v, 'alpha': alpha, 'F': F_wheels}
    
    return sol, variables