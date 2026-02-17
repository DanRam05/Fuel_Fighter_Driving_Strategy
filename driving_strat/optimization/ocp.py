import casadi as ca
import numpy as np
from model.car_model import RaceCar
from optimization.warmstart import generate_warmstart

def solve_energy_ocp(s_grid, f_kappa, track_radius=6.0):
    car = RaceCar()
    car.mass = 80.0 
    
    opti = ca.Opti()
    N = len(s_grid) - 1
    ds = s_grid[1] - s_grid[0]

    # --- 1. Variables ---
    n = opti.variable(N+1)      
    alpha = opti.variable(N+1)  
    v = opti.variable(N+1)      
    F_wheels = opti.variable(N) 

    # --- 2. Objective Function (High Realism) ---
    target_v = 8.33  # Target 30 km/h in m/s
    
    energy_cost = ca.sum1(ca.fmax(0, F_wheels) * ds)
    
    # Relaxed speed tracking allows natural braking/acceleration in corners
    speed_tracking = 0.5 * ca.sum1((v - target_v)**2 * ds)
    
    # Distance reward: Guides the car to the apex
    kappa_vec = f_kappa(s_grid)
    dist_reward = 0.4 * ca.sum1(n * kappa_vec * ds) 
    
    # --- REALISM FIXES: Penalize rapid changes ---
    # Increased weight on d_alpha (steering rate) removes the 'wiggles'
    d_alpha = (alpha[1:] - alpha[:-1]) / ds
    steering_smoothness = 5.0 * ca.sum1(d_alpha**2 * ds)
    
    # Penalize second derivative of n (curvature of the path) for extra smoothness
    n_dot = (n[1:] - n[:-1]) / ds
    n_ddot = (n_dot[1:] - n_dot[:-1]) / ds
    path_smoothness = 10.0 * ca.sum1(n_ddot**2 * ds)

    F_delta = F_wheels[1:] - F_wheels[:-1]
    force_smoothness = 1e-1 * ca.sum1(F_delta**2)
    
    opti.minimize(energy_cost + speed_tracking + steering_smoothness + 
                  path_smoothness + force_smoothness - dist_reward)

    # --- 3. Initial Guess ---
    guess = generate_warmstart(s_grid, v_init=target_v)
    opti.set_initial(n, guess['n'])
    opti.set_initial(v, guess['v'])
    opti.set_initial(alpha, guess['alpha'])
    opti.set_initial(F_wheels, guess['F'])

    # --- 4. Dynamics & Physics ---
    for k in range(N):
        k_val = f_kappa(s_grid[k])
        opti.subject_to(n[k+1] == n[k] + (1 - n[k]*k_val) * ca.tan(alpha[k]) * ds)
        
        # Physics
        F_drag = 0.5 * car.rho * car.CdA * v[k]**2
        F_rr = car.mass * car.g * car.Crr
        accel = (F_wheels[k] - F_drag - F_rr) / car.mass
        opti.subject_to(v[k+1]**2 == v[k]**2 + 2 * accel * ds)

        # Friction Ellipse (Physics-based speed limit)
        f_lat = car.mass * (v[k]**2 * ca.fabs(k_val))
        max_grip = car.mass * 9.81 * 0.7 
        opti.subject_to((F_wheels[k]/car.max_force)**2 + (f_lat/max_grip)**2 <= 1.0)

    # --- 5. Constraints ---
    opti.subject_to(opti.bounded(-track_radius, n, track_radius))
    opti.subject_to(opti.bounded(car.min_speed, v, car.max_speed))
    opti.subject_to(opti.bounded(-np.deg2rad(15), alpha, np.deg2rad(15))) # Tightened alpha

    # Continuity
    opti.subject_to(n[0] == n[N])
    opti.subject_to(v[0] == v[N])
    opti.subject_to(alpha[0] == alpha[N])

    # --- 6. Solver Setup ---
    opts = {'ipopt.max_iter': 1500, 'ipopt.tol': 1e-3, 'ipopt.hessian_approximation': 'limited-memory'}
    opti.solver('ipopt', opts)

    variable_map = {'n': n, 'v': v, 'alpha': alpha, 'F': F_wheels}
    return opti, variable_map