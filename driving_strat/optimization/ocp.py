import casadi as ca
import numpy as np
from model.car_model import RaceCar

from optimization.warmstart import generate_warmstart

def solve_energy_ocp(s_grid, f_kappa, f_elevation, track_radius=6.0, target_v=None):
    car = RaceCar()
    if target_v is None:
        target_v = car.avg_speed_target
    
    opti = ca.Opti()
    N = len(s_grid) - 1
    ds = s_grid[1] - s_grid[0]

    def max_drive_force_expr(v_val):
        vel = ca.fmax(v_val, 0.0)
        best = 0.0
        for ratio in car.gear_ratios:
            wheel_rpm = (vel / (2.0 * np.pi * car.wheel_radius)) * 60.0
            motor_rpm = wheel_rpm * ratio * car.final_drive_ratio

            torque_limited = (
                car.motor_max_torque * ratio * car.final_drive_ratio * car.drivetrain_efficiency
                / car.wheel_radius
            )
            power_limited = car.motor_max_power / ca.fmax(vel, 0.5)
            gear_force = ca.fmin(torque_limited, power_limited)
            gear_force = ca.fmin(gear_force, car.max_force)

            valid_force = ca.if_else(motor_rpm <= car.motor_max_rpm, gear_force, 0.0)
            best = ca.fmax(best, valid_force)
        return best

    # --- 1. Variables ---
    n = opti.variable(N+1)      
    alpha = opti.variable(N+1)  
    v = opti.variable(N+1)
    accel = opti.variable(N)     # Explicit acceleration variable
    F_wheels = opti.variable(N) 

    # --- 2. Objective Function (High Realism) ---
    # Minimize propulsion energy only. 
    # Gravity is handled by the physics constraints, not the objective.
    energy_cost = ca.sum1(ca.fmax(0, F_wheels) * ds)
    
    # Gravitational potential energy change (3D)
    # When going uphill: gravity does negative work
    elevation_vec = f_elevation(s_grid)
    dz = elevation_vec[1:] - elevation_vec[:-1]
    potential_energy_cost = car.mass * car.g * ca.sum1(ca.fmax(0, dz))  # Only cost when gaining elevation
    
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
    
    # Jerk constraint: Penalize rapid changes in acceleration for smooth transitions
    accel_delta = accel[1:] - accel[:-1]
    jerk_smoothness = 2.0 * ca.sum1(accel_delta**2)
    
    opti.minimize(energy_cost + speed_tracking + steering_smoothness + 
              path_smoothness + force_smoothness + jerk_smoothness - dist_reward)

    # --- 3. Initial Guess ---
    guess = generate_warmstart(s_grid, v_init=1.0)  # Start with lower guess velocity
    opti.set_initial(n, guess['n'])
    opti.set_initial(v, guess['v'])
    opti.set_initial(alpha, guess['alpha'])
    opti.set_initial(accel, 0.0)  # Start with zero acceleration
    opti.set_initial(F_wheels, guess['F'])

    # --- 4. Dynamics & Physics ---
    for k in range(N):
        # Calculate slope angle theta
        dz_ds = (f_elevation(s_grid[k+1]) - f_elevation(s_grid[k])) / ds
        theta = ca.atan(dz_ds)
        
        # Physics with Gravity
        F_drag = 0.5 * car.rho * car.CdA * v[k]**2
        # Rolling resistance depends on the Normal Force
        F_rr = car.mass * car.g * ca.cos(theta) * car.Crr
        F_gravity = car.mass * car.g * ca.sin(theta)
        
        # Link acceleration to forces through Newton's second law
        opti.subject_to(accel[k] == (F_wheels[k] - F_drag - F_rr - F_gravity) / car.mass)
        
        # Velocity dynamics
        opti.subject_to(v[k+1]**2 == v[k]**2 + 2 * accel[k] * ds)

        # Friction Ellipse: Grip is reduced on slopes because Normal Force is lower
        f_lat = car.mass * (v[k]**2 * ca.fabs(f_kappa(s_grid[k])))
        max_grip = (car.mass * 9.81 * ca.cos(theta)) * 0.7 
        drive_cap = max_drive_force_expr(v[k])
        opti.subject_to(F_wheels[k] <= drive_cap)
        opti.subject_to((F_wheels[k]/ca.fmax(drive_cap, 1.0))**2 + (f_lat/max_grip)**2 <= 1.0)

    # --- 5. Constraints ---
    
    # Realistic acceleration limits: prevent instant acceleration from 0 to max speed
    # Negative acceleration can still occur naturally from drag/rolling/slope.
    opti.subject_to(opti.bounded(-8.0, accel, 3.5))
    opti.subject_to(opti.bounded(0.0, F_wheels, car.max_force))
    opti.subject_to(opti.bounded(-track_radius, n, track_radius))
    opti.subject_to(opti.bounded(0.0, v, car.max_speed))  # Allow v=0 at start/end
    opti.subject_to(opti.bounded(-np.deg2rad(15), alpha, np.deg2rad(15))) # Tightened alpha

    # Boundary Conditions: Start and end velocity must be zero
    opti.subject_to(v[0] == 0.0)
    opti.subject_to(v[N] == 0.0)
    
    # Continuity for lateral and steering (closed loop)
    opti.subject_to(n[0] == n[N])
    opti.subject_to(alpha[0] == alpha[N])

    # --- 6. Solver Setup ---
    opts = {
        'ipopt.max_iter': 500,
        'ipopt.tol': 1e-2,  # Less strict tolerance
        'ipopt.hessian_approximation': 'limited-memory',
        'ipopt.acceptable_tol': 1e-1,  # Allow earlier convergence
        'ipopt.acceptable_iter': 5
    }
    opti.solver('ipopt', opts)

    variable_map = {'n': n, 'v': v, 'alpha': alpha, 'F': F_wheels, 'accel': accel}
    return opti, variable_map