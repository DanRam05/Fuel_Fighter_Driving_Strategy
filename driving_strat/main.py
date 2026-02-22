import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline

from track.track_loader import load_track
from track.curvature import compute_curvature_splines
from optimization.ocp import solve_energy_ocp
from optimization.pulse_glide import find_energy_optimal_pulse_glide
from utils.plotting import plot_track  # Ensure this is imported

def main():
    # --- 1. Robust Pathing ---
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / 'data' / 'sem_2025_eu.csv'

    # --- 2. Load Track Data ---
    s, x, y, xl, yl, xr, yr, z = load_track(str(data_path), radius=6.0)
    f_x, f_y, f_psi, f_kappa = compute_curvature_splines(s, x, y)
    
    # Create elevation spline for 3D optimization
    f_elevation = CubicSpline(s, z, bc_type='natural')
    
    s_ocp = np.linspace(0, s[-1], 300) 
    
    # --- 3. Compute Optimal Trajectory (lateral path) ---
    opti, opt_vars = solve_energy_ocp(s_ocp, f_kappa, f_elevation, track_radius=6.0)
    trajectory_status = "OCP converged"
    try:
        sol = opti.solve()
        n_opt = sol.value(opt_vars['n'])
        print("Loaded optimal trajectory from OCP solve.")
    except Exception:
        print("OCP did not converge; using best available debug trajectory.")
        n_opt = opti.debug.value(opt_vars['n'])
        trajectory_status = "OCP debug trajectory"

    # --- 4. Run Energy-Optimal Pulse-and-Glide Strategy (longitudinal control) ---
    strategy = find_energy_optimal_pulse_glide(
        s_ocp,
        f_kappa,
        f_elevation,
        start_speed=0.0,
        end_speed=0.0,
    )
    v_opt = strategy['v']
    accel_opt = strategy['accel']
    pulse_mask = strategy['pulse_mask']
    speed_cap = strategy['speed_cap']
    title_prefix = "Pulse-and-Glide Strategy"
    
    # Print boundary conditions and acceleration stats
    print(f"\nPulse-and-Glide Results:")
    print(f"  Initial velocity: {v_opt[0]:.6f} m/s")
    print(f"  Final velocity: {v_opt[-1]:.6f} m/s")
    print(f"  Max velocity: {np.max(v_opt):.6f} m/s")
    print(f"  Min velocity: {np.min(v_opt):.6f} m/s")
    print(f"  Lap time: {strategy['lap_time_s']:.2f} s")
    print(f"  Propulsion energy: {strategy['energy_j'] / 1000.0:.2f} kJ")
    print(f"  Electrical energy: {strategy['electrical_energy_j'] / 1000.0:.2f} kJ")
    print(f"  Pulse share: {100.0 * np.mean(pulse_mask):.1f} %")
    print(f"  Start speed error: {strategy['start_speed_error']:.6f} m/s")
    print(f"  End speed error: {strategy['end_speed_error']:.6f} m/s")
    print(f"  Feasible force profile: {strategy['feasible']}")
    print("  Selected parameters:")
    print(f"    mu_tire: {strategy['params']['mu_tire']:.2f}")
    print(f"    pulse_accel: {strategy['params']['pulse_accel']:.2f} m/s²")
    print(f"    low_ratio: {strategy['params']['low_ratio']:.2f}")
    print(f"    high_ratio: {strategy['params']['high_ratio']:.2f}")
    print(f"\nAcceleration Stats:")
    print(f"  Max acceleration: {np.max(accel_opt):.3f} m/s²")
    print(f"  Max braking: {np.min(accel_opt):.3f} m/s²")

    # --- 5. Transform to Cartesian ---
    x_opt, y_opt = [], []
    for si, ni in zip(s_ocp, n_opt):
        psi = float(f_psi(si))
        x_opt.append(float(f_x(si)) - ni * np.sin(psi))
        y_opt.append(float(f_y(si)) + ni * np.cos(psi))
    
    # Pad acceleration array to match velocity length (accel is N, velocity is N+1)
    accel_padded = np.append(accel_opt, accel_opt[-1])
    pulse_padded = np.append(pulse_mask, pulse_mask[-1]).astype(float)
    speed_cap_kmh = speed_cap * 3.6

    # --- 6. Visualization ---
    fig = plt.figure(figsize=(14, 12))
    # Create a 4x2 grid to easily split the top half from the bottom
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

    # Top Half: Track with velocity (Rows 0 and 1, all columns)
    ax1 = fig.add_subplot(gs[0:2, :])  

    # Bottom Half: Distributed subplots
    ax2 = fig.add_subplot(gs[2, 0])  # Velocity profile (Row 2, Left)
    ax3 = fig.add_subplot(gs[2, 1])  # Force/Acceleration profile (Row 2, Right)
    ax4 = fig.add_subplot(gs[3, :])  # Elevation profile (Row 3, Full Width)

    # --- RESTORED: The Gray Track Boundaries ---
    # Use your original plot_track utility to draw the 6m wide track
    plot_track(x, y, ax=ax1, show=False, line_radius_m=6.0, color='gray', alpha=0.3)
    ax1.plot(xl, yl, 'k--', alpha=0.1) # Left boundary line
    ax1.plot(xr, yr, 'k--', alpha=0.1) # Right boundary line

    # Trajectory overlay colored by acceleration
    path = ax1.scatter(x_opt, y_opt, c=pulse_padded, cmap='coolwarm', s=10, zorder=5, vmin=0.0, vmax=1.0)
    cbar1 = plt.colorbar(path, ax=ax1)
    cbar1.set_label('Driving Mode (0 = Glide, 1 = Pulse)')
    ax1.set_title(f"{title_prefix} on {trajectory_status} - 80kg Setup")
    ax1.set_aspect('equal')

    # Acceleration Profile (left)
    ax2.plot(s_ocp, accel_padded, 'r', linewidth=2, label='Acceleration')
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.fill_between(s_ocp, accel_padded, where=(accel_padded > 0), alpha=0.3, color='red', label='Accelerating')
    ax2.fill_between(s_ocp, accel_padded, where=(accel_padded <= 0), alpha=0.3, color='blue', label='Braking')
    ax2.set_ylabel("Acceleration [m/s²]")
    ax2.set_xlabel("Track Distance [m]")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper right')
    ax2.set_title("Acceleration Profile (dv/ds)")
    
    # Velocity Profile (right)
    v_kmh = v_opt * 3.6
    ax3.plot(s_ocp, v_kmh, 'b', linewidth=2, label='Pulse-and-Glide')
    ax3.plot(s_ocp, speed_cap_kmh, 'k--', linewidth=1.2, alpha=0.8, label='Local Speed Cap')
    ax3.fill_between(s_ocp, v_kmh, alpha=0.3, color='blue')
    ax3.set_ylabel("Velocity [km/h]")
    ax3.set_xlabel("Track Distance [m]")
    ax3.grid(True, alpha=0.2)
    ax3.set_title("Velocity Profile")
    ax3.legend(loc='upper right')
    
    # Elevation Profile
    z_ocp = f_elevation(s_ocp) - f_elevation(0)  # Shift so first point is at 0 elevation
    ax4.plot(s_ocp, z_ocp, 'g', linewidth=2)
    ax4.fill_between(s_ocp, z_ocp, alpha=0.3, color='green')
    ax4.set_ylabel("Elevation [m]")
    ax4.set_xlabel("Track Distance [m]")
    ax4.grid(True, alpha=0.2)
    ax4.set_title("Elevation Profile")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()