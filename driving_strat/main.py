import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline

from track.track_loader import load_track
from track.curvature import compute_curvature_splines
from optimization.ocp import solve_energy_ocp
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
    
    # --- 3. Run Optimization with 3D elevation ---
    opti, opt_vars = solve_energy_ocp(s_ocp, f_kappa, f_elevation, track_radius=6.0)
    
    try:
        sol = opti.solve()
        n_opt = sol.value(opt_vars['n'])
        v_opt = sol.value(opt_vars['v'])
        title_prefix = "Optimal Trajectory"
    except:
        print("Using debug values due to solver timeout...")
        n_opt = opti.debug.value(opt_vars['n'])
        v_opt = opti.debug.value(opt_vars['v'])
        title_prefix = "Debug Path (Non-Converged)"
    
    # Print boundary conditions
    print(f"\nBoundary Conditions:")
    print(f"  Initial velocity: {v_opt[0]:.6f} m/s")
    print(f"  Final velocity: {v_opt[-1]:.6f} m/s")
    print(f"  Max velocity: {np.max(v_opt):.6f} m/s")
    print(f"  Min velocity: {np.min(v_opt):.6f} m/s")

    # --- 4. Transform to Cartesian ---
    x_opt, y_opt = [], []
    for si, ni in zip(s_ocp, n_opt):
        psi = float(f_psi(si))
        x_opt.append(float(f_x(si)) - ni * np.sin(psi))
        y_opt.append(float(f_y(si)) + ni * np.cos(psi))
    
    # --- Calculate Acceleration (dv/ds) ---
    dv_ds = np.gradient(v_opt, s_ocp)

    # --- 5. Visualization ---
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
    path = ax1.scatter(x_opt, y_opt, c=dv_ds, cmap='RdBu_r', s=10, zorder=5)
    cbar1 = plt.colorbar(path, ax=ax1, label='Acceleration [m/s²]')
    ax1.set_title(f"{title_prefix} - 80kg Setup (3D with Elevation)")
    ax1.set_aspect('equal')

    # Acceleration Profile (left)
    ax2.plot(s_ocp, dv_ds, 'r', linewidth=2)
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.fill_between(s_ocp, dv_ds, where=(dv_ds > 0), alpha=0.3, color='red', label='Accelerating')
    ax2.fill_between(s_ocp, dv_ds, where=(dv_ds <= 0), alpha=0.3, color='blue', label='Braking')
    ax2.set_ylabel("Acceleration [m/s²]")
    ax2.set_xlabel("Track Distance [m]")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper right')
    ax2.set_title("Acceleration Profile (dv/ds)")
    
    # Velocity Profile (right)
    v_kmh = v_opt * 3.6
    ax3.plot(s_ocp, v_kmh, 'b', linewidth=2)
    ax3.fill_between(s_ocp, v_kmh, alpha=0.3, color='blue')
    ax3.set_ylabel("Velocity [km/h]")
    ax3.set_xlabel("Track Distance [m]")
    ax3.grid(True, alpha=0.2)
    ax3.set_title("Velocity Profile")
    
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