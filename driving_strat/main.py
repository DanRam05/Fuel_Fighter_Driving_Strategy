import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from track.track_loader import load_track
from track.curvature import compute_curvature_splines
from optimization.ocp import solve_energy_ocp
from utils.plotting import plot_track  # Ensure this is imported

def main():
    # --- 1. Robust Pathing ---
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / 'data' / 'sem_2025_eu.csv'

    # --- 2. Load Track Data ---
    s, x, y, xl, yl, xr, yr = load_track(str(data_path), radius=6.0)
    f_x, f_y, f_psi, f_kappa = compute_curvature_splines(s, x, y)
    s_ocp = np.linspace(0, s[-1], 300) 
    
    # --- 3. Run Optimization ---
    opti, opt_vars = solve_energy_ocp(s_ocp, f_kappa, track_radius=6.0)
    
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

    # --- 4. Transform to Cartesian ---
    x_opt, y_opt = [], []
    for si, ni in zip(s_ocp, n_opt):
        psi = float(f_psi(si))
        x_opt.append(float(f_x(si)) - ni * np.sin(psi))
        y_opt.append(float(f_y(si)) + ni * np.cos(psi))

    # --- 5. Visualization ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})

    # --- RESTORED: The Gray Track Boundaries ---
    # Use your original plot_track utility to draw the 6m wide track
    plot_track(x, y, ax=ax1, show=False, line_radius_m=6.0, color='gray', alpha=0.3)
    ax1.plot(xl, yl, 'k--', alpha=0.1) # Left boundary line
    ax1.plot(xr, yr, 'k--', alpha=0.1) # Right boundary line

    # Trajectory overlay
    v_min, v_max = np.min(v_opt), np.max(v_opt)
    if v_max - v_min < 0.05: v_max += 0.2; v_min -= 0.2
    
    # Force the colorbar to show a 10 km/h range for contrast
    v_kmh = v_opt * 3.6
    path = ax1.scatter(x_opt, y_opt, c=v_kmh, cmap='jet', s=10, zorder=5, vmin=20, vmax=35)
    plt.colorbar(path, ax=ax1, label='Velocity [m/s]')
    ax1.set_title(f"{title_prefix} - 80kg Setup")

    # Lateral Position Plot
    ax2.plot(s_ocp, n_opt, 'g', linewidth=2)
    ax2.axhline(6, color='r', linestyle='--', alpha=0.3)
    ax2.axhline(-6, color='r', linestyle='--', alpha=0.3)
    ax2.set_ylabel("Lateral Position n [m]")
    ax2.set_xlabel("Track Distance [s]")
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()