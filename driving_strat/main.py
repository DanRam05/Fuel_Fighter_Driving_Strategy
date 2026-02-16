import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Internal imports based on your directory structure
from track.track_loader import load_track
from track.curvature import compute_curvature_splines
from optimization.ocp import solve_energy_ocp
from utils.plotting import plot_track

def main():
    # --- 1. Setup Paths ---
    pkg_root = Path(__file__).resolve().parent
    data_path = pkg_root / 'data' / 'sem_2025_eu.csv'

    # --- 2. Load Raw Track Data ---
    # s: distance, x/y: centerline, xl/yl/xr/yr: boundaries
    s, x, y, xl, yl, xr, yr = load_track(str(data_path), radius=6.0)
    print(f"Track loaded: {s[-1]:.2f}m total length.")

    # --- 3. Build Smooth Geometry ---
    # These are CasADi functions needed by the OCP and for plotting
    f_x, f_y, f_psi, f_kappa = compute_curvature_splines(s, x, y)

    # --- 4. Run Optimization ---
    # We create a discretization grid (s_ocp) for the solver
    s_ocp = np.linspace(0, s[-1], 300) 
    
    try:
        # solve_energy_ocp must return (sol, vars_dict)
        sol, vars = solve_energy_ocp(s_ocp, f_kappa, track_radius=6.0)
        
        # Extract numerical results from the solver
        n_opt = sol.value(vars['n'])
        v_opt = sol.value(vars['v'])
        
        # --- 5. Transform Frenet (n, s) to Cartesian (x, y) ---
        # This converts lateral offsets into map coordinates
        x_opt = []
        y_opt = []
        
        for i in range(len(s_ocp)):
            si = s_ocp[i]
            ni = n_opt[i]
            
            # Get centerline position and heading at distance si
            cx = float(f_x(si))
            cy = float(f_y(si))
            psi = float(f_psi(si))
            
            # Rotate n-offset by the track heading to get global coordinates
            # Positive n is left, Negative n is right
            x_opt.append(cx - ni * np.sin(psi))
            y_opt.append(cy + ni * np.cos(psi))

        # --- 6. Visualization ---
        # Pass show=False to prevent the window from blocking execution
        ax = plot_track(x, y, show=False, line_radius_m=6.0, color='gray', alpha=0.3)
        
        # Plot the boundaries for clarity
        ax.plot(xl, yl, 'k--', alpha=0.2)
        ax.plot(xr, yr, 'k--', alpha=0.2)
        
        # Plot the optimized trajectory
        # Using a scatter plot colored by velocity (v) to see braking/acceleration zones
        scatter = ax.scatter(x_opt, y_opt, c=v_opt, cmap='jet', s=10, label='Speed (m/s)')
        ax.plot(x_opt, y_opt, 'r-', linewidth=1.5, alpha=0.7, label='Optimal Path')
        
        plt.colorbar(scatter, ax=ax, label='Velocity [m/s]')
        ax.legend()
        ax.set_title("Energy Efficient Trajectory")
        
        print("Optimization successful. Displaying plot...")
        plt.show()

    except Exception as e:
        print(f"An error occurred during optimization: {e}")

if __name__ == "__main__":
    main()