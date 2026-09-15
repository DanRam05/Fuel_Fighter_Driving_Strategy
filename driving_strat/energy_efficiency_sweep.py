import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline

from model.car_model import RaceCar
from track.track_loader import load_track
from track.curvature import compute_curvature_splines
from optimization.ocp import solve_energy_ocp
from optimization.pulse_glide import find_energy_optimal_pulse_glide
from utils.plotting import plot_track


def compute_path_curvature(s_vals, x_vals, y_vals):
    s_vals = np.asarray(s_vals, dtype=float)
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    dx_ds = np.gradient(x_vals, s_vals)
    dy_ds = np.gradient(y_vals, s_vals)
    d2x_ds2 = np.gradient(dx_ds, s_vals)
    d2y_ds2 = np.gradient(dy_ds, s_vals)

    denom = np.power(dx_ds**2 + dy_ds**2, 1.5)
    denom = np.maximum(denom, 1e-9)
    kappa = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / denom
    return kappa


def smooth_and_bound_curvature(kappa_opt, kappa_center, smoothing_window=9, max_gain=1.25):
    kappa_opt = np.asarray(kappa_opt, dtype=float)
    kappa_center = np.asarray(kappa_center, dtype=float)

    win = max(3, int(smoothing_window))
    if win % 2 == 0:
        win += 1

    kernel = np.ones(win, dtype=float) / win
    kappa_opt_abs = np.abs(kappa_opt)
    kappa_smooth = np.convolve(kappa_opt_abs, kernel, mode='same')

    kappa_floor = np.maximum(np.abs(kappa_center), 1e-5)
    kappa_cap = max_gain * kappa_floor + 0.01
    return np.clip(kappa_smooth, kappa_floor, kappa_cap)


def offset_path_from_n(s_vals, n_vals, f_x, f_y, f_psi):
    x_opt, y_opt = [], []
    for si, ni in zip(s_vals, n_vals):
        psi = float(f_psi(si))
        x_opt.append(float(f_x(si)) - ni * np.sin(psi))
        y_opt.append(float(f_y(si)) + ni * np.cos(psi))
    return np.asarray(x_opt), np.asarray(y_opt)


def run_optimization_for_max_speed(s_ocp, f_kappa, f_elevation, f_x, f_y, f_psi, max_speed_mps):
    """Run a complete optimization for a given max speed and return energy metrics."""
    
    # Solve OCP
    target_v_iter = min(8.0, max_speed_mps)
    num_coupling_iters = 2
    
    for iter_idx in range(num_coupling_iters):
        opti, opt_vars = solve_energy_ocp(
            s_ocp,
            f_kappa,
            f_elevation,
            track_radius=6.0,
            target_v=target_v_iter,
        )
        
        try:
            sol = opti.solve()
            n_opt = sol.value(opt_vars['n'])
        except Exception:
            n_opt = opti.debug.value(opt_vars['n'])
        
        x_opt, y_opt = offset_path_from_n(s_ocp, n_opt, f_x, f_y, f_psi)
        kappa_opt = compute_path_curvature(s_ocp, x_opt, y_opt)
        kappa_center = np.asarray(f_kappa(s_ocp), dtype=float).reshape(-1)
        kappa_for_longitudinal = smooth_and_bound_curvature(
            kappa_opt,
            kappa_center,
            smoothing_window=9,
            max_gain=1.25,
        )
        
        strategy = find_energy_optimal_pulse_glide(
            s_ocp,
            f_kappa,
            f_elevation,
            start_speed=0.0,
            end_speed=0.0,
            enforce_end_speed=False,
            max_lap_time_s=None,
            curvature_override=kappa_for_longitudinal,
        )
        
        target_v_iter = float(np.clip(np.mean(strategy['v']), 4.0, max_speed_mps))
    
    # Extract metrics
    v_opt = strategy['v']
    energy = strategy['electrical_energy_j']
    lap_time = strategy['lap_time_s']
    avg_speed = np.mean(v_opt[v_opt > 0.1])  # Average non-zero speed
    max_achieved_speed = np.max(v_opt)
    pulse_share = np.mean(strategy['pulse_mask'])
    
    return {
        'energy_kj': energy / 1000.0,
        'lap_time_s': lap_time,
        'avg_speed_mps': avg_speed,
        'max_speed_mps': max_achieved_speed,
        'pulse_share': pulse_share,
        'energy_per_meter': energy / s_ocp[-1],
        'energy_per_second': energy / lap_time,
    }


def main():
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / 'data' / 'sem_2025_eu.csv'
    car = RaceCar()

    s, x, y, xl, yl, xr, yr, z = load_track(str(data_path), radius=6.0)
    f_x, f_y, f_psi, f_kappa = compute_curvature_splines(s, x, y)
    f_elevation = CubicSpline(s, z, bc_type='natural')

    s_ocp = np.linspace(0, s[-1], 300)

    # Test different max speeds (km/h)
    speeds_kmh = np.arange(15, int(round(car.max_speed * 3.6)) + 1, 5)
    speeds_mps = speeds_kmh / 3.6
    
    results = []
    
    print("Running energy efficiency sweep across different max speeds...")
    print(f"Global speed cap from car model: {car.max_speed * 3.6:.0f} km/h")
    print("=" * 80)
    
    for speed_kmh, speed_mps in zip(speeds_kmh, speeds_mps):
        print(f"\nTesting max speed: {speed_kmh} km/h ({speed_mps:.2f} m/s)")
        
        try:
            metrics = run_optimization_for_max_speed(
                s_ocp, f_kappa, f_elevation, f_x, f_y, f_psi, speed_mps
            )
            metrics['max_speed_kmh'] = speed_kmh
            results.append(metrics)
            
            print(f"  Energy: {metrics['energy_kj']:.2f} kJ")
            print(f"  Energy per meter: {metrics['energy_per_meter']:.4f} J/m")
            print(f"  Energy per second: {metrics['energy_per_second']:.2f} W (avg)")
            print(f"  Lap time: {metrics['lap_time_s']:.2f} s")
            print(f"  Average speed: {metrics['avg_speed_mps']:.2f} m/s ({metrics['avg_speed_mps']*3.6:.1f} km/h)")
            print(f"  Max achieved speed: {metrics['max_speed_mps']:.2f} m/s ({metrics['max_speed_mps']*3.6:.1f} km/h)")
            print(f"  Pulse share: {100*metrics['pulse_share']:.1f}%")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Max Speed':<12} {'Energy':<12} {'J/m':<12} {'W (avg)':<12} {'Lap Time':<12} {'Avg Speed':<12}")
    print(f"{'(km/h)':<12} {'(kJ)':<12} {'(J/m)':<12} {'(W)':<12} {'(s)':<12} {'(km/h)':<12}")
    print("-" * 80)
    
    for res in results:
        print(f"{res['max_speed_kmh']:<12.0f} {res['energy_kj']:<12.2f} {res['energy_per_meter']:<12.4f} "
              f"{res['energy_per_second']:<12.2f} {res['lap_time_s']:<12.2f} {res['avg_speed_mps']*3.6:<12.1f}")
    
    # Find optimal speeds
    if results:
        min_energy_idx = np.argmin([r['energy_kj'] for r in results])
        min_per_meter_idx = np.argmin([r['energy_per_meter'] for r in results])
        min_per_second_idx = np.argmin([r['energy_per_second'] for r in results])

        best_total_energy = results[min_energy_idx]
        best_distance_efficiency = results[min_per_meter_idx]
        best_power_efficiency = results[min_per_second_idx]
        
        print("\n" + "=" * 80)
        print("OPTIMAL SPEEDS")
        print("=" * 80)
        print(f"Minimum total energy: {best_total_energy['max_speed_kmh']:.0f} km/h "
              f"({best_total_energy['energy_kj']:.2f} kJ)")
        print(f"Most efficient (J/m): {best_distance_efficiency['max_speed_kmh']:.0f} km/h "
              f"({best_distance_efficiency['energy_per_meter']:.4f} J/m)")
        print(f"Most efficient (W avg): {best_power_efficiency['max_speed_kmh']:.0f} km/h "
              f"({best_power_efficiency['energy_per_second']:.2f} W)")
        print(f"\nBest sweep speed for energy per distance: {best_distance_efficiency['max_speed_kmh']:.0f} km/h")
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        speeds_plot = [r['max_speed_kmh'] for r in results]
        
        # Total energy
        axes[0, 0].plot(speeds_plot, [r['energy_kj'] for r in results], 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Max Speed (km/h)')
        axes[0, 0].set_ylabel('Total Energy (kJ)')
        axes[0, 0].set_title('Total Energy vs Max Speed')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(results[min_energy_idx]['max_speed_kmh'], color='r', linestyle='--', alpha=0.5)
        
        # Energy per meter (distance efficiency)
        axes[0, 1].plot(speeds_plot, [r['energy_per_meter'] for r in results], 'o-', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_xlabel('Max Speed (km/h)')
        axes[0, 1].set_ylabel('Energy per Meter (J/m)')
        axes[0, 1].set_title('Distance Efficiency vs Max Speed')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(results[min_per_meter_idx]['max_speed_kmh'], color='r', linestyle='--', alpha=0.5)
        
        # Energy per second (time efficiency)
        axes[1, 0].plot(speeds_plot, [r['energy_per_second'] for r in results], 'o-', linewidth=2, markersize=8, color='purple')
        axes[1, 0].set_xlabel('Max Speed (km/h)')
        axes[1, 0].set_ylabel('Average Power (W)')
        axes[1, 0].set_title('Power Consumption vs Max Speed')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(results[min_per_second_idx]['max_speed_kmh'], color='r', linestyle='--', alpha=0.5)
        
        # Lap time
        axes[1, 1].plot(speeds_plot, [r['lap_time_s'] for r in results], 'o-', linewidth=2, markersize=8, color='orange')
        axes[1, 1].set_xlabel('Max Speed (km/h)')
        axes[1, 1].set_ylabel('Lap Time (s)')
        axes[1, 1].set_title('Lap Time vs Max Speed')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = script_dir / 'efficiency_sweep.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        plt.close()


if __name__ == '__main__':
    main()
