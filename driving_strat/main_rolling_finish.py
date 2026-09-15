import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from scipy.interpolate import CubicSpline

from track.track_loader import load_track
from track.curvature import compute_curvature_splines
from optimization.ocp import solve_energy_ocp
from optimization.pulse_glide import find_energy_optimal_pulse_glide
from utils.plotting import plot_track

MAX_SPEED_MPS = 30.0 / 3.6


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


def save_lap_cheat_sheet(
    output_path,
    x,
    y,
    xl,
    yl,
    xr,
    yr,
    x_opt,
    y_opt,
    s_ocp,
    pulse_mask,
    v_kmh,
    speed_cap_kmh,
    title,
    speed_title,
    lap_time_s,
    start_speed_mps,
    end_speed_mps,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.22)

    pulse_legend = [
        Patch(facecolor=plt.cm.coolwarm(1.0), edgecolor='none', label='Pulse'),
        Patch(facecolor=plt.cm.coolwarm(0.0), edgecolor='none', label='Glide'),
    ]

    ax_track = fig.add_subplot(gs[0, 0])
    plot_track(x, y, ax=ax_track, show=False, line_radius_m=6.0, color='gray', alpha=0.25)
    ax_track.plot(xl, yl, 'k--', alpha=0.12, linewidth=1)
    ax_track.plot(xr, yr, 'k--', alpha=0.12, linewidth=1)
    ax_track.scatter(x_opt, y_opt, c=np.asarray(pulse_mask, dtype=float), cmap='coolwarm', s=12, zorder=5, vmin=0.0, vmax=1.0)
    ax_track.set_title(title)
    ax_track.set_aspect('equal')
    ax_track.set_xlabel('X')
    ax_track.set_ylabel('Y')
    ax_track.legend(handles=pulse_legend, loc='upper right', frameon=True)
    ax_track.text(
        0.03,
        0.03,
        f'Start: {start_speed_mps:.2f} m/s\nFinish: {end_speed_mps:.2f} m/s\nLap time: {lap_time_s:.1f} s',
        transform=ax_track.transAxes,
        ha='left',
        va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none'),
    )

    ax_speed = fig.add_subplot(gs[1, 0])
    ax_speed.plot(s_ocp, v_kmh, color='tab:blue', linewidth=2.2, label='Speed')
    ax_speed.plot(s_ocp, speed_cap_kmh, 'k--', linewidth=1.1, alpha=0.85, label='Speed cap')
    ax_speed.fill_between(s_ocp, v_kmh, alpha=0.25, color='tab:blue')
    ax_speed.set_title(speed_title)
    ax_speed.set_xlabel('Track Distance [m]')
    ax_speed.set_ylabel('Velocity [km/h]')
    ax_speed.grid(True, alpha=0.2)
    ax_speed.legend(loc='upper right', fontsize=8)
    ax_speed.text(
        0.03,
        0.86,
        'Use pulse in the red zones, glide in the blue zones.\nFollow the speed curve as the target.',
        transform=ax_speed.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none'),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / 'data' / 'sem_2025_eu.csv'

    s, x, y, xl, yl, xr, yr, z = load_track(str(data_path), radius=6.0)
    f_x, f_y, f_psi, f_kappa = compute_curvature_splines(s, x, y)
    f_elevation = CubicSpline(s, z, bc_type='natural')

    s_ocp = np.linspace(0, s[-1], 300)

    trajectory_status = "OCP converged"
    strategy = None
    n_opt = None
    target_v_iter = min(8.0, MAX_SPEED_MPS)
    num_coupling_iters = 2

    for iter_idx in range(num_coupling_iters):
        print(f"\nCoupled optimization iteration {iter_idx + 1}/{num_coupling_iters}")
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
            print("Loaded optimal trajectory from OCP solve.")
        except Exception:
            print("OCP did not converge; using best available debug trajectory.")
            n_opt = opti.debug.value(opt_vars['n'])
            trajectory_status = "OCP debug trajectory"

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

        target_v_iter = float(np.clip(np.mean(strategy['v']), 4.0, MAX_SPEED_MPS))

    v_opt = strategy['v']
    accel_opt = strategy['accel']
    pulse_mask = strategy['pulse_mask']
    speed_cap = strategy['speed_cap']
    lap1_terminal_speed = float(v_opt[-1])

    strategy_lap2 = find_energy_optimal_pulse_glide(
        s_ocp,
        f_kappa,
        f_elevation,
        start_speed=lap1_terminal_speed,
        end_speed=0.0,
        enforce_end_speed=False,
        max_lap_time_s=None,
        curvature_override=kappa_for_longitudinal,
    )

    v_lap2 = strategy_lap2['v']
    accel_lap2 = strategy_lap2['accel']
    lap2_terminal_speed = float(v_lap2[-1])

    strategy_lap3 = find_energy_optimal_pulse_glide(
        s_ocp,
        f_kappa,
        f_elevation,
        start_speed=lap2_terminal_speed,
        end_speed=0.0,
        enforce_end_speed=True,
        max_lap_time_s=None,
        curvature_override=kappa_for_longitudinal,
    )

    v_lap3 = strategy_lap3['v']
    accel_lap3 = strategy_lap3['accel']
    title_prefix = "Pulse-and-Glide Strategy (Rolling Finish)"

    print("\nPulse-and-Glide Results (Rolling Finish):")
    print(f"  Initial velocity: {v_opt[0]:.6f} m/s")
    print(f"  Final velocity: {v_opt[-1]:.6f} m/s")
    print(f"  Max velocity: {np.max(v_opt):.6f} m/s")
    print(f"  Min velocity: {np.min(v_opt):.6f} m/s")
    print(f"  Lap time: {strategy['lap_time_s']:.2f} s")
    print(f"  Propulsion energy: {strategy['energy_j'] / 1000.0:.2f} kJ")
    print(f"  Electrical energy: {strategy['electrical_energy_j'] / 1000.0:.2f} kJ")
    print(f"  Pulse share: {100.0 * np.mean(pulse_mask):.1f} %")
    print(f"  Start speed error: {strategy['start_speed_error']:.6f} m/s")
    print(f"  End speed constrained: {strategy.get('end_speed_enforced', True)}")
    print(f"  Feasible force profile: {strategy['feasible']}")
    if 'gear_profile' in strategy and 'shift_profile' in strategy:
        gear_profile = np.asarray(strategy['gear_profile'], dtype=int)
        shift_profile = np.asarray(strategy['shift_profile'], dtype=bool)
        shift_idx = np.where(shift_profile)[0]
        print(f"  Number of shifts: {len(shift_idx)}")
        if len(shift_idx) > 0:
            print("  Gear switches (distance -> gear):")
            for k in shift_idx:
                gear_before = int(gear_profile[k - 1] + 1) if k > 0 else int(gear_profile[k] + 1)
                gear_after = int(gear_profile[k] + 1)
                print(f"    s={s_ocp[k]:7.1f} m: G{gear_before} -> G{gear_after}")
    print("  Selected parameters:")
    print(f"    mu_tire: {strategy['params']['mu_tire']:.2f}")
    print(f"    pulse_accel: {strategy['params']['pulse_accel']:.2f} m/s²")
    print(f"    low_ratio: {strategy['params']['low_ratio']:.2f}")
    print(f"    high_ratio: {strategy['params']['high_ratio']:.2f}")
    print("\nAcceleration Stats:")
    print(f"  Max acceleration: {np.max(accel_opt):.3f} m/s²")
    print(f"  Min acceleration: {np.min(accel_opt):.3f} m/s²")

    print("\nSecond Lap Results (seeded by lap-1 terminal speed):")
    print(f"  Requested initial velocity: {lap1_terminal_speed:.6f} m/s")
    print(f"  Actual initial velocity: {v_lap2[0]:.6f} m/s")
    print(f"  Final velocity: {v_lap2[-1]:.6f} m/s")
    print(f"  Lap time: {strategy_lap2['lap_time_s']:.2f} s")
    print(f"  Electrical energy: {strategy_lap2['electrical_energy_j'] / 1000.0:.2f} kJ")

    print("\nThird Lap Results (seeded by lap-2 terminal speed):")
    print(f"  Requested initial velocity: {lap2_terminal_speed:.6f} m/s")
    print(f"  Actual initial velocity: {v_lap3[0]:.6f} m/s")
    print(f"  Final velocity: {v_lap3[-1]:.6f} m/s")
    print(f"  Lap time: {strategy_lap3['lap_time_s']:.2f} s")
    print(f"  Electrical energy: {strategy_lap3['electrical_energy_j'] / 1000.0:.2f} kJ")
    print(f"  End speed constrained: {strategy_lap3.get('end_speed_enforced', True)}")

    x_opt, y_opt = offset_path_from_n(s_ocp, n_opt, f_x, f_y, f_psi)

    accel_padded = np.append(accel_opt, accel_opt[-1])
    accel_lap2_padded = np.append(accel_lap2, accel_lap2[-1])
    accel_lap3_padded = np.append(accel_lap3, accel_lap3[-1])
    pulse_padded = np.append(pulse_mask, pulse_mask[-1]).astype(float)
    pulse_lap2_padded = np.append(strategy_lap2['pulse_mask'], strategy_lap2['pulse_mask'][-1]).astype(float)
    pulse_lap3_padded = np.append(strategy_lap3['pulse_mask'], strategy_lap3['pulse_mask'][-1]).astype(float)
    speed_cap_kmh = speed_cap * 3.6

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0:2, :])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[2, 1])
    ax4 = fig.add_subplot(gs[3, :])

    plot_track(x, y, ax=ax1, show=False, line_radius_m=6.0, color='gray', alpha=0.3)
    ax1.plot(xl, yl, 'k--', alpha=0.1)
    ax1.plot(xr, yr, 'k--', alpha=0.1)

    path = ax1.scatter(x_opt, y_opt, c=pulse_padded, cmap='coolwarm', s=10, zorder=5, vmin=0.0, vmax=1.0)
    ax1.set_title(f"{title_prefix} on {trajectory_status} - 150kg Setup")
    ax1.set_aspect('equal')

    ax2.plot(s_ocp, accel_padded, 'r', linewidth=2, label='Lap 1 acceleration')
    ax2.plot(s_ocp, accel_lap2_padded, color='tab:orange', linewidth=2, linestyle='--', label='Lap 2 acceleration')
    ax2.plot(s_ocp, accel_lap3_padded, color='tab:green', linewidth=2, linestyle='-.', label='Lap 3 acceleration')
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.fill_between(s_ocp, accel_padded, where=(accel_padded > 0), alpha=0.3, color='red', label='Accelerating')
    ax2.fill_between(s_ocp, accel_padded, where=(accel_padded <= 0), alpha=0.3, color='blue', label='Coasting deceleration')
    ax2.set_ylabel('Acceleration [m/s²]')
    ax2.set_xlabel('Track Distance [m]')
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper right')
    ax2.set_title('Acceleration Profile (dv/ds)')

    v_kmh = v_opt * 3.6
    v_lap2_kmh = v_lap2 * 3.6
    v_lap3_kmh = v_lap3 * 3.6
    ax3.plot(s_ocp, v_kmh, 'b', linewidth=2, label='Lap 1 velocity (rolling finish)')
    ax3.plot(s_ocp, v_lap2_kmh, color='tab:orange', linewidth=2, linestyle='--', label='Lap 2 velocity (seeded)')
    ax3.plot(s_ocp, v_lap3_kmh, color='tab:green', linewidth=2, linestyle='-.', label='Lap 3 velocity (stop at finish)')
    ax3.plot(s_ocp, speed_cap_kmh, 'k--', linewidth=1.2, alpha=0.8, label='Local Speed Cap')
    ax3.fill_between(s_ocp, v_kmh, alpha=0.3, color='blue')
    ax3.set_ylabel('Velocity [km/h]')
    ax3.set_xlabel('Track Distance [m]')
    ax3.grid(True, alpha=0.2)
    ax3.set_title('Velocity Profile')
    ax3.legend(loc='upper right')

    z_ocp = f_elevation(s_ocp) - f_elevation(0)
    ax4.plot(s_ocp, z_ocp, 'g', linewidth=2)
    ax4.fill_between(s_ocp, z_ocp, alpha=0.3, color='green')
    ax4.set_ylabel('Elevation [m]')
    ax4.set_xlabel('Track Distance [m]')
    ax4.grid(True, alpha=0.2)
    ax4.set_title('Elevation Profile')

    # Additional chart: second-lap track with pulse/glide overlay
    fig2, ax5 = plt.subplots(figsize=(8, 7))
    plot_track(x, y, ax=ax5, show=False, line_radius_m=6.0, color='gray', alpha=0.3)
    ax5.plot(xl, yl, 'k--', alpha=0.1)
    ax5.plot(xr, yr, 'k--', alpha=0.1)
    path_lap2 = ax5.scatter(x_opt, y_opt, c=pulse_lap2_padded, cmap='coolwarm', s=12, zorder=5, vmin=0.0, vmax=1.0)
    ax5.set_title('Second Lap Track (seeded by Lap 1 terminal speed)')
    ax5.set_aspect('equal')

    fig3, ax6 = plt.subplots(figsize=(8, 7))
    plot_track(x, y, ax=ax6, show=False, line_radius_m=6.0, color='gray', alpha=0.3)
    ax6.plot(xl, yl, 'k--', alpha=0.1)
    ax6.plot(xr, yr, 'k--', alpha=0.1)
    path_lap3 = ax6.scatter(x_opt, y_opt, c=pulse_lap3_padded, cmap='coolwarm', s=12, zorder=5, vmin=0.0, vmax=1.0)
    ax6.set_title('Third Lap Track (seeded by Lap 2 terminal speed, stop at finish)')
    ax6.set_aspect('equal')

    cheat_sheet_dir = script_dir.parent / 'pictures'
    save_lap_cheat_sheet(
        cheat_sheet_dir / 'driver_cheat_sheet_lap1.png',
        x,
        y,
        xl,
        yl,
        xr,
        yr,
        x_opt,
        y_opt,
        s_ocp,
        pulse_padded,
        v_kmh,
        speed_cap_kmh,
        'Driver Cheat Sheet - Lap 1',
        'Lap 1 Velocity Profile',
        strategy['lap_time_s'],
        0.0,
        lap1_terminal_speed,
    )
    save_lap_cheat_sheet(
        cheat_sheet_dir / 'driver_cheat_sheet_lap2.png',
        x,
        y,
        xl,
        yl,
        xr,
        yr,
        x_opt,
        y_opt,
        s_ocp,
        pulse_lap2_padded,
        v_lap2_kmh,
        speed_cap_kmh,
        'Driver Cheat Sheet - Lap 2',
        'Lap 2 Velocity Profile',
        strategy_lap2['lap_time_s'],
        lap1_terminal_speed,
        lap2_terminal_speed,
    )
    save_lap_cheat_sheet(
        cheat_sheet_dir / 'driver_cheat_sheet_lap3.png',
        x,
        y,
        xl,
        yl,
        xr,
        yr,
        x_opt,
        y_opt,
        s_ocp,
        pulse_lap3_padded,
        v_lap3_kmh,
        speed_cap_kmh,
        'Driver Cheat Sheet - Lap 3',
        'Lap 3 Velocity Profile',
        strategy_lap3['lap_time_s'],
        lap2_terminal_speed,
        float(v_lap3[-1]),
    )
    print(f"\nSaved driver cheat sheets to: {cheat_sheet_dir}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
