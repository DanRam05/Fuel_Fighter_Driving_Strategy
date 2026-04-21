import numpy as np

from model.car_model import RaceCar


def _flatten(values):
    return np.asarray(values, dtype=float).reshape(-1)


def generate_pulse_and_glide_profile(
    s_grid,
    f_kappa,
    f_elevation,
    car=None,
    mu_tire=0.75,
    pulse_accel=1.3,
    low_ratio=0.82,
    high_ratio=0.96,
    start_speed=0.0,
    end_speed=0.0,
    enforce_end_speed=True,
    curvature_override=None,
):
    """
    Simulate a pulse-and-glide strategy along track distance with boundary speeds.

    The strategy alternates between:
    - Pulse: positive drive force until a local speed ceiling is reached.
    - Glide: zero drive force, letting aero/rolling/slope decelerate the car.

    Args:
        s_grid (np.ndarray): Monotonic distance coordinates [m].
        f_kappa (callable): Curvature interpolant kappa(s) [1/m].
        f_elevation (callable): Elevation interpolant z(s) [m].
        car (RaceCar | None): Vehicle model instance.
        mu_tire (float): Longitudinal/lateral friction scaling.
        pulse_accel (float): Desired pulse acceleration [m/s^2].
        low_ratio (float): Lower threshold ratio of local speed cap.
        high_ratio (float): Upper threshold ratio of local speed cap.
        start_speed (float): Boundary speed at start [m/s].
        end_speed (float): Boundary speed at finish [m/s].
        enforce_end_speed (bool): If True, enforce end_speed with a backward
            glide envelope. If False, leave terminal speed unconstrained.
        curvature_override (np.ndarray | None): Optional absolute curvature profile
            aligned with s_grid. When provided, this overrides f_kappa(s_grid).

    Returns:
        dict: Strategy results and feasibility metadata.
    """
    if car is None:
        car = RaceCar()

    s_grid = np.asarray(s_grid, dtype=float)
    n_points = len(s_grid)
    if n_points < 2:
        raise ValueError("s_grid must contain at least 2 points")

    ds = float(np.mean(np.diff(s_grid)))
    if curvature_override is None:
        curvature = np.abs(_flatten(f_kappa(s_grid)))
    else:
        curvature = np.abs(np.asarray(curvature_override, dtype=float).reshape(-1))
        if len(curvature) != n_points:
            raise ValueError("curvature_override must have same length as s_grid")
    elevation = _flatten(f_elevation(s_grid))

    kappa_eps = 1e-6
    v_curve_cap = np.sqrt(np.maximum(mu_tire * car.g / np.maximum(curvature, kappa_eps), 0.0))
    v_local_cap = np.clip(v_curve_cap, 0.0, car.max_speed)
    
    # Smooth v_local_cap backward to prevent drastic deceleration spikes:
    # each point can't exceed its neighbors' average (acts as low-pass filter)
    for _ in range(3):  # Apply smoothing 3 times for gentler effect
        v_local_cap_smooth = np.copy(v_local_cap)
        for k in range(1, n_points - 1):
            v_local_cap_smooth[k] = min(v_local_cap[k], 0.5 * (v_local_cap[k-1] + v_local_cap[k+1]))
        v_local_cap = v_local_cap_smooth

    v = np.zeros(n_points)
    v_forward = np.zeros(n_points)
    accel = np.zeros(n_points - 1)
    force = np.zeros(n_points - 1)
    pulse_mask = np.zeros(n_points - 1, dtype=bool)
    gear_profile = np.zeros(n_points - 1, dtype=int)
    shift_profile = np.zeros(n_points - 1, dtype=bool)

    v_forward[0] = max(0.0, min(start_speed, v_local_cap[0]))
    pulse_on = True
    current_gear = car.select_gear(v_forward[0])
    shift_timer_s = 0.0

    for k in range(n_points - 1):
        dt_k = ds / max(v_forward[k], 0.5)
        requested_gear, shifted = car.choose_gear_with_hysteresis(
            v_forward[k],
            current_gear,
            shift_timer_s=shift_timer_s,
        )
        if shifted:
            current_gear = requested_gear
            shift_timer_s = car.shift_time_s
            shift_profile[k] = True
        elif shift_timer_s > 0.0:
            shift_timer_s = max(0.0, shift_timer_s - dt_k)

        dz_ds = (elevation[k + 1] - elevation[k]) / ds
        theta = np.arctan(dz_ds)

        v_cap = max(0.0, min(v_local_cap[k], v_local_cap[k + 1]))
        v_low = low_ratio * v_cap
        v_high = high_ratio * v_cap

        if v_forward[k] <= v_low:
            pulse_on = True
        elif v_forward[k] >= v_high:
            pulse_on = False

        drag = 0.5 * car.rho * car.CdA * v_forward[k] ** 2
        rr = car.mass * car.g * car.Crr * np.cos(theta)
        gravity = car.mass * car.g * np.sin(theta)
        resistive = drag + rr + gravity

        lat_force = car.mass * v_forward[k] ** 2 * curvature[k]
        normal_force = car.mass * car.g * np.cos(theta)
        max_total_tire = max(mu_tire * normal_force, 1.0)
        max_longitudinal = np.sqrt(max(max_total_tire**2 - lat_force**2, 0.0))
        max_drive_force_powertrain = car.get_gear_force_limit(v_forward[k], current_gear)
        if shift_timer_s > 0.0:
            max_drive_force_powertrain *= (1.0 - car.shift_torque_cut)
        max_drive_force = min(max_drive_force_powertrain, max_longitudinal)
        gear_profile[k] = current_gear

        if pulse_on:
            desired_force = resistive + car.mass * pulse_accel
            wheel_force = min(max(desired_force, 0.0), max_drive_force)
        else:
            wheel_force = 0.0

        a_k = (wheel_force - resistive) / car.mass
        v_sq_next = max(v_forward[k] ** 2 + 2.0 * a_k * ds, 0.0)
        v_next = np.sqrt(v_sq_next)

        if v_next > v_local_cap[k + 1]:
            v_next = v_local_cap[k + 1]
            a_k = (v_next**2 - v_forward[k] ** 2) / (2.0 * ds)
            wheel_force = max(0.0, car.mass * a_k + resistive)

        v_forward[k + 1] = v_next

    if enforce_end_speed:
        # Backward glide envelope: maximum speed that can still reach end_speed
        # using only passive deceleration (wheel_force = 0).
        aero_coeff = 0.5 * car.rho * car.CdA
        v_glide = np.zeros(n_points)
        v_glide[-1] = max(0.0, min(end_speed, v_local_cap[-1]))
        for k in range(n_points - 2, -1, -1):
            dz_ds = (elevation[k + 1] - elevation[k]) / ds
            theta = np.arctan(dz_ds)

            rr = car.mass * car.g * car.Crr * np.cos(theta)
            gravity = car.mass * car.g * np.sin(theta)
            static_resistive = rr + gravity

            denom = 1.0 - (2.0 * ds * aero_coeff / car.mass)
            denom = max(denom, 1e-6)
            v_prev_sq = (v_glide[k + 1] ** 2 + (2.0 * ds * static_resistive / car.mass)) / denom
            v_glide[k] = np.sqrt(max(v_prev_sq, 0.0))

        v = np.minimum(v_forward, v_glide)
        v = np.minimum(v, v_local_cap)
    else:
        v = np.minimum(v_forward, v_local_cap)

    v[0] = max(0.0, min(start_speed, v_local_cap[0]))
    if enforce_end_speed:
        v[-1] = max(0.0, min(end_speed, v_local_cap[-1]))
    
    # Smooth final velocity to eliminate drastic deceleration spikes
    # Backward-looking constraint: limit speed drop between segments
    for _ in range(5):  # Apply multiple smoothing passes
        v_smooth = np.copy(v)
        ds_val = np.mean(np.diff(s_grid))
        max_decel_m_s2 = -2.5  # Limit deceleration to -2.5 m/s²
        for k in range(n_points - 2, 0, -1):
            dv_max = max_decel_m_s2 * (ds_val / 10.0)  # Decel budget per segment
            v_smooth[k] = min(v[k], v[k+1] - dv_max)
        v = v_smooth

    feasible = True
    for k in range(n_points - 1):
        dz_ds = (elevation[k + 1] - elevation[k]) / ds
        theta = np.arctan(dz_ds)
        a_k = (v[k + 1] ** 2 - v[k] ** 2) / (2.0 * ds)

        drag = 0.5 * car.rho * car.CdA * v[k] ** 2
        rr = car.mass * car.g * car.Crr * np.cos(theta)
        gravity = car.mass * car.g * np.sin(theta)
        resistive = drag + rr + gravity

        lat_force = car.mass * v[k] ** 2 * curvature[k]
        normal_force = car.mass * car.g * np.cos(theta)
        max_total_tire = max(mu_tire * normal_force, 1.0)
        max_longitudinal = np.sqrt(max(max_total_tire**2 - lat_force**2, 0.0))
        max_drive_force = min(car.get_gear_force_limit(v[k], gear_profile[k]), max_longitudinal)

        required_force = car.mass * a_k + resistive
        if required_force > max_drive_force + 1e-6 or required_force < -1e-4:
            feasible = False

        force[k] = np.clip(required_force, car.min_force, max_drive_force)
        accel[k] = a_k
        pulse_mask[k] = force[k] > 1e-3

    v_mid = 0.5 * (v[:-1] + v[1:])
    lap_time_s = float(np.sum(ds / np.maximum(v_mid, 0.2)))

    propulsion_energy = float(np.sum(np.maximum(force, 0.0) * ds))
    elec_energy_segments = np.zeros_like(force)
    for k in range(n_points - 1):
        if force[k] <= 0.0:
            continue
        eta_k = car.get_efficiency(force[k], v_mid[k], gear_index=gear_profile[k])
        elec_energy_segments[k] = (force[k] * ds) / max(eta_k, 1e-3)
    electrical_energy = float(np.sum(elec_energy_segments))

    return {
        "v": v,
        "accel": accel,
        "force": force,
        "pulse_mask": pulse_mask,
        "speed_cap": v_local_cap,
        "lap_time_s": lap_time_s,
        "energy_j": propulsion_energy,
        "electrical_energy_j": electrical_energy,
        "feasible": feasible,
        "start_speed_error": float(abs(v[0] - start_speed)),
        "end_speed_error": float(abs(v[-1] - end_speed)) if enforce_end_speed else 0.0,
        "end_speed_enforced": bool(enforce_end_speed),
        "params": {
            "mu_tire": mu_tire,
            "pulse_accel": pulse_accel,
            "low_ratio": low_ratio,
            "high_ratio": high_ratio,
        },
        "gear_profile": gear_profile,
        "shift_profile": shift_profile,
        "num_shifts": int(np.sum(shift_profile)),
    }


def find_energy_optimal_pulse_glide(
    s_grid,
    f_kappa,
    f_elevation,
    car=None,
    mu_tire_values=(0.68, 0.72, 0.76),
    pulse_accel_values=(0.8, 1.0, 1.2, 1.4),
    low_ratio_values=(0.65, 0.72, 0.80),
    high_ratio_values=(0.88, 0.93, 0.97),
    start_speed=0.0,
    end_speed=0.0,
    enforce_end_speed=True,
    max_lap_time_s=None,
    curvature_override=None,
):
    """Search pulse-and-glide parameters for minimum electrical energy."""
    if car is None:
        car = RaceCar()

    best_result = None
    for mu_tire in mu_tire_values:
        for pulse_accel in pulse_accel_values:
            for low_ratio in low_ratio_values:
                for high_ratio in high_ratio_values:
                    if low_ratio >= high_ratio:
                        continue

                    candidate = generate_pulse_and_glide_profile(
                        s_grid=s_grid,
                        f_kappa=f_kappa,
                        f_elevation=f_elevation,
                        car=car,
                        mu_tire=mu_tire,
                        pulse_accel=pulse_accel,
                        low_ratio=low_ratio,
                        high_ratio=high_ratio,
                        start_speed=start_speed,
                        end_speed=end_speed,
                        enforce_end_speed=enforce_end_speed,
                        curvature_override=curvature_override,
                    )

                    if not candidate["feasible"]:
                        continue
                    if max_lap_time_s is not None and candidate["lap_time_s"] > max_lap_time_s:
                        continue

                    if best_result is None or candidate["electrical_energy_j"] < best_result["electrical_energy_j"]:
                        best_result = candidate

    if best_result is None:
        best_result = generate_pulse_and_glide_profile(
            s_grid=s_grid,
            f_kappa=f_kappa,
            f_elevation=f_elevation,
            car=car,
            mu_tire=0.72,
            pulse_accel=1.0,
            low_ratio=0.72,
            high_ratio=0.93,
            start_speed=start_speed,
            end_speed=end_speed,
            enforce_end_speed=enforce_end_speed,
            curvature_override=curvature_override,
        )

    return best_result
