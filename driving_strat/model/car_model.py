import numpy as np

class RaceCar:
    """
    Defines the physical parameters and constraints of the vehicle.
    """
    # model/car_model.py

    def __init__(self):
        self.mass = 80.0 + 70.0 #Driver
        self.frontal_area = 0.7651  # m^2, frontal area of the vehicle
        self.Cd = 0.15  # Drag coefficient, typical for streamlined efficiency vehicles
        self.CdA = self.Cd * self.frontal_area  # Aerodynamic drag area (Cd * A)
        self.Crr = 0.005 # Typical for low rolling resistance tires
        self.rho = 1.225 
        self.g = 9.81
        
        # Critical Values 
        self.max_force = 250.0
        self.min_force = 0.0
        self.min_speed = 2.0 # Raised from 1.0
        self.max_speed = 15.0
        self.avg_speed_target = 9.0 # This MUST be higher than 1.0

        # Drivetrain / Gearbox 
        self.gear_ratios = np.array([3.6, 2.4, 1.6, 1.15], dtype=float) # Tuned for better acceleration and top speed balance, with more realistic spacing between gears, G1: 3.6, G2: 2.4, G3: 1.6, G4: 1.15
        self.final_drive_ratio = 2.8 # For increase torque at the wheels, tuned for better acceleration without exceeding motor limits
        self.wheel_radius = 0.26  # m
        self.drivetrain_efficiency = 0.95

        # Motor limits (kept consistent with previous ~250 N force scale)
        self.motor_max_torque = 6.8   # Nm
        self.motor_max_power = 2600.0 # W
        self.motor_max_rpm = 4200.0 # Increased from 4000 to allow better performance at higher speeds

        # Shift behavior (hysteresis + finite shift duration)
        self.upshift_rpm = 3600.0 # Increased upshift RPM to allow higher torque at low speeds without lugging
        self.downshift_rpm = 2200.0 # Increased downshift RPM to reduce lugging
        self.shift_time_s = 0.20 # Time taken to complete a gear shift (during which torque is reduced)
        self.shift_torque_cut = 0.50 # Reduce torque by 50% during shifts to simulate realistic power loss

    def get_drag_force(self, velocity):
        """Calculates aerodynamic drag at a given velocity."""
        return 0.5 * self.rho * self.CdA * velocity**2

    def get_rolling_resistance(self):
        """Calculates rolling resistance force."""
        return self.mass * self.g * self.Crr

    def speed_to_motor_rpm(self, velocity, gear_ratio):
        vel = np.asarray(velocity, dtype=float)
        wheel_rpm = (vel / (2.0 * np.pi * self.wheel_radius)) * 60.0
        return wheel_rpm * gear_ratio * self.final_drive_ratio

    def get_gear_force_limit(self, velocity, gear_index):
        vel = max(float(velocity), 0.0)
        ratio = float(self.gear_ratios[int(gear_index)])

        motor_rpm = self.speed_to_motor_rpm(vel, ratio)
        if motor_rpm > self.motor_max_rpm:
            return 0.0

        torque_limited = (
            self.motor_max_torque * ratio * self.final_drive_ratio * self.drivetrain_efficiency
            / self.wheel_radius
        )
        power_limited = self.motor_max_power / max(vel, 0.5)
        return float(min(torque_limited, power_limited, self.max_force))

    def select_gear(self, velocity):
        vel = max(float(velocity), 0.0)
        limits = [self.get_gear_force_limit(vel, idx) for idx in range(len(self.gear_ratios))]
        best_idx = int(np.argmax(limits))
        return best_idx

    def choose_gear_with_hysteresis(self, velocity, current_gear, shift_timer_s=0.0):
        gear = int(np.clip(current_gear, 0, len(self.gear_ratios) - 1))
        if shift_timer_s > 0.0:
            return gear, False

        rpm = float(self.speed_to_motor_rpm(max(float(velocity), 0.0), self.gear_ratios[gear]))
        new_gear = gear

        if rpm > self.upshift_rpm and gear < len(self.gear_ratios) - 1:
            new_gear = gear + 1
        elif rpm < self.downshift_rpm and gear > 0:
            new_gear = gear - 1

        return int(new_gear), (new_gear != gear)

    def get_max_drive_force(self, velocity):
        vel = np.asarray(velocity, dtype=float)
        if vel.ndim == 0:
            best_gear = self.select_gear(float(vel))
            return self.get_gear_force_limit(float(vel), best_gear)

        out = np.zeros_like(vel, dtype=float)
        for i, v_val in enumerate(vel):
            best_gear = self.select_gear(float(v_val))
            out[i] = self.get_gear_force_limit(float(v_val), best_gear)
        return out

    def get_efficiency(self, force, velocity, gear_index=None):
        """
        Gear-aware drivetrain efficiency estimate.
        """
        vel = max(float(velocity), 0.0)
        wheel_force = max(float(force), 0.0)
        if gear_index is None:
            gear_idx = self.select_gear(vel)
        else:
            gear_idx = int(np.clip(gear_index, 0, len(self.gear_ratios) - 1))
        gear_ratio = float(self.gear_ratios[gear_idx])

        rpm = float(self.speed_to_motor_rpm(vel, gear_ratio))
        rpm_norm = np.clip(rpm / self.motor_max_rpm, 0.0, 1.2)

        max_force = max(self.get_max_drive_force(vel), 1e-6)
        load = np.clip(wheel_force / max_force, 0.0, 1.2)

        # Peak around mid-load and mid-rpm; slight penalty in lower gears.
        gear_penalty = 0.008 * gear_idx
        eta = 0.90 - 0.10 * (load - 0.65) ** 2 - 0.08 * (rpm_norm - 0.60) ** 2 - gear_penalty
        return float(np.clip(eta, 0.75, 0.92))