import numpy as np

class RaceCar:
    """
    Defines the physical parameters and constraints of the vehicle.
    """
    # model/car_model.py

    def __init__(self):
        self.mass = 80.0
        self.CdA = 0.2
        self.Crr = 0.005
        self.rho = 1.225
        self.g = 9.81
        
        # --- Critical Values ---
        self.max_force = 250.0
        self.min_force = -350.0
        self.min_speed = 2.0        # Raised from 1.0
        self.max_speed = 15.0
        self.avg_speed_target = 8.0 # This MUST be higher than 1.0

    def get_drag_force(self, velocity):
        """Calculates aerodynamic drag at a given velocity."""
        return 0.5 * self.rho * self.CdA * velocity**2

    def get_rolling_resistance(self):
        """Calculates rolling resistance force."""
        return self.mass * self.g * self.Crr

    def get_efficiency(self, force, velocity):
        """
        Placeholder for motor/engine efficiency.
        Typically energy used = (Force * Velocity) / efficiency
        """
        # You can expand this with a lookup table or a curve later
        return 0.85