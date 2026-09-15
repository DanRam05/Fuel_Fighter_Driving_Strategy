"""
GPS and RPM Data Logger
Handles real-time data collection from GPS module and vehicle RPM/CAN bus.
"""

import threading
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class DataSource(ABC):
    """Abstract base class for data sources (GPS, RPM, etc.)"""
    
    @abstractmethod
    def connect(self):
        """Connect to data source."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from data source."""
        pass
    
    @abstractmethod
    def read(self) -> dict:
        """Read latest data point."""
        pass


class GPSModule(DataSource):
    """
    GPS/GNSS module interface.
    Reads latitude, longitude, and speed from a GPS receiver.
    
    Placeholder implementation - replace with actual hardware interface:
    - Serial port communication (pyserial)
    - NTRIP for real-time kinematic GPS
    - Integration with specific GPS hardware (u-blox, Garmin, etc.)
    """
    
    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.connected = False
        self.last_data = None
    
    def connect(self):
        """Connect to GPS module."""
        # TODO: Implement actual serial connection
        # import serial
        # self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
        self.connected = True
        print(f"GPS connected on {self.port}")
    
    def disconnect(self):
        """Disconnect from GPS module."""
        # if self.serial:
        #     self.serial.close()
        self.connected = False
    
    def read(self) -> Optional[Tuple[float, float, float]]:
        """
        Read GPS data.
        
        Returns:
            (latitude, longitude, speed_mps) or None if no data
        """
        if not self.connected:
            return None
        
        # TODO: Parse NMEA or proprietary GPS format
        # For now, return mock data
        # Example: (59.6552, 10.7748, 15.0)  # Oslo coordinates, 15 m/s
        return self.last_data
    
    def update_mock_data(self, lat: float, lon: float, speed: float):
        """For testing: set mock GPS data."""
        self.last_data = (lat, lon, speed)


class RPMModule(DataSource):
    """
    RPM and vehicle data interface.
    Reads RPM, gear, throttle, and other OBD-II parameters via CAN bus.
    
    Placeholder implementation - replace with actual hardware interface:
    - CAN bus interface (python-can library)
    - OBD-II adapter (pyobd, obd)
    - Vehicle-specific CAN decoding
    """
    
    def __init__(self, can_interface: str = "can0", baudrate: int = 250000):
        self.can_interface = can_interface
        self.baudrate = baudrate
        self.connected = False
        self.last_data = None
    
    def connect(self):
        """Connect to CAN bus."""
        # TODO: Implement actual CAN connection
        # import can
        # self.bus = can.interface.Bus(self.can_interface, bustype='socketcan', bitrate=self.baudrate)
        self.connected = True
        print(f"RPM/CAN connected on {self.can_interface}")
    
    def disconnect(self):
        """Disconnect from CAN bus."""
        # if self.bus:
        #     self.bus.shutdown()
        self.connected = False
    
    def read(self) -> Optional[Tuple[float, int, float]]:
        """
        Read RPM and vehicle data.
        
        Returns:
            (rpm, gear, throttle_percent) or None if no data
        """
        if not self.connected:
            return None
        
        # TODO: Parse CAN messages for specific vehicle
        # For now, return mock data
        # Example: (3000.0, 3, 45.5)  # 3000 RPM, 3rd gear, 45.5% throttle
        return self.last_data
    
    def update_mock_data(self, rpm: float, gear: int, throttle: float):
        """For testing: set mock RPM data."""
        self.last_data = (rpm, gear, throttle)


class GPSRPMLogger:
    """
    Orchestrates GPS and RPM data collection.
    Manages hardware connections and data buffering.
    """
    
    def __init__(self, use_mock: bool = True):
        """
        Initialize logger.
        
        Args:
            use_mock: If True, use mock data for testing. If False, connect to real hardware.
        """
        self.use_mock = use_mock
        self.gps = GPSModule()
        self.rpm = RPMModule()
        
        self.is_logging = False
        self.gps_buffer = []
        self.rpm_buffer = []
    
    def start(self):
        """Connect to data sources and start logging."""
        try:
            self.gps.connect()
            self.rpm.connect()
            self.is_logging = True
            print("Data logger started.")
        except Exception as e:
            print(f"Error starting logger: {e}")
            self.is_logging = False
    
    def stop(self):
        """Stop logging and disconnect from data sources."""
        self.is_logging = False
        self.gps.disconnect()
        self.rpm.disconnect()
        print("Data logger stopped.")
    
    def get_gps_data(self) -> Optional[Tuple[float, float, float]]:
        """
        Get latest GPS data.
        
        Returns:
            (lat, lon, speed_mps) or None
        """
        return self.gps.read()
    
    def get_rpm_data(self) -> Optional[Tuple[float, int, float]]:
        """
        Get latest RPM data.
        
        Returns:
            (rpm, gear, throttle_percent) or None
        """
        return self.rpm.read()
    
    def set_mock_gps(self, lat: float, lon: float, speed: float):
        """For testing: inject mock GPS data."""
        self.gps.update_mock_data(lat, lon, speed)
    
    def set_mock_rpm(self, rpm: float, gear: int, throttle: float):
        """For testing: inject mock RPM data."""
        self.rpm.update_mock_data(rpm, gear, throttle)
