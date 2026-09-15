"""
Telemetry System
Standalone GPS + RPM logging and post-lap analysis.
Independent of driving_strat optimization.
"""

from .logger import GPSRPMLogger, GPSModule, RPMModule
from .lap_detector import LapDetector
from .analyzer import PostLapAnalyzer
from .main_telemetry import TelemetrySession

__version__ = "0.1.0"
__all__ = [
    'GPSRPMLogger',
    'GPSModule',
    'RPMModule',
    'LapDetector',
    'PostLapAnalyzer',
    'TelemetrySession',
]
