"""
Lap Detector
Detects lap boundaries by monitoring GPS position relative to start/finish line.
"""

from typing import List, Tuple, Optional
import numpy as np


class LapDetector:
    """
    Detects lap boundaries using GPS coordinates.
    Tracks when the vehicle crosses the start/finish line.
    """
    
    def __init__(self, finish_line_lat: float = 59.6552, finish_line_lon: float = 10.7748, 
                 tolerance_m: float = 50.0):
        """
        Initialize lap detector.
        
        Args:
            finish_line_lat: Latitude of start/finish line (default: Oslo coordinates)
            finish_line_lon: Longitude of start/finish line
            tolerance_m: Distance tolerance around finish line (meters) to count as crossing
        """
        self.finish_line_lat = finish_line_lat
        self.finish_line_lon = finish_line_lon
        self.tolerance_m = tolerance_m
        
        self.lap_count = 0
        self.last_position = None
        self.was_near_finish = False
        self.lap_boundaries = []
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two GPS coordinates in meters.
        
        Args:
            lat1, lon1: Coordinate 1
            lat2, lon2: Coordinate 2
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def check_lap_boundary(self, gps_data: Tuple[float, float, float]) -> Optional[int]:
        """
        Check if the vehicle has crossed the start/finish line.
        
        Args:
            gps_data: (lat, lon, speed_mps)
            
        Returns:
            Lap number if lap boundary detected, None otherwise
        """
        if gps_data is None:
            return None
        
        lat, lon, speed = gps_data
        
        # Calculate distance to finish line
        dist_to_finish = self.haversine_distance(
            lat, lon,
            self.finish_line_lat, self.finish_line_lon
        )
        
        # Check if near finish line
        is_near_finish = dist_to_finish < self.tolerance_m
        
        # Detect crossing: was far, now near, then far again
        if is_near_finish and not self.was_near_finish:
            # Just entered the zone
            self.was_near_finish = True
        elif not is_near_finish and self.was_near_finish:
            # Just exited the zone - lap complete!
            self.lap_count += 1
            self.was_near_finish = False
            return self.lap_count
        
        self.last_position = gps_data
        return None
    
    def detect_all_laps(self, raw_data: List[dict]) -> List[Tuple[int, int, int, float, float]]:
        """
        Post-process raw data to detect all lap boundaries.
        
        Args:
            raw_data: List of data points, each with 'gps' key containing (lat, lon, speed_mps)
            
        Returns:
            List of (lap_num, start_idx, end_idx, start_time, end_time) tuples
        """
        lap_boundaries = []
        in_zone = False
        zone_enter_idx = None
        zone_enter_time = None
        lap_count = 0
        
        for idx, point in enumerate(raw_data):
            gps = point['gps']
            ts = point['timestamp']
            
            if gps is None:
                continue
            
            lat, lon, speed = gps
            
            # Calculate distance to finish line
            dist_to_finish = self.haversine_distance(
                lat, lon,
                self.finish_line_lat, self.finish_line_lon
            )
            
            is_near_finish = dist_to_finish < self.tolerance_m
            
            # Detect entry into zone
            if is_near_finish and not in_zone:
                in_zone = True
                zone_enter_idx = idx
                zone_enter_time = ts
            
            # Detect exit from zone (lap complete)
            elif not is_near_finish and in_zone:
                in_zone = False
                lap_count += 1
                
                # Record lap boundary
                lap_boundaries.append((
                    lap_count,
                    zone_enter_idx,
                    idx,
                    zone_enter_time,
                    ts
                ))
        
        return lap_boundaries
    
    def set_finish_line(self, lat: float, lon: float, tolerance_m: float = 50.0):
        """Update finish line location."""
        self.finish_line_lat = lat
        self.finish_line_lon = lon
        self.tolerance_m = tolerance_m
