"""
Post-Lap Analyzer
Computes metrics and insights from recorded telemetry data.
"""

from typing import List, Dict, Any
import numpy as np


class PostLapAnalyzer:
    """
    Analyzes individual laps to extract performance metrics and insights.
    """
    
    def __init__(self):
        pass
    
    def analyze_lap(self, lap_num: int, lap_data: List[dict], 
                    track_name: str = "", driver_name: str = "") -> Dict[str, Any]:
        """
        Analyze a single lap's telemetry data.
        
        Args:
            lap_num: Lap number
            lap_data: List of data points for this lap
            track_name: Name of track
            driver_name: Name of driver
            
        Returns:
            Dictionary with lap metrics
        """
        if not lap_data:
            return {}
        
        # Extract GPS and RPM data
        gps_points = [p['gps'] for p in lap_data if p['gps'] is not None]
        rpm_points = [p['rpm'] for p in lap_data if p['rpm'] is not None]
        timestamps = [p['timestamp'] for p in lap_data]
        
        # Calculate metrics
        result = {
            'lap_num': lap_num,
            'track_name': track_name,
            'driver_name': driver_name,
            'data_points': len(lap_data),
        }
        
        # Time metrics
        if len(timestamps) >= 2:
            duration = timestamps[-1] - timestamps[0]
            result['duration'] = duration
            result['avg_update_rate'] = len(lap_data) / duration if duration > 0 else 0
        
        # Speed metrics (from GPS)
        if gps_points:
            speeds = np.array([p[2] for p in gps_points])  # Extract speed_mps
            result['speeds'] = speeds
            result['avg_speed'] = np.mean(speeds)
            result['max_speed'] = np.max(speeds)
            result['min_speed'] = np.min(speeds)
            result['speed_std'] = np.std(speeds)
            
            # Speed profile: percentage of lap at different speed ranges
            result['speed_profile'] = self._compute_speed_profile(speeds)
        
        # RPM metrics (from vehicle data)
        if rpm_points:
            rpms = np.array([p[0] for p in rpm_points])  # Extract RPM
            gears = np.array([p[1] for p in rpm_points])  # Extract gear
            throttles = np.array([p[2] for p in rpm_points])  # Extract throttle %
            
            result['rpms'] = rpms
            result['avg_rpm'] = np.mean(rpms)
            result['max_rpm'] = np.max(rpms)
            result['min_rpm'] = np.min(rpms)
            
            # Gear analysis
            unique_gears = np.unique(gears)
            result['gears_used'] = unique_gears.tolist()
            result['avg_throttle'] = np.mean(throttles)
            result['max_throttle'] = np.max(throttles)
            
            # Efficiency metrics
            result['throttle_profile'] = self._compute_throttle_profile(throttles)
            result['engine_load_factor'] = self._compute_engine_load(rpms, throttles)
        
        # Position metrics (track coverage)
        if gps_points:
            lats = np.array([p[0] for p in gps_points])
            lons = np.array([p[1] for p in gps_points])
            result['track_bounds'] = {
                'lat_min': float(np.min(lats)),
                'lat_max': float(np.max(lats)),
                'lon_min': float(np.min(lons)),
                'lon_max': float(np.max(lons)),
            }
        
        # Generate notes/insights
        result['notes'] = self._generate_insights(result)
        
        return result
    
    def _compute_speed_profile(self, speeds: np.ndarray) -> Dict[str, float]:
        """
        Compute percentage of lap time in each speed range.
        
        Args:
            speeds: Array of speeds in m/s
            
        Returns:
            Dictionary with speed range percentages
        """
        profile = {
            'slow_pct': float(np.sum(speeds < 5) / len(speeds) * 100),
            'medium_pct': float(np.sum((speeds >= 5) & (speeds < 15)) / len(speeds) * 100),
            'fast_pct': float(np.sum((speeds >= 15) & (speeds < 25)) / len(speeds) * 100),
            'very_fast_pct': float(np.sum(speeds >= 25) / len(speeds) * 100),
        }
        return profile
    
    def _compute_throttle_profile(self, throttles: np.ndarray) -> Dict[str, float]:
        """
        Compute percentage of lap time at different throttle levels.
        
        Args:
            throttles: Array of throttle percentages (0-100)
            
        Returns:
            Dictionary with throttle level percentages
        """
        profile = {
            'idle_pct': float(np.sum(throttles < 10) / len(throttles) * 100),
            'partial_pct': float(np.sum((throttles >= 10) & (throttles < 75)) / len(throttles) * 100),
            'full_throttle_pct': float(np.sum(throttles >= 75) / len(throttles) * 100),
        }
        return profile
    
    def _compute_engine_load(self, rpms: np.ndarray, throttles: np.ndarray) -> float:
        """
        Compute overall engine load factor (0-1).
        Combines RPM and throttle into a single efficiency metric.
        
        Args:
            rpms: Array of RPM values
            throttles: Array of throttle percentages
            
        Returns:
            Load factor between 0 and 1
        """
        # Normalize RPM (max typically ~7000 for street cars, race cars can go higher)
        rpm_factor = np.mean(rpms) / 7000.0
        throttle_factor = np.mean(throttles) / 100.0
        
        # Combine: weighted average
        load_factor = 0.6 * min(rpm_factor, 1.0) + 0.4 * throttle_factor
        return float(load_factor)
    
    def _generate_insights(self, result: Dict[str, Any]) -> str:
        """
        Generate human-readable insights from lap metrics.
        
        Args:
            result: Lap analysis result dictionary
            
        Returns:
            Insights string
        """
        insights = []
        
        if 'avg_speed' in result:
            insights.append(f"Avg speed: {result['avg_speed']*3.6:.1f} km/h")
        
        if 'duration' in result:
            insights.append(f"Duration: {result['duration']:.1f} s")
        
        if 'avg_throttle' in result:
            if result['avg_throttle'] < 30:
                insights.append("Conservative throttle usage")
            elif result['avg_throttle'] > 70:
                insights.append("Aggressive throttle profile")
        
        if 'speed_std' in result:
            if result['speed_std'] < 2:
                insights.append("Smooth, consistent pace")
            elif result['speed_std'] > 8:
                insights.append("Highly variable speed - work on smoothness")
        
        return "; ".join(insights) if insights else "No insights available"
    
    def compare_laps(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple lap results to identify trends and best practices.
        
        Args:
            results: List of lap analysis results
            
        Returns:
            Comparison metrics
        """
        if not results:
            return {}
        
        lap_nums = [r['lap_num'] for r in results]
        durations = [r.get('duration', 0) for r in results]
        avg_speeds = [r.get('avg_speed', 0) for r in results]
        avg_rpms = [r.get('avg_rpm', 0) for r in results]
        
        best_lap_idx = np.argmin(durations)
        
        comparison = {
            'num_laps': len(results),
            'best_lap': lap_nums[best_lap_idx],
            'best_lap_time': durations[best_lap_idx],
            'avg_lap_time': np.mean(durations),
            'consistency': np.std(durations),  # Lower is more consistent
            'improvement_trend': self._compute_trend(durations),
        }
        
        return comparison
    
    def _compute_trend(self, values: np.ndarray) -> str:
        """
        Determine if values are improving, degrading, or stable.
        
        Args:
            values: Array of metric values across laps
            
        Returns:
            "improving", "degrading", or "stable"
        """
        if len(values) < 2:
            return "insufficient_data"
        
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        improvement = (first_half - second_half) / first_half
        
        if improvement > 0.05:
            return "improving"
        elif improvement < -0.05:
            return "degrading"
        else:
            return "stable"
