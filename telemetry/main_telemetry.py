"""
Telemetry Logger and Post-Lap Analyzer
Standalone system for logging GPS and RPM data, detecting laps, and analyzing performance.
Independent of driving_strat optimization.
"""

import time
from datetime import datetime
from pathlib import Path
import csv
import numpy as np
from logger import GPSRPMLogger
from lap_detector import LapDetector
from analyzer import PostLapAnalyzer


class TelemetrySession:
    """
    Main orchestrator for a telemetry recording session.
    Handles data collection, lap detection, and post-session analysis.
    """
    
    def __init__(self, track_name: str, driver_name: str, output_dir: Path = None):
        """
        Initialize telemetry session.
        
        Args:
            track_name: Name of the track (e.g., "SEM_2025_EU")
            driver_name: Name of the driver
            output_dir: Directory to save telemetry data (default: telemetry/data/)
        """
        self.track_name = track_name
        self.driver_name = driver_name
        self.session_start = datetime.now()
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path(__file__).parent / "data"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.logger = GPSRPMLogger()
        self.lap_detector = LapDetector()
        self.analyzer = PostLapAnalyzer()
        
        # Session data storage
        self.raw_data = []  # List of (timestamp, gps_coords, rpm) tuples
        self.lap_boundaries = []  # List of (lap_num, start_idx, end_idx, start_time, end_time)
        self.lap_analyses = []  # List of lap analysis results
        
        self.is_recording = False
        
    def start_recording(self):
        """Start logging GPS and RPM data."""
        self.is_recording = True
        self.logger.start()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Recording started - {self.track_name} ({self.driver_name})")
    
    def stop_recording(self):
        """Stop logging and finalize session."""
        self.is_recording = False
        self.logger.stop()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Recording stopped")
    
    def collect_data_loop(self, polling_interval: float = 0.1):
        """
        Main data collection loop (run this in a thread or async context).
        
        Args:
            polling_interval: Time between GPS/RPM samples (seconds)
        """
        while self.is_recording:
            # Get latest GPS and RPM data
            gps_data = self.logger.get_gps_data()  # (lat, lon, speed_mps)
            rpm_data = self.logger.get_rpm_data()   # (rpm, gear, throttle_percent)
            
            if gps_data is not None and rpm_data is not None:
                timestamp = time.time()
                data_point = {
                    'timestamp': timestamp,
                    'gps': gps_data,
                    'rpm': rpm_data,
                }
                self.raw_data.append(data_point)
                
                # Optionally check for lap boundaries in real-time
                lap_num = self.lap_detector.check_lap_boundary(gps_data)
                if lap_num is not None:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Lap {lap_num} detected!")
            
            time.sleep(polling_interval)
    
    def process_session(self):
        """
        Post-session processing: detect laps and analyze each.
        Call this after stop_recording().
        """
        if not self.raw_data:
            print("No data to process.")
            return
        
        print(f"\nProcessing {len(self.raw_data)} data points...")
        
        # Detect lap boundaries
        self.lap_boundaries = self.lap_detector.detect_all_laps(self.raw_data)
        print(f"Detected {len(self.lap_boundaries)} laps.")
        
        # Analyze each lap
        for lap_num, start_idx, end_idx, start_time, end_time in self.lap_boundaries:
            lap_data = self.raw_data[start_idx:end_idx]
            
            # Run analysis on this lap
            lap_result = self.analyzer.analyze_lap(
                lap_num=lap_num,
                lap_data=lap_data,
                track_name=self.track_name,
                driver_name=self.driver_name,
            )
            self.lap_analyses.append(lap_result)
            
            print(f"\n--- Lap {lap_num} Summary ---")
            print(f"Duration: {lap_result['duration']:.2f} s")
            print(f"Avg Speed: {lap_result['avg_speed']:.2f} m/s ({lap_result['avg_speed']*3.6:.2f} km/h)")
            print(f"Max Speed: {lap_result['max_speed']:.2f} m/s ({lap_result['max_speed']*3.6:.2f} km/h)")
            print(f"Avg RPM: {lap_result['avg_rpm']:.0f}")
            print(f"Max RPM: {lap_result['max_rpm']:.0f}")
    
    def save_session(self):
        """Save raw data and analysis results to CSV files."""
        timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Save raw telemetry
        raw_filename = self.output_dir / f"telemetry_raw_{self.track_name}_{timestamp_str}.csv"
        self._save_raw_data(raw_filename)
        print(f"Raw data saved: {raw_filename}")
        
        # Save lap summaries
        summary_filename = self.output_dir / f"lap_summary_{self.track_name}_{timestamp_str}.csv"
        self._save_lap_summary(summary_filename)
        print(f"Lap summary saved: {summary_filename}")
    
    def _save_raw_data(self, filename: Path):
        """Save raw GPS and RPM data to CSV."""
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['timestamp', 'lat', 'lon', 'speed_mps', 'rpm', 'gear', 'throttle_pct']
            )
            writer.writeheader()
            
            for point in self.raw_data:
                ts = point['timestamp']
                gps = point['gps']
                rpm = point['rpm']
                
                writer.writerow({
                    'timestamp': ts,
                    'lat': gps[0],
                    'lon': gps[1],
                    'speed_mps': gps[2],
                    'rpm': rpm[0],
                    'gear': rpm[1],
                    'throttle_pct': rpm[2],
                })
    
    def _save_lap_summary(self, filename: Path):
        """Save lap-by-lap analysis summary to CSV."""
        if not self.lap_analyses:
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'lap_num', 'duration_s', 'avg_speed_ms', 'max_speed_ms',
                    'avg_rpm', 'max_rpm', 'avg_throttle_pct', 'notes'
                ]
            )
            writer.writeheader()
            
            for result in self.lap_analyses:
                writer.writerow({
                    'lap_num': result['lap_num'],
                    'duration_s': f"{result['duration']:.2f}",
                    'avg_speed_ms': f"{result['avg_speed']:.2f}",
                    'max_speed_ms': f"{result['max_speed']:.2f}",
                    'avg_rpm': f"{result['avg_rpm']:.0f}",
                    'max_rpm': f"{result['max_rpm']:.0f}",
                    'avg_throttle_pct': f"{result['avg_throttle']:.1f}",
                    'notes': result.get('notes', ''),
                })
    
    def print_session_summary(self):
        """Print final session summary."""
        print("\n" + "="*60)
        print("TELEMETRY SESSION SUMMARY")
        print("="*60)
        print(f"Track: {self.track_name}")
        print(f"Driver: {self.driver_name}")
        print(f"Total Laps: {len(self.lap_analyses)}")
        print(f"Total Duration: {sum(l['duration'] for l in self.lap_analyses):.2f} s")
        
        if self.lap_analyses:
            avg_speeds = [l['avg_speed'] for l in self.lap_analyses]
            print(f"Average Speed (all laps): {np.mean(avg_speeds):.2f} m/s")
            print(f"Best Lap: Lap {self.lap_analyses[np.argmin([l['duration'] for l in self.lap_analyses])]['lap_num']}")
        print("="*60)


def main():
    """
    Example usage of the telemetry system.
    """
    # Initialize session
    session = TelemetrySession(
        track_name="SEM_2025_EU",
        driver_name="Test Driver"
    )
    
    print("Starting telemetry session...")
    print("Make sure GPS and RPM data sources are connected.")
    print("\nPress Enter to start recording, then again to stop.")
    input()
    
    session.start_recording()
    
    # In a real scenario, this would run in a thread or async context
    # For now, we'll wait for user input to stop
    input()
    
    session.stop_recording()
    
    # Post-process the data
    session.process_session()
    
    # Save results
    session.save_session()
    
    # Print summary
    session.print_session_summary()


if __name__ == "__main__":
    main()
