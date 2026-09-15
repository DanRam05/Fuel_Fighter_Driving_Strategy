# Telemetry System Documentation

Standalone GPS + RPM logging and post-lap analysis for driver feedback.

## Overview

This is a **completely independent system** from `driving_strat`. It records real-time vehicle performance data and analyzes lap metrics. No connection to the optimization program required.

## Architecture

```
telemetry/
├── main_telemetry.py       # Entry point and orchestrator
├── logger.py               # GPS and RPM hardware interfaces
├── lap_detector.py         # Lap boundary detection
├── analyzer.py             # Post-lap metrics and insights
├── __init__.py             # Package initialization
└── data/                   # Output CSV files (created automatically)
```

## Components

### 1. **logger.py** - Data Collection
- `GPSModule`: Interface to GPS/GNSS hardware
  - Currently: Placeholder with mock data support
  - TODO: Integrate with actual GPS hardware (serial, NTRIP, etc.)
  - Data: Latitude, Longitude, Speed (m/s)

- `RPMModule`: Interface to vehicle CAN bus / OBD-II
  - Currently: Placeholder with mock data support
  - TODO: Implement CAN bus parsing for your vehicle
  - Data: RPM, Gear, Throttle %

- `GPSRPMLogger`: Orchestrates both modules
  - Manages hardware connections
  - Provides buffering and synchronization

### 2. **lap_detector.py** - Lap Boundary Detection
- `LapDetector`: Detects lap boundaries using GPS
- Uses haversine distance to track when vehicle crosses start/finish line
- Configurable tolerance zone around finish line
- Real-time detection or post-processing mode

### 3. **analyzer.py** - Post-Lap Analysis
- `PostLapAnalyzer`: Computes lap metrics
- Metrics generated:
  - **Time**: Duration, update frequency
  - **Speed**: Average, max, min, consistency
  - **RPM**: Average, max, min, gear usage
  - **Throttle**: Profile (idle/partial/full), efficiency
  - **Position**: Track bounds (lat/lon ranges)
  - **Insights**: Human-readable observations

### 4. **main_telemetry.py** - Session Orchestration
- `TelemetrySession`: Main controller
- Workflow:
  1. Initialize session
  2. Start recording (begins data collection)
  3. Stop recording (ends data collection)
  4. Process session (detect laps, analyze each)
  5. Save results (CSV files)
  6. Print summary

## Quick Start

### Mock Data (Testing)
```python
from telemetry import TelemetrySession

session = TelemetrySession(track_name="SEM_2025_EU", driver_name="John Doe")
session.start_recording()
# Inject mock data for testing
# ...
session.stop_recording()
session.process_session()
session.save_session()
```

### Real Hardware
1. **Update `logger.py`** to connect to real GPS and CAN bus
2. Run the same code above - data source is abstracted

## Hardware Integration Guide

### GPS Integration
In `logger.py`, replace the TODO in `GPSModule`:

```python
import serial
import pynmea2

# Parse NMEA sentences from serial GPS
ser = serial.Serial('/dev/ttyUSB0', 9600)
while True:
    data = ser.readline()
    msg = pynmea2.parse(data)
    if msg.sentence_type == 'GGA':
        lat, lon = msg.latitude, msg.longitude
```

Or use NTRIP for real-time kinematic:
```python
from ntrip import NTRIPClient
client = NTRIPClient(host='rtk.provider.com', port=2101)
```

### RPM/CAN Integration
In `logger.py`, replace the TODO in `RPMModule`:

```python
import can
from can.interfaces.socketcan import SocketCanBus

bus = SocketCanBus(channel='can0', bitrate=250000)

# Define DBC file or decode manually
# Example: Fuel Fighter vehicle specific CAN message parsing
msg = bus.recv()
rpm = extract_rpm_from_can(msg)      # Custom decoder
gear = extract_gear_from_can(msg)
throttle = extract_throttle_from_can(msg)
```

Or use OBD-II adapter:
```python
import obd
connection = obd.OBD()
rpm = connection.query(obd.commands.RPM).value
speed = connection.query(obd.commands.SPEED).value
```

## Output

### Raw Telemetry CSV
`telemetry_raw_<TRACK>_<TIMESTAMP>.csv`

Columns: timestamp, lat, lon, speed_mps, rpm, gear, throttle_pct

### Lap Summary CSV
`lap_summary_<TRACK>_<TIMESTAMP>.csv`

Columns: lap_num, duration_s, avg_speed_ms, max_speed_ms, avg_rpm, max_rpm, avg_throttle_pct, notes

## Example Metrics

For a completed lap:
```
Lap 1 Summary:
├── Duration: 245.32 s
├── Avg Speed: 14.8 m/s (53.3 km/h)
├── Max Speed: 28.5 m/s (102.6 km/h)
├── Avg RPM: 3200
├── Max RPM: 6800
└── Insights: Smooth, consistent pace; Conservative throttle usage
```

## Comparison to driving_strat

| Aspect | driving_strat | telemetry |
|--------|---|---|
| **Purpose** | Pre-lap optimization | Real-time logging + analysis |
| **When** | Before driving | During/after driving |
| **Output** | Ideal lap profile | Actual performance metrics |
| **Use Case** | "What should I do?" | "What did I do?" |
| **Integration** | Not required | Optional (for comparison feedback) |

## Future Enhancements

- [ ] Real hardware integration (GPS, CAN)
- [ ] Sector-based analysis (not just lap totals)
- [ ] Telemetry visualization (matplotlib/web dashboard)
- [ ] Optional integration with driving_strat for delta analysis
- [ ] RPM/throttle efficiency scoring
- [ ] Driver coaching suggestions based on trends
- [ ] Export to vehicle telemetry format (MoTeC, AIM, etc.)

## Notes

- All speed values in m/s (convert to km/h with × 3.6)
- All distances in meters
- Lat/Lon in decimal degrees
- Finish line defaults to Oslo coordinates - **update for your track!**
- GPS accuracy depends on hardware (typically ±2-5m for RTK, ±5-10m for standard)
- CAN/OBD-II integration is vehicle-specific - customize for Fuel Fighter
