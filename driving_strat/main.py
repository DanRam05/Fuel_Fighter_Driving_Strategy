import os
import sys
from pathlib import Path

# Ensure `driving_strat` folder is on sys.path so `track` imports work
pkg_root = Path(__file__).resolve().parent
if str(pkg_root) not in sys.path:
	sys.path.insert(0, str(pkg_root))

from track.track_loader import load_track
from track.track_interpolant import build_track_interpolants

# Use a path relative to this file so script works regardless of cwd
data_path = pkg_root / 'data' / 'sem_2025_eu.csv'

s, x, y = load_track(str(data_path))

x_s, y_s = build_track_interpolants(s, x, y)

# Later:
# build OCP
# solve
# plot