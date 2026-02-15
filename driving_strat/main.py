import os
import sys
from pathlib import Path

# Ensure `driving_strat` folder is on sys.path so `track` imports work
pkg_root = Path(__file__).resolve().parent
if str(pkg_root) not in sys.path:
	sys.path.insert(0, str(pkg_root))

from track.track_loader import load_track
from track.track_interpolant import build_track_interpolants
from utils.plotting import plot_track

# Use a path relative to this file so script works regardless of cwd
data_path = pkg_root / 'data' / 'sem_2025_eu.csv'

s, x, y, xl, yl, xr, yr = load_track(str(data_path), radius=6.0)

# build interpolants from centerline
x_s, y_s = build_track_interpolants(s, x, y)

# Plot the raw track centerline and boundaries (visual radius shown separately)

plot_track(x, y, line_radius_m=6)
try:
	import matplotlib.pyplot as plt
	ax = plt.gca()
	ax.plot(xl, yl, linestyle='--', color='C1', label='left boundary')
	ax.plot(xr, yr, linestyle='--', color='C2', label='right boundary')
	# Mark and label the origin (first point, now at x=0, y=0)
	ax.plot(x[0], y[0], 'ro', label='origin (x=0, y=0)')
	ax.annotate('origin (x=0, y=0)', (x[0], y[0]), textcoords="offset points", xytext=(10,10), ha='left', color='red', fontsize=10, fontweight='bold')
	ax.legend()
except Exception:
	pass

# Later:
# build OCP
# solve