import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

def load_track(path, radius=6.0):
    df = pd.read_csv(path)

    # Support multiple possible column names for distance
    if 'Distance' in df.columns:
        s_col = 'Distance'
    elif 'Distance from Lap Line (m)' in df.columns:
        s_col = 'Distance from Lap Line (m)'
    else:
        s_col = df.columns[0]

    s = df[s_col].values.astype(float)
    x = df['UTMX'].values.astype(float)
    y = df['UTMY'].values.astype(float)
    z = df['Elevation (m)'].values.astype(float) if 'Elevation (m)' in df.columns else np.zeros_like(s)

    # --- Shift so first point is at (0,0) ---
    x_shift = x[0]
    y_shift = y[0]
    x = x - x_shift
    y = y - y_shift

    # Compute unit tangents using central differences (forward/backward at ends)
    pts = np.column_stack((x, y))
    n = pts.shape[0]
    tangents = np.zeros_like(pts)
    for i in range(n):
        if i == 0:
            d = pts[1] - pts[0]
        elif i == n - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[i + 1] - pts[i - 1]
        norm = np.linalg.norm(d)
        if norm == 0:
            tangents[i] = np.array([1.0, 0.0])
        else:
            tangents[i] = d / norm

    # normals: rotate tangent by 90 deg
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

    left = pts + normals * float(radius)
    right = pts - normals * float(radius)

    xl = left[:, 0]
    yl = left[:, 1]
    xr = right[:, 0]
    yr = right[:, 1]

    return s, x, y, xl, yl, xr, yr, z