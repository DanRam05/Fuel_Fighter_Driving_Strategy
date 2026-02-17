import casadi as ca
import numpy as np

def compute_curvature_splines(s, x, y):
    """
    Creates smooth B-spline interpolants for track geometry.
    Ensures heading and curvature are continuous for the solver.
    """
    # 1. Pre-calculate continuous heading (unwrapped)
    dx_raw = np.gradient(x, s)
    dy_raw = np.gradient(y, s)
    psi_raw = np.arctan2(dy_raw, dx_raw)
    
    # Ensure heading doesn't jump from pi to -pi
    psi_continuous = np.unwrap(psi_raw)

    # 2. Build B-splines for coordinates and the continuous heading
    # Using degree 3 (cubic) ensures C2 continuity for curvature
    x_spline = ca.interpolant('x', 'bspline', [s], x)
    y_spline = ca.interpolant('y', 'bspline', [s], y)
    psi_spline = ca.interpolant('psi', 'bspline', [s], psi_continuous)

    # 3. Define symbolic derivatives
    s_sym = ca.MX.sym('s')
    
    # Calculate curvature directly from the continuous heading spline
    # kappa = d(psi) / ds
    kappa = ca.gradient(psi_spline(s_sym), s_sym)

    # 4. Wrap into CasADi functions
    f_x = ca.Function('f_x', [s_sym], [x_spline(s_sym)])
    f_y = ca.Function('f_y', [s_sym], [y_spline(s_sym)])
    f_psi = ca.Function('f_psi', [s_sym], [psi_spline(s_sym)])
    f_kappa = ca.Function('f_kappa', [s_sym], [kappa])

    return f_x, f_y, f_psi, f_kappa