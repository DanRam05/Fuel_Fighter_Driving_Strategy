import casadi as ca
import numpy as np

def compute_curvature_splines(s, x, y):
    """
    Creates smooth B-spline interpolants for track geometry.
    Returns: x(s), y(s), heading(s), curvature(s)
    """
    # Use degree 3 for C2 continuity (smooth acceleration/steering)
    x_spline = ca.interpolant('x', 'bspline', [s], x)
    y_spline = ca.interpolant('y', 'bspline', [s], y)

    # We define a symbolic s to differentiate and get heading/curvature
    s_sym = ca.MX.sym('s')
    dx = ca.gradient(x_spline(s_sym), s_sym)
    dy = ca.gradient(y_spline(s_sym), s_sym)
    
    # Heading (angle)
    psi = ca.atan2(dy, dx)
    
    # Curvature (kappa = d_psi / ds)
    kappa = ca.gradient(psi, s_sym)

    # Wrap these into CasADi functions for the OCP
    f_x = ca.Function('f_x', [s_sym], [x_spline(s_sym)])
    f_y = ca.Function('f_y', [s_sym], [y_spline(s_sym)])
    f_psi = ca.Function('f_psi', [s_sym], [psi])
    f_kappa = ca.Function('f_kappa', [s_sym], [kappa])

    return f_x, f_y, f_psi, f_kappa