import casadi as ca


def build_track_interpolants(s, x, y):
	x_s = ca.interpolant('x_s', 'linear', [s], x)
	y_s = ca.interpolant('y_s', 'linear', [s], y)
	return x_s, y_s
