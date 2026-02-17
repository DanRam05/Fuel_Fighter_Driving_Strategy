import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

# Set up the problem
N = 100 # number of control intervals
opti = ca.Opti() # Optimization problem

# Declare decision variables
X = opti.variable(2, N+1) # state trajectory
pos = X[0, :]
speed = X[1, :]
U = opti.variable(1,N) # control trajectory (throttle)
T = opti.variable() # final time

# Set the objective, here time
opti.minimize(T) # race in minimal time

# Specify system dynamics
f = lambda x,u: ca.vertcat(x[1], u-x[1]) # dx/dt = f(x,u)

# Create gap closing constraints, picking Runge-Kutta as integration method
dt = T/N # length of a control interval
for k in range(N): # loop over control intervals
    # Runge-Kutta 4 integration
    k1 = f(X[:,k], U[:,k])
    k2 = f(X[:,k]+dt/2*k1, U[:,k])
    k3 = f(X[:,k]+dt/2*k2, U[:,k])
    k4 = f(X[:,k]+dt*k3, U[:,k])
    x_next = X[:,k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    opti.subject_to(X[:,k+1] == x_next) # close the gaps
    
# Set path constraints.
limit = lambda pos: 1 - ca.sin(2*ca.pi*pos)/2
opti.subject_to(speed <= limit(pos)) # track speed limit
opti.subject_to(opti.bounded(0, U, 1)) # control is limited

# Set boundary conditions
opti.subject_to(pos[0] == 0) # start at position 0 
opti.subject_to(speed[0] == 0) # from stand-still
opti.subject_to(pos[-1] == 1) # finish line at position 1

# One extra constraint
opti.subject_to(T >= 0.1) # Time must be positive

# Provide initial guesses for the solver
opti.set_initial(pos, np.linspace(0, 1, N+1))
opti.set_initial(speed, 0.5)
opti.set_initial(U, 0.5)
opti.set_initial(T, 2.0)

# Solve the NLP using IPOPT
opti.solver('ipopt', {}, {
    'max_iter': 2000,
    'print_level': 5}) # set numerical backend
sol = opti.solve() # actual solve

# Plot speed
plt.figure()
plt.plot(sol.value(speed))
plt.title("Speed")
plt.xlabel("Time")
plt.ylabel("Speed")
plt.show()

# Plot position
plt.figure()
plt.plot(sol.value(pos))
plt.title("Position")
plt.xlabel("Time")
plt.ylabel("Position")
plt.show()