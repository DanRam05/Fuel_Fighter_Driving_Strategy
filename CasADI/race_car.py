import casadi as ca

N = 100
opti = ca.Opti()

X = opti.variable(2, N+1)
pos = X[0]
speed = X[1]
U = opti.variable(1,N)
T = opti.variable()

opti.minimize(T)

f = lambda x,u: ca.vertcat(x[1], u-x[1])

dt = T/N
for k in range(N):
    k1 = f(X[:,k], U[:,k])
    k2 = f(X[:,k]+dt/2*k1, U[:,k])
    k3 = f(X[:,k]+dt/2*k2, U[:,k])
    k4 = f(X[:,k]+dt*k3, U[:,k])
    x_next = X[:,k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    opti.subject_to(X[:,k+1] == x_next)