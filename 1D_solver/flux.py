import numpy as np
import math as m

def LaxF(U, U_prim, nx, gamma, dt, dx):
    F = np.zeros(U_prim.shape)
    U_next = np.zeros(U_prim.shape)
    F[0, :] = U_prim[0, :]*U_prim[1, :]
    F[1, :] = U_prim[0, :]*U_prim[1, :]**2 + U_prim[2, :]
    F[2, :] = U_prim[1, :]*(0.5*U_prim[0, :]*U_prim[1, :]**2 + gamma/(gamma-1)*U_prim[2, :])

    for i in range(1, nx-1):
        U_next[:, i] = 0.5*(U[:, i-1] + U[:, i+1]) - 0.5*dt/dx*(F[:, i+1] - F[:, i-1])

    return U_next

def LaxW(U, U_prim, nx, gamma, dt, dx):
    # u_i - 0.5*dt/dx*(F_i+1 - F_i-1) + 0.5*dt/dx^2*0.5*((ui+ - ui)*(Fi+ - Fi) - (ui - ui-)*(Fi - Fi-))
    F = np.zeros(U_prim.shape)
    U_next = np.zeros(U_prim.shape)
    F[0, :] = U_prim[0, :] * U_prim[1, :]
    F[1, :] = U_prim[0, :] * U_prim[1, :] ** 2 + U_prim[2, :]
    F[2, :] = U_prim[1, :] * (0.5 * U_prim[0, :] * U_prim[1, :] ** 2 + gamma / (gamma - 1) * U_prim[2, :])

    for i in range(1, nx-1):
        U_next[:, i] = U[:, i] - 0.5*(dt/dx)*(F[:, i+1] - F[:, i-1]) + ((0.5*dt/dx)**2)*((U[:, i+1] - U[:, i])*(F[:, i+1] - F[:, i])- (U[:, i] - U[:, i-1])*(F[:, i] - F[:, i-1]))

    return U_next

def Godunov(U, U_prim, nx, gamma, dt, dx):
    F = np.zeros(U_prim.shape)
    U_next = np.zeros(U_prim.shape)
    F[0, :] = U_prim[0, :]*U_prim[1, :]
    F[1, :] = U_prim[0, :]*U_prim[1, :]**2 + U_prim[2, :]
    F[2, :] = U_prim[1, :]*(0.5*U_prim[0, :]*U_prim[1, :]**2 + gamma/(gamma-1)*U_prim[2, :])

    for i in range(1, nx-1):
        U_next[:, i] = U[:, i] + dt/dx*(F[:, i] - F[:, i-1])

    return U_next

########################
### Other functions ####
########################

def time_step(U_prim, nx, dx, CFL, gamma):
    u_max = 0.0
    for i in range(nx):
        u_loc = abs(U_prim[1, i]) + m.sqrt(abs(gamma*U_prim[2, i]/U_prim[0, i]))
        if u_loc > u_max:
            u_max = u_loc

    dt = dx*CFL/u_max
    return dt

def bc(U, nx, type):
    """ Type 1 : transmission boundary conditions, Type 2 : periodic"""
    if type == 2:
        U[:, 0] = U[:, nx-2]
        U[:, nx-1] = U[:, 2]
    else:
        U[:, 0] = U[:, 1]
        U[:, nx-1] = U[:, nx-2]
    return U

def variables(U_prim, gamma):
    """ Integration variables [ rho, rho*u, 0.5*rho*u^2 + p/(gamma-1)]"""
    U = np.zeros(U_prim.shape)
    U[0, :] = U_prim[0, :]
    U[1, :] = U_prim[0, :]*U_prim[1, :]
    U[2, :] = 0.5*U_prim[0, :]*(U_prim[1, :]**2) + U_prim[2, :]/(gamma - 1)
    return U

def primitive(U, gamma):
    """ Primitive variables [ rho, u, p] """
    U_prim = np.zeros(U.shape)
    U_prim[0, :] = U[0, :]
    U_prim[1, :] = U[1, :]/U[0, :]
    U_prim[2, :] = (gamma-1)*(U[2, :] - 0.5*(U[1, :]**2)/U[0, :])
    return U_prim

def Jacobian(U_prim, nx, gamma):
    J = np.zeros((3,3,nx))
    rho = U_prim[0, :]
    u = U_prim[1, :]
    p = U_prim[2, :]
    for i in range(nx):
        c = m.sqrt(gamma*p[i]/rho[i])
        J[0, 0, i] = 1
        J[0, 1, i] = 0.5*rho[i]/c
        J[0, 2, i] = - J[0, 1, i]

        J[1, 0, i] = u[i]
        J[1, 1, i] = J[0, 1, i] * (u[i] - c)
        J[1, 2, i] = -J[0, 1, i] * (u[i] - c)

        J[2, 0, i] = 0.5*u[i]*u[i]
        J[2, 1, i] = J[0, 1, i]*( 0.5*u[i]*u[i] + c*c/(gamma-1) + c*u[i])
        J[2, 2, i] = -J[0, 1, i] * (0.5 * u[i] * u[i] + c * c / (gamma - 1) - c * u[i])
    return J