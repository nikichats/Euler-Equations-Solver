import numpy as np
import math as m

def Godunov(U, U_prim, nx, gamma, dt, dx):
    U_next = np.zeros(U_prim.shape)     # nx-2 domain cells, 2 ghost cells
    F = np.zeros((3, nx-1))             # nx-1 flux interfaces
    for i in range(0, nx-1):
        F[0, i] = U[1, i]
        F[1, i] = 0.5*(3 - gamma)*(U[1, i]**2/U[0, i]) + (gamma-1)*U[2, i]
        F[2, i] = gamma*(U[1, i]*U[0, i]/U[2, i]) - 0.5*(gamma-1)*((U[1, i]**3)/(U[1, i]**2))
    for i in range(1, nx-1):
        U_next[:, i] = U[:, i] - (dt/dx)*(F[:, i] - F[:, i-1])

    return U_next

def LaxF(U, U_prim, nx, gamma, dt, dx):
    F = np.zeros(U.shape)
    U_next = np.zeros(U.shape)
    F[0, :] = U_prim[0, :]*U_prim[1, :]
    F[1, :] = U_prim[0, :]*U_prim[1, :]**2 + U_prim[2, :]
    F[2, :] = U_prim[1, :]*(0.5*U_prim[0, :]*U_prim[1, :]**2 + gamma/(gamma-1)*U_prim[2, :])
    for i in range(1, nx-1):
        U_next[:, i] = 0.5*(U[:, i-1] + U[:, i+1]) - 0.5*dt/dx*(F[:, i+1] - F[:, i-1])

    return U_next

def LaxW(U, U_prim, nx, gamma, dt, dx):
    # u_i - 0.5*dt/dx*(F_i+1 - F_i-1) + 0.5*dt/dx^2*0.5*((ui+ - ui)*(Fi+ - Fi) - (ui - ui-)*(Fi - Fi-))
    F = np.zeros(U.shape)
    U_next = np.zeros(U.shape)
    F[0, :] = U_prim[0, :] * U_prim[1, :]
    F[1, :] = U_prim[0, :] * U_prim[1, :] ** 2 + U_prim[2, :]
    F[2, :] = U_prim[1, :] * (0.5 * U_prim[0, :] * U_prim[1, :] ** 2) + gamma / (gamma - 1) * U_prim[2, :]
    for i in range(1, nx-1):
        U_next[:, i] = U[:, i] - 0.5*(dt/dx)*(F[:, i+1] - F[:, i-1]) + ((0.5*dt/dx)**2)*((U[:, i+1] - U[:, i])*(F[:, i+1] - F[:, i])- (U[:, i] - U[:, i-1])*(F[:, i] - F[:, i-1]))

    return U_next

def basic(U, U_prim, nx, gamma, dt, dx):
    U_next = np.zeros(U.shape)
    F = F_physical(U_prim, nx, gamma)       # face centered
    for i in range(1,nx-1):
        # U_next[:, i] = 0.5*(U[:, i+1] -U[:, i-1]) - (dt/dx)*(F[:, i+1] - F[:, i])
        U_next[:, i] = U[:, i]  - (dt/dx)*(F[:, i] - F[:, i-1])

    return U_next

def basic2(U, U_prim, nx, gamma, dt, dx):
    U_next = np.zeros(U.shape)
    dU_prim = np.zeros(U.shape)
    dU_prim_L = np.zeros((3, nx-2))
    dU_prim_R = np.zeros((3, nx-2))
    for i in range(nx-2):
        dU_prim_L[:, i] =  U_prim[:, i+1] - U_prim[:, i]
        dU_prim_R[:, i] = U_prim[:, i+2] - U_prim[:, i+1]

        dU_prim[0, i] = minmod(dU_prim_L[0, i], dU_prim_R[0, i])
        dU_prim[1, i] = minmod(dU_prim_L[1, i], dU_prim_R[1, i])
        dU_prim[2, i] = minmod(dU_prim_L[2, i], dU_prim_R[2, i])

    U_prim_L = U_prim + 0.5*dU_prim
    U_prim_R = U_prim - 0.5*dU_prim

    F = 0.5*(F_physical(U_prim_L, nx, gamma) + F_physical(U_prim_R, nx, gamma))
    for i in range(1,nx-1):
        # U_next[:, i] = 0.5*(U[:, i+1] -U[:, i-1]) - (dt/dx)*(F[:, i+1] - F[:, i])
        U_next[:, i] = U[:, i]  - (dt/dx)*(F[:, i] - F[:, i-1])

    return U_next

########################
### Other functions ####
#######################

def time_step(U_prim, nx, dx, CFL, gamma):
    """ Calculates dt required to satisfy the CFL condition using the maximum wave speed in the cell reference frame"""
    u_max = 0.0
    for i in range(nx):
        u_loc = abs(U_prim[1, i]) + m.sqrt(abs(gamma*U_prim[2, i]/U_prim[0, i])) # u + c_s
        if u_loc > u_max:
            u_max = u_loc

    dt = dx*CFL/u_max
    return dt

def bc(U, nx, type):
    """ Type 1 : transmission boundary conditions, Type 2 : periodic"""
    if type == 1:
        U[:, 0] = U[:, 1]
        U[:, nx-1] = U[:, nx-2]
    elif type == 2:
        U[:, 0] = U[:, nx-2]
        U[:, nx-1] = U[:, 2]

    return U

def variables(U_prim, gamma):
    """ Integration variables [ rho, rho*u, E = 0.5*rho*u^2 + p/(gamma-1)]"""
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

def minmod(a, b):
    minmod = 0.0
    if a == 0 or b == 0:
        minmod = 0.0
    elif abs(a) <= abs(b):
        minmod = a
    else:
        minmod = b
    return minmod

def F_physical(U_prim, nx, gamma):
    F = np.zeros((3, nx-1))
    for i in range(0, nx-1):
        # calculate cell centered values
        rho_L = U_prim[0, i]
        rho_R = U_prim[0, i+1]
        u_L = U_prim[1, i]
        u_R = U_prim[1, i+1]
        p_L = U_prim[2, i]
        p_R = U_prim[2, i+1]
        H_L = 0.5*rho_L*u_L**2 + gamma/(gamma-1)*p_L
        H_R = 0.5*rho_R*u_R**2 + gamma/(gamma-1)*p_R

        # simple averaging between cells F = (F_L + F_R)/2
        F[0, i] = 0.5*( rho_L*u_L + rho_R*u_R )
        F[1, i] = 0.5*((rho_L*u_L**2 + p_L) + (rho_R*u_R**2 + p_R))
        F[2, i] = 0.5*( u_L*H_L + u_R*H_R)
    return F

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
