import numpy as np

def LaxF(U, U_prim, nx, gamma, dt, dx):
    F = np.zeros(U_prim.shape)
    U_next = np.zeros(U_prim.shape)
    F[0, :] = U_prim[0, :]*U_prim[1, :]
    F[1, :] = U_prim[0, :]*U_prim[1, :]**2 + U_prim[2, :]
    F[2, :] = U_prim[1, :]*(0.5*U_prim[0, :]*U_prim[1, :]**2 + gamma/(gamma-1)*U_prim[2, :])

    for i in range(1, nx-1):
        U_next[:, i] = 0.5*(U[:, i-1] + U[:, i+1]) - 0.5*dt/dx*(F[:, i+1] - F[:, i-1])

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
