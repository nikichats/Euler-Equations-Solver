import numpy as np


def initial_cond(x, rho_L, rho_R, u_L, u_R, p_L, p_R, gamma, nx, dx, shock_position):
    density_0 = np.zeros(nx)
    momentum_0 = np.zeros(nx)
    energy_0 = np.zeros(nx)
    velocity_0 = np.zeros(nx)
    pressure_0 = np.zeros(nx)

    for i in range(nx):
        if x[i] < shock_position:
            density_0[i] = rho_L
            momentum_0[i] = rho_L*u_L
            energy_0[i] = rho_L*p_L/(gamma-1) + 0.5*rho_L*u_L**2
            velocity_0[i] = u_L
            pressure_0[i] = p_L
        else:
            density_0[i] = rho_R
            momentum_0[i] = rho_R*u_R
            energy_0[i] = rho_R*p_R/(gamma-1) + 0.5*rho_R*u_R**2
            velocity_0[i] = u_R
            pressure_0[i] = p_R

    return density_0, momentum_0, energy_0, velocity_0, pressure_0
