import numpy as np
import math as m

def left_right_states(gamma, test):

    if test == 2:
        rho_l = 1.
        rho_r = 1.0
        u_l = -2.0
        u_r = 2.0
        p_l = 0.4
        p_r = 0.4
        cs_l = m.sqrt(gamma*p_l/rho_l)
        cs_r = m.sqrt(gamma*p_r/rho_r)

    elif test == 3:
        rho_l = 1.0
        rho_r = 1.0
        u_l = 0.0
        u_r = 0.0
        p_l = 1000.0
        p_r = 0.01
        cs_l = m.sqrt(gamma * p_l / rho_l)
        cs_r = m.sqrt(gamma * p_r / rho_r)

    elif test == 4:
        rho_l = 1.0
        rho_r = 1.0
        u_l = 0.0
        u_r = 0.0
        p_l = 0.01
        p_r = 100.0
        cs_l = m.sqrt(gamma * p_l / rho_l)
        cs_r = m.sqrt(gamma * p_r / rho_r)

    elif test == 5:
        rho_l = 5.99924
        rho_r = 5.99242
        u_l = 19.5975
        u_r = -6.19633
        p_l = 460.894
        p_r = 46.095
        cs_l = m.sqrt(gamma * p_l / rho_l)
        cs_r = m.sqrt(gamma * p_r / rho_r)

    else:
        rho_l = 1.
        rho_r = 0.125
        u_l = 0.0
        u_r = 0.0
        p_l = 1.0
        p_r = 0.1
        cs_l = m.sqrt(gamma*p_l/rho_l)
        cs_r = m.sqrt(gamma*p_r/rho_r)

    return rho_l, rho_r, u_l, u_r, p_l, p_r, cs_l, cs_r


def initial_cond(x, rho_l, rho_r, u_l, u_r, p_l, p_r, gamma, nx, shock_position):
    density_0 = np.zeros(nx)
    momentum_0 = np.zeros(nx)
    energy_0 = np.zeros(nx)
    velocity_0 = np.zeros(nx)
    pressure_0 = np.zeros(nx)

    for i in range(nx):
        if x[i] < shock_position:
            density_0[i] = rho_l
            momentum_0[i] = rho_l*u_l
            energy_0[i] = p_l/(gamma-1) + 0.5*rho_l*u_l**2
            velocity_0[i] = u_l
            pressure_0[i] = p_l
        else:
            density_0[i] = rho_r
            momentum_0[i] = rho_r*u_r
            energy_0[i] = p_r/(gamma-1) + 0.5*rho_r*u_r**2
            velocity_0[i] = u_r
            pressure_0[i] = p_r

    return density_0, momentum_0, energy_0, velocity_0, pressure_0
