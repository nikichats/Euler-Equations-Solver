import numpy as np
import math as m

def left_right_states(gamma, test):

    if test == 2:
        x_0 = 0.5
        rho_l = 1.
        rho_r = 1.0
        u_l = -2.0
        u_r = 2.0
        p_l = 0.4
        p_r = 0.4
        a_l = m.sqrt(gamma*p_l/rho_l)
        a_r = m.sqrt(gamma*p_r/rho_r)

    elif test == 3:
        x_0 = 0.5
        rho_l = 1.0
        rho_r = 1.0
        u_l = 0.0
        u_r = 0.0
        p_l = 1000.0
        p_r = 0.01
        a_l = m.sqrt(gamma * p_l / rho_l)
        a_r = m.sqrt(gamma * p_r / rho_r)

    elif test == 4:
        x_0 = 0.5
        rho_l = 1.0
        rho_r = 1.0
        u_l = 0.0
        u_r = 0.0
        p_l = 0.01
        p_r = 100.0
        a_l = m.sqrt(gamma * p_l / rho_l)
        a_r = m.sqrt(gamma * p_r / rho_r)

    elif test == 5:
        x_0 = 0.5
        rho_l = 5.99924
        rho_r = 5.99242
        u_l = 19.5975
        u_r = -6.19633
        p_l = 460.894
        p_r = 46.095
        a_l = m.sqrt(gamma * p_l / rho_l)
        a_r = m.sqrt(gamma * p_r / rho_r)

    else:
        x_0 = 0.3
        rho_l = 1.
        rho_r = 0.125
        u_l = 0.75
        u_r = 0.0
        p_l = 1.0
        p_r = 0.1
        a_l = m.sqrt(gamma*p_l/rho_l)
        a_r = m.sqrt(gamma*p_r/rho_r)

    return x_0, rho_l, rho_r, u_l, u_r, p_l, p_r, a_l, a_r


def time_out(test):
    if test == 2:
        t_out = 0.15
    elif test == 3:
        t_out = 0.012
    elif test == 4:
        t_out = 0.035
    elif test == 5:
        t_out = 0.035
    else:
        t_out = 0.25

    return t_out


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
