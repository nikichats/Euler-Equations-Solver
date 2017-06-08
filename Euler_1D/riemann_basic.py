import numpy as np
import matplotlib.pyplot as plt
import initial_conds as ic
import functions as f


def pressure_functions(p, rho_lr, p_lr, a_lr, gamma):
    gamma_2p = (gamma + 1) / (2 * gamma)
    gamma_2m = (gamma - 1) / (2 * gamma)
    A = 2. / ((gamma + 1) * rho_lr)
    B = p_lr * (gamma - 1) / (gamma + 1)

    if p > p_lr:
        f_p = (p - p_lr) * (A / (p + B))**0.5
        f_p_deriv = ((A / (B + p))**0.5) * (1 - ( 0.5 * (p - p_lr) / (B + p)))
    else:
        f_p = (2 * a_lr / (gamma - 1)) * (((p / p_lr)**gamma_2m) - 1)
        f_p_deriv = (1. / (rho_lr * a_lr)) * (p / p_lr)**(-gamma_2p)

    return f_p, f_p_deriv


def star_states(rho_l, rho_r, u_l, u_r, p_l, p_r, a_l, a_r, gamma):

    # FIND PRESSURE STAR USING NEWTON-RAPHSON ITERATION
    tolerance = 1e-6
    relative_error = 1.0

    p_star = 0.0
    p_PV = 0.5 * (p_l + p_r) - 0.8 * (u_r - u_l) * (rho_l + rho_r) * (a_l + a_r)
    p_star_next = max(tolerance, p_PV)

    while relative_error > tolerance:
        if p_star_next < 0.0:
            p_star = tolerance
        else:
            p_star = p_star_next

        [f_p_l, f_p_deriv_l] = pressure_functions(p_star, rho_l, p_l, a_l, gamma)
        [f_p_r, f_p_deriv_r] = pressure_functions(p_star, rho_r, p_r, a_r, gamma)
        f_p = f_p_l + f_p_r + (u_r - u_l)
        f_p_deriv = f_p_deriv_l + f_p_deriv_r

        p_star_next = p_star - (f_p / f_p_deriv)
        relative_error = 2. * abs(p_star_next - p_star) / abs(p_star_next + p_star)

    # FIND VELOCITY STAR USING OBTAINED PRESSURE STAR VALUE
    [f_p_l, f_p_deriv_l] = pressure_functions(p_star, rho_l, p_l, a_l, gamma)
    [f_p_r, f_p_deriv_l] = pressure_functions(p_star, rho_r, p_r, a_r, gamma)

    u_star = 0.5 * (u_l + u_r) + 0.5 * (f_p_r - f_p_l)

    # FIND LEFT AND RIGHT DENSITY STAR USING OBTAINED PRESSURE STAR VALUE
    gamma_pm1 = (gamma - 1.) / (gamma + 1.)

    if p_star > p_l:    # left shock
        rho_star_l = rho_l * ((p_star / p_l) + gamma_pm1) / (gamma_pm1 * (p_star / p_l) + 1.)

    else:               # left expansion fan
        rho_star_l = rho_l * (p_star / p_l)**(1./gamma)

    if p_star > p_r:    # right shock
        rho_star_r = rho_r * ((p_star / p_r) + gamma_pm1) / (gamma_pm1 * (p_star / p_r) + 1.)

    else:               # right expansion fan
        rho_star_r = rho_r * (p_star / p_r)**(1./gamma)

    return rho_star_l, rho_star_r, u_star, p_star


def fan_states(S, rho_lr, u_lr, p_lr, a_lr, gamma):
    gamma_2p = 2. / (gamma + 1)
    gamma_2m = 2. / (gamma - 1)
    gamma_pm = (gamma - 1) / (gamma + 1)

    rho_fan = rho_lr * (gamma_2p + (gamma_pm / a_lr) * (u_lr - S))**gamma_2m
    u_fan = gamma_2p * (a_lr + (u_lr / gamma_2m) + S)
    p_fan = p_lr * (gamma_2p + (gamma_pm / a_lr) * (u_lr - S))**(gamma_2m * gamma)

    return rho_fan, u_fan, p_fan


def shock_speeds(p_star, u_star, p_l, p_r, rho_l, rho_r, a_l, a_r, gamma):
    S_l = S_r = S_h_l = S_h_r = S_t_l = S_t_r = 0.0

    gamma_2p = (gamma + 1) / (2. * gamma)
    gamma_2m = (gamma - 1) / (2. * gamma)
    a_star_l = a_l * (p_star / p_l)**gamma_2m
    a_star_r = a_r * (p_star / p_r)**gamma_2m

    if p_star > p_l:
        S_l = u_l - a_l*(gamma_2p * (p_star / p_l) + gamma_2m)**0.5
    else:
        S_h_l = u_l - a_l
        S_t_l = u_star - a_star_l

    if p_star > p_r:
        S_r = u_r + a_r*(gamma_2p * (p_star / p_r) + gamma_2m)**0.5
    else:
        S_h_r = u_r + a_r
        S_t_r = u_star + a_star_r

    return S_l , S_r , S_h_l , S_h_r , S_t_l , S_t_r


def sample(x, t, rho_l, rho_r, p_l, p_r, u_l, u_r, a_l, a_r, gamma):
    """ For input values of x and t, return the state variables depending on input variables"""
    S = x/t
    [rho_star_l, rho_star_r, u_star, p_star] = star_states(rho_l, rho_r, u_l, u_r, p_l, p_r, a_l, a_r, gamma)
    [rho_fan_l, u_fan_l, p_fan_l] = fan_states(S, rho_l, u_l, p_l, a_l, gamma)
    [rho_fan_r, u_fan_r, p_fan_r] = fan_states(S, rho_r, u_r, p_r, -a_r, gamma) # note the - a_r
    [S_l , S_r , S_h_l , S_h_r , S_t_l , S_t_r] = shock_speeds(p_star, u_star, p_l, p_r, rho_l, rho_r, a_l, a_r, gamma)

    if S < u_star:
        if p_star > p_l:
            if S < S_l:         # left state (shock)
                W = np.array([rho_l, u_l, p_l])

            else:               # left star state (shock)
                W = np.array([rho_star_l, u_star, p_star])

        else:
            if S < S_h_l:       # left state (fan)
                W = np.array([rho_l, u_l, p_l])

            else:
                if S > S_t_l:   # left state (fan)
                    W = np.array([rho_star_l, u_star, p_star])

                else:           # left fan state (fan)
                    W = np.array([rho_fan_l, u_fan_l, p_fan_l])
    else:
        if p_star > p_r:
            if S > S_r:         # right state (shock)
                W = np.array([rho_r, u_r, p_r])

            else:               # right star state (shock)
                W = np.array([rho_star_r, u_star, p_star])

        else:
            if S > S_h_r:       # right state (fan)
                W = np.array([rho_r, u_r, p_r])

            else:
                if S < S_t_r:   # right star state (fan)
                    W = np.array([rho_star_r, u_star, p_star])

                else:           # right fan state (fan)
                    W = np.array([rho_fan_r, u_fan_r, p_fan_r])

    rho_state = W[0]
    u_state = W[1]
    p_state = W[2]

    return rho_state, u_state, p_state

##############################################################################

TEST = 5
gamma = 1.4
[rho_l, rho_r, u_l, u_r, p_l, p_r, a_l, a_r] = ic.left_right_states(gamma, TEST)
t_out = ic.time_out(TEST)

domain_nx = 100
ghost_nx = 2
nx = domain_nx + 2*ghost_nx
x_min = 0.
x_max = 1.
dx = (x_max - x_min) / domain_nx
shock_position = 0.5 * (x_max + x_min)

x = np.linspace(x_min - ghost_nx * dx, x_max + ghost_nx * dx, nx + 1)
rho = np.zeros(nx + 1)
u = np.zeros(nx + 1)
p = np.zeros(nx + 1)

for i in range(ghost_nx, nx - ghost_nx):
    [rho_i, u_i, p_i] = sample(x[i] - shock_position, t_out, rho_l, rho_r, p_l, p_r, u_l, u_r, a_l, a_r, gamma)
    rho[i] = rho_i
    u[i] = u_i
    p[i] = p_i

stars = star_states(rho_l, rho_r, u_l, u_r, p_l, p_r, a_l, a_r, gamma)
print(stars)

rho = f.boundary_conditions(rho, ghost_nx, nx)
u = f.boundary_conditions(u, ghost_nx, nx)
p = f.boundary_conditions(p, ghost_nx, nx)

plt.plot(x, rho, 'k', x, u, 'b')
plt.show()
