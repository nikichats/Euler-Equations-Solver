import initial_conds as ic


# find p in star regions
def pressure_star(p_l, p_r, rho_l, rho_r, u_l, u_r, cs_l, cs_r, gamma):
    tolerance = 1e-6
    relative_error = 1.0

    p_star = 0.0
    iter = 0
    p_PV = 0.5 * (p_l + p_r) - 0.8 * (u_r - u_l) * (rho_l + rho_r) * (cs_l + cs_r)
    p_star_next = max(tolerance, p_PV)
    # p_star_next = 0.5 * (p_l + p_r)

    while relative_error > tolerance:
        if p_star_next < 0.0:
            p_star = tolerance
        else:
            p_star = p_star_next

        [f_p_l, f_p_deriv_l] = pressure_functions(p_star, rho_l, p_l, cs_l, gamma)
        [f_p_r, f_p_deriv_r] = pressure_functions(p_star, rho_r, p_r, cs_r, gamma)
        f_p = f_p_l + f_p_r + (u_r - u_l)
        f_p_deriv = f_p_deriv_l + f_p_deriv_r

        p_star_next = p_star - (f_p / f_p_deriv)

        relative_error = 2. * abs(p_star_next - p_star) / abs(p_star_next + p_star)
        iter = iter + 1
        print(relative_error)

    return p_star


def pressure_functions(p, rho_lr, p_lr, cs_lr, gamma):
    gamma_2p = (gamma + 1) / (2 * gamma)
    gamma_2m = (gamma - 1) / (2 * gamma)
    A = 2. / ((gamma + 1) * rho_lr)
    B = rho_lr * (gamma - 1) / (gamma + 1)

    if p > p_lr:
        f_p = (p - p_lr) * (A / (p + B))**0.5
        f_p_deriv = ((A / (B + p))**0.5) * (1 - ((p - p_lr) * 0.5 / (B + p)))
    else:
        f_p = (2 * cs_lr / (gamma - 1)) * (((p / p_lr)**gamma_2m) - 1)
        f_p_deriv = (1. / (rho_lr * cs_lr)) * (p / p_lr)**(-gamma_2p)

    return f_p, f_p_deriv


# find u in star region
def velocity_star(u_l, u_r, p_star, p_l, p_r, rho_l, rho_r, cs_l, cs_r, gamma):
    [f_p_l, f_p_deriv_l] = pressure_functions(p_star, rho_l, p_l, cs_l, gamma)
    [f_p_r, f_p_deriv_l] = pressure_functions(p_star, rho_r, p_r, cs_r, gamma)
    u_star = 0.5 * (u_l + u_r) + 0.5 * (f_p_r - f_p_l)

    return u_star


# find left and right densities in star region
def density_star(p_star, p_l, p_r, rho_l, rho_r, gamma):
    gamma_pm1 = (gamma - 1) / (gamma + 1)

    if p_star > p_l:
        rho_star_l = rho_l * ( (p_star / p_l) + gamma_pm1) / (gamma_pm1 * (p_star / p_l) + 1)
    elif p_star < p_l:
        rho_star_l = rho_l * (p_star / p_l)**(1./gamma)
    else:
        rho_star_l = rho_l

    if p_star > p_r:
        rho_star_r = rho_r * ( (p_star / p_r) + gamma_pm1) / (gamma_pm1 * (p_star / p_r) + 1)
    elif p_star < p_r:
        rho_star_r = rho_r * (p_star / p_r)**(1./gamma)
    else:
        rho_star_r = rho_r

    return rho_star_l, rho_star_r


# find shock speed
def shock_speed(p_star, u_star, p_l, p_r, rho_l, rho_r, cs_l, cs_r, gamma):
    u_shock_l = u_shock_r = u_head_l = u_head_r = u_tail_l = u_tail_r = 0.0

    gamma_2p = (gamma + 1) / (2. * gamma)
    gamma_2m = (gamma - 1) / (2. * gamma)
    cs_star_l = cs_l * (p_star / p_l)**(1./gamma)
    cs_star_r = cs_r * (p_star / p_r)**(1./gamma)

    if p_star > p_l:
        u_shock_l = u_l - (gamma_2p * (p_star / p_l) + gamma_2m)**0.5
    else:
        u_head_l = u_l - cs_l
        u_tail_l = u_star - cs_star_l
    if p_star > p_r:
        u_shock_r = u_r - (gamma_2p * (p_star / p_r) + gamma_2m)**0.5
    else:
        u_head_r = u_r - cs_r
        u_tail_r = u_star + cs_star_r

    return u_shock_l , u_shock_r , u_head_l , u_head_r , u_tail_l , u_tail_r


gamma = 1.4
TEST = 5
[rho_l, rho_r, u_l, u_r, p_l, p_r, cs_l, cs_r] = ic.left_right_states(gamma, TEST)
print("initial states")
print(rho_l, rho_r, u_l, u_r, p_l, p_r)

p_star = pressure_star(p_l, p_r, rho_l, rho_r, u_l, u_r, cs_l, cs_r, gamma)
u_star = velocity_star(u_l, u_r, p_star, p_l, p_r, rho_l, rho_r, cs_l, cs_r, gamma)
[rho_star_l, rho_star_r] = density_star(p_star, p_l, p_r, rho_l, rho_r, gamma)
shock_speeds = shock_speed(p_star, u_star, p_l, p_r, rho_l, rho_r, cs_l, cs_r, gamma)

print("p* = " + str(p_star))
print("v* = " + str(u_star))
print("rho*_l = " + str(rho_star_l))
print("rho*_r = " + str(rho_star_r))
