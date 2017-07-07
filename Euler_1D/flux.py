import numpy as np
import functions as f


def flux_cc(density, momentum, energy, gamma):
    velocity = momentum / density
    kinetic_energy = 0.5 * momentum**2 / density
    internal_energy = energy - kinetic_energy
    pressure = (gamma-1)*internal_energy

    flux_density = density * velocity
    flux_momentum = momentum * velocity + pressure
    flux_energy = (energy + pressure) * velocity

    return flux_density, flux_momentum, flux_energy


def flux_fc(density, momentum, energy, gamma):
    density_l = density[:-1]
    density_r = density[1:]

    momentum_l = momentum[:-1]
    momentum_r = momentum[1:]

    energy_l = energy[:-1]
    energy_r = energy[1:]

    velocity_l = momentum_l / density_l
    velocity_r = momentum_r / density_r

    kinetic_energy_l = 0.5 * momentum_l**2 / density_l
    kinetic_energy_r = 0.5 * momentum_r**2 / density_r

    internal_energy_l = energy_l - kinetic_energy_l
    internal_energy_r = energy_r - kinetic_energy_r

    pressure_l = (gamma - 1) * internal_energy_l
    pressure_r = (gamma - 1) * internal_energy_r

    flux_density_l = density_l * velocity_l
    flux_density_r = density_r * velocity_r
    flux_density = 0.5 * (flux_density_l + flux_density_r)

    flux_momentum_l = pressure_l + momentum_l * velocity_l
    flux_momentum_r = pressure_r + momentum_r * velocity_r
    flux_momentum = 0.5 * (flux_momentum_l + flux_momentum_r)

    flux_energy_l = (energy_l + pressure_l) * velocity_l
    flux_energy_r = (energy_r + pressure_r) * velocity_r
    flux_energy = 0.5 * (flux_energy_l + flux_energy_r)

    return flux_density, flux_momentum, flux_energy


def jacobian_primitive(density, velocity, pressure, sound_speed):
    A = np.array([[velocity, density, 0],
                 [0, velocity, 1./density],
                 [0, density * sound_speed * sound_speed, velocity]])

    return A


def jacobian_conserved(density, momentum, energy, gamma):
    assert(isinstance(density, float)), "density type is not float"
    assert(isinstance(momentum, float)), "momentum type is not float"
    assert(isinstance(energy, float)), "energy type is not float"

    velocity = momentum / density

    A10 = -0.5 * (3 - gamma) * velocity * velocity
    A11 = (3 - gamma) * velocity
    A12 = (gamma - 1)
    A20 = (- gamma * energy * velocity / density) + (gamma - 1) * (velocity**3)
    A21 = (gamma * energy / density) - 1.5 * (gamma - 1) * velocity * velocity
    A22 = gamma * velocity

    A = np.array([[0, 1, 0], [A10, A11, A12], [A20, A21, A22]])

    return A


def eigenvalues(velocity, soundspeed):

    return np.array([velocity - soundspeed, velocity, velocity + soundspeed])


def left_eigenvectors(density, sound_speed):
    L1 = np.array([0, 1, - 1. / (density * sound_speed)])
    L2 = np.array([1, 0, - 1. / (sound_speed * sound_speed)])
    L3 = np.array([0, 1, 1./ (density * sound_speed)])

    return np.array([L1, L2, L3])


def right_eigenvectors(density, sound_speed):

    K1 = np.transpose([1, - sound_speed / density, sound_speed * sound_speed])
    K2 = np.transpose([1, 0, 0])
    K3 = np.transpose([1, sound_speed / density, sound_speed * sound_speed])

    return np.array([K1, K2, K3])


def basic(density, momentum, energy, ghost_nx, nx, gamma, dt, dx):
    density_new = np.zeros(nx)
    momentum_new = np.zeros(nx)
    energy_new = np.zeros(nx)
    [flux_density, flux_momentum, flux_energy] = flux_fc(density, momentum, energy, gamma)

    for i in range(ghost_nx, nx - ghost_nx):
        density_new[i] = density[i] - (dt/dx) * (flux_density[i] - flux_density[i-1])
        momentum_new[i] = momentum_new[i] - (dt/dx) * (flux_momentum[i] - flux_momentum[i-1])
        energy_new[i] = energy_new[i] - (dt/dx) * (flux_energy[i] - flux_energy[i-1])

    return density_new, momentum_new, energy_new


def LaxFriedrich(density, momentum, energy, ghost_nx, nx, gamma, dt, dx):
    density_new = np.zeros(len(density))
    momentum_new = np.zeros(len(density))
    energy_new = np.zeros(len(density))

    # FACE CENTERED - STAGGERED
    # [flux_density, flux_momentum, flux_energy] = flux_fc(density, momentum, energy, gamma)
    # for i in range(ghost_nx, nx - ghost_nx):
    #     density_new[i] = 0.5 * (density[i + 1] + density[i - 1]) - (dt/dx) * (flux_density[i] - flux_density[i-1])
    #     momentum_new[i] = 0.5 * (momentum[i + 1] + momentum[i - 1]) - (dt/dx) * (flux_momentum[i] - flux_momentum[i-1])
    #     energy_new[i] = 0.5 * (energy[i + 1] + energy[i - 1]) - (dt/dx) * (flux_energy[i] - flux_energy[i-1])

    # CELL CENTERED - NON STAGGERED
    [flux_density, flux_momentum, flux_energy] = flux_cc(density, momentum, energy, gamma)
    for i in range(ghost_nx, nx - ghost_nx):
        density_new[i] = 0.5 * (density[i + 1] + density[i - 1]) - 0.5 * (dt / dx) * (flux_density[i + 1] - flux_density[i - 1])
        momentum_new[i] = 0.5 * (momentum[i + 1] + momentum[i - 1]) - 0.5 * (dt / dx) * (flux_momentum[i + 1] - flux_momentum[i - 1])
        energy_new[i] = 0.5 * (energy[i + 1] + energy[i - 1]) - 0.5 * (dt / dx) * (flux_energy[i + 1] - flux_energy[i - 1])

    # # CONSERVATIVE FORMULATION
    # [flux_density, flux_momentum, flux_energy] = flux_cc(density, momentum, energy, gamma)
    # for i in range(ghost_nx, nx - ghost_nx):
    #     flux_density_left = 0.5 * (flux_density[i-1] + flux_density[i]) - 0.5 * (dx / dt) * (density[i] - density[i-1])
    #     flux_density_right = 0.5 * (flux_density[i] + flux_density[i+1]) - 0.5 * (dx / dt) * (density[i+1] - density[i])
    #     density_new[i] = density[i] - (dt / dx) * (flux_density_right - flux_density_left)
    #
    #     flux_momentum_left = 0.5 * (flux_momentum[i-1] + flux_momentum[i]) - 0.5 * (dx / dt) * (momentum[i] - momentum[i-1])
    #     flux_momentum_right = 0.5 * (flux_momentum[i] + flux_momentum[i+1]) - 0.5 * (dx / dt) * (momentum[i+1] - momentum[i])
    #     momentum_new[i] = momentum[i] - (dt / dx) * (flux_momentum_right - flux_momentum_left)
    #
    #     flux_energy_left = 0.5 * (flux_energy[i-1] + flux_energy[i]) - 0.5 * (dx / dt) * (energy[i] - energy[i-1])
    #     flux_energy_right = 0.5 * (flux_energy[i] + flux_energy[i+1]) - 0.5 * (dx / dt) * (energy[i+1] - energy[i])
    #     energy_new[i] = energy[i] - (dt / dx) * (flux_energy_right - flux_energy_left)

    return density_new, momentum_new, energy_new


def MacCormack(density, momentum, energy, ghost_nx, nx, gamma, dt, dx):
    density_new = np.zeros(nx)
    momentum_new = np.zeros(nx)
    energy_new = np.zeros(nx)
    density_predictor = np.zeros(nx)
    momentum_predictor = np.zeros(nx)
    energy_predictor = np.zeros(nx)

    # (1) prediction step
    [flux1_density, flux1_momentum, flux1_energy] = flux_cc(density, momentum, energy, gamma)
    for i in range(ghost_nx, nx - ghost_nx):
        density_predictor[i] = density[i] - (dt/dx) * (flux1_density[i] - flux1_density[i-1])
        momentum_predictor[i] = momentum[i] - (dt/dx) * (flux1_momentum[i] - flux1_momentum[i-1])
        energy_predictor[i] = energy[i] - (dt/dx) * (flux1_energy[i] - flux1_energy[i-1])

    density_predictor = f.boundary_conditions(density_predictor, ghost_nx, nx)
    momentum_predictor = f.boundary_conditions(momentum_predictor, ghost_nx, nx)
    energy_predictor = f.boundary_conditions(energy_predictor, ghost_nx, nx)

    # (2) corrector step
    [flux2_density, flux2_momentum, flux2_energy] = flux_cc(density_predictor, momentum_predictor, energy_predictor, gamma)
    for i in range(ghost_nx, nx - ghost_nx):
        density_new[i] = 0.5 * (density[i] + density_predictor[i]) - 0.5 * (dt/dx) * (flux2_density[i+1] - flux2_density[i])
        momentum_new[i] = 0.5 * (momentum[i] + momentum_predictor[i]) - 0.5 * (dt/dx) * (flux2_momentum[i+1] - flux2_momentum[i])
        energy_new[i] = 0.5 * (energy[i] + energy_predictor[i]) - 0.5 * (dt/dx) * (flux2_energy[i+1] - flux2_energy[i])

    return density_new, momentum_new, energy_new


def RichtMyer(density, momentum, energy, ghost_nx, nx, gamma, dt, dx):
    density_new = np.zeros(nx)
    momentum_new = np.zeros(nx)
    energy_new = np.zeros(nx)
    density_predictor = np.zeros(nx)
    momentum_predictor = np.zeros(nx)
    energy_predictor = np.zeros(nx)

    # (1) prediction step
    [flux1_density, flux1_momentum, flux1_energy] = flux_cc(density, momentum, energy, gamma)
    for i in range(ghost_nx, nx - ghost_nx):
        density_predictor[i] = 0.5 * (density[i + 1] + density[i] - (dt/dx) * (flux1_density[i+1] - flux1_density[i]))
        momentum_predictor[i] = 0.5 * (momentum[i + 1] + momentum[i] - (dt/dx) * (flux1_momentum[i+1] - flux1_momentum[i]))
        energy_predictor[i] = 0.5 * (energy[i + 1] + energy[i] - (dt/dx) * (flux1_energy[i+1] - flux1_energy[i]))

    density_predictor = f.boundary_conditions(density_predictor, ghost_nx, nx)
    momentum_predictor = f.boundary_conditions(momentum_predictor, ghost_nx, nx)
    energy_predictor = f.boundary_conditions(energy_predictor, ghost_nx, nx)

    # (2) corrector step
    [flux2_density, flux2_momentum, flux2_energy] = flux_cc(density_predictor, momentum_predictor, energy_predictor, gamma)
    for i in range(ghost_nx, nx - ghost_nx):
        density_new[i] = density[i] - (dt/dx) * (flux2_density[i] - flux2_density[i-1])
        momentum_new[i] = momentum[i] - (dt/dx) * (flux2_momentum[i] - flux2_momentum[i-1])
        energy_new[i] = energy[i] - (dt/dx) * (flux2_energy[i] - flux2_energy[i-1])

    return density_new, momentum_new, energy_new


def Nessyahu_Tadmor(density, momentum, energy, ghost_nx, nx, gamma, dt, dx):

    density, velocity, pressure = f.return_primitive(density, momentum, energy, gamma)
    sound_speed = f.sound_speed(density, momentum, energy, gamma)

    density_new = np.zeros(nx)
    momentum_new = np.zeros(nx)
    energy_new = np.zeros(nx)

    for i in range(ghost_nx, nx - ghost_nx):
        W = np.array([density[i], momentum[i], energy[i]])
        W_m = np.array([density[i-1], momentum[i-1], energy[i-1]])
        W_m2 = np.array([density[i-2], momentum[i-2], energy[i-2]])
        W_p = np.array([density[i+1], momentum[i+1], energy[i+1]])
        W_p2 = np.array([density[i+2], momentum[i+2], energy[i+2]])

        W_deriv_m = f.min_mod3((W - W_m), (W_m - W_m2), (0.5 * (W + W_m2)))
        W_deriv_p = f.min_mod3((W_p2 - W_p), (W_p - W), (0.5 * (W + W_p2)))

        # # 1
        A_p = jacobian_conserved(W_p[0], W_p[1], W_p[2], gamma)
        A_m = jacobian_conserved(W_m[0], W_m[1], W_m[2], gamma)
        f_deriv_p = np.dot(A_p, W_deriv_p)
        f_deriv_m = np.dot(A_m, W_deriv_m)

        # 2
        # [rp2, mp2, ep2] = flux_cc(W_p2[0], W_p2[1], W_p2[2], gamma)
        # [rp, mp, ep] = flux_cc(W_p[0], W_p[1], W_p[2], gamma)
        # [r, m, e] = flux_cc(W[0], W[1], W[2], gamma)
        # [rm, mm, em] = flux_cc(W_m[0], W_m[1], W_m[2], gamma)
        # [rm2, mm2, em2] = flux_cc(W_m2[0], W_m2[1], W_m2[2], gamma)
        # f_del_p2 = np.array([rp2, mp2, ep2]) - np.array([rp, mp, ep])
        # f_del_p = np.array([rp, mp, ep]) - np.array([r, m, e])
        # f_del_m = np.array([r, m, e]) - np.array([rm, mm, em])
        # f_del_m2 = np.array([rm, mm, em]) - np.array([rm2, mm2, em2])
        # f_deriv_p = np.array([f.min_mod(f_del_p2[j], f_del_p[j], 0.5*(f_del_p2[j] + f_del_p[j])) for j in range(3)])
        # f_deriv_m = np.array([f.min_mod(f_del_m2[j], f_del_m[j], 0.5*(f_del_m2[j] + f_del_m[j])) for j in range(3)])

        W_halfstep_p = W_p - 0.5 * (dt / dx) * f_deriv_p
        W_halfstep_m = W_m - 0.5 * (dt / dx) * f_deriv_m

        f_p = np.array(flux_cc(W_halfstep_p[0], W_halfstep_p[1], W_halfstep_p[2], gamma))
        f_m = np.array(flux_cc(W_halfstep_m[0], W_halfstep_m[1], W_halfstep_m[2], gamma))

        W_new = 0.5 * (W_p + W_m) + 0.25 *(W_deriv_p + W_deriv_m) - 0.5 * (dt / dx) * (f_p - f_m)

        density_new[i] = W_new[0]
        momentum_new[i] = W_new[1]
        energy_new[i] = W_new[2]

    return density_new, momentum_new, energy_new


def Nessyahu_Tadmor2(density, momentum, energy, ghost_nx, nx, gamma, dt, dx):

    density, velocity, pressure = f.return_primitive(density, momentum, energy, gamma)
    sound_speed = f.sound_speed(density, momentum, energy, gamma)

    density_new = np.zeros(nx)
    momentum_new = np.zeros(nx)
    energy_new = np.zeros(nx)

    for i in range(ghost_nx, nx - ghost_nx):
        W = np.array([density[i], velocity[i], pressure[i]])
        W_m = np.array([density[i-1], velocity[i-1], pressure[i-1]])
        W_m2 = np.array([density[i-2], velocity[i-2], pressure[i-2]])
        W_p = np.array([density[i+1], velocity[i+1], pressure[i+1]])
        W_p2 = np.array([density[i+2], velocity[i+2], pressure[i+2]])

        W_deriv_p = f.min_mod3((W_p2 - W_p), (W_p - W))
        W_deriv_m = f.min_mod3((W - W_m), (W - W_m2))

        A_p = jacobian_primitive(W_p[0], W_p[1], W_p[2], sound_speed[i+1])
        A_m = jacobian_primitive(W_m[0], W_m[1], W_m[2], sound_speed[i-1])

        f_deriv_p = np.dot(A_p, W_deriv_p)
        f_deriv_m = np.dot(A_m, W_deriv_m)

        W_halfstep_p = W_p - 0.5 * (dt / dx) * f_deriv_p
        W_halfstep_m = W_m - 0.5 * (dt / dx) * f_deriv_m

        f_p = np.array(flux_cc(W_halfstep_p[0], W_halfstep_p[1], W_halfstep_p[2], gamma))
        f_m = np.array(flux_cc(W_halfstep_m[0], W_halfstep_m[1], W_halfstep_m[2], gamma))

        W_new = 0.5 * (W_p + W_m) + 0.25 * (W_deriv_p + W_deriv_m) - 0.5 * (dt / dx) * (f_p - f_m)

        [rho_new, mom_new, en_new] = f.return_conserved(W_new[0], W_new[1], W_new[2], gamma)

        density_new[i] = rho_new
        momentum_new[i] = mom_new
        energy_new[i] = en_new

    return density_new, momentum_new, energy_new


def Nessyahu_Tadmor_non_staggered(density, momentum, energy, ghost_nx, nx, gamma, dt, dx):
    density_new = np.zeros(nx)
    momentum_new = np.zeros(nx)
    energy_new = np.zeros(nx)

    def get_w_dash(w_jp1, w_j, w_jm1):
        del_1 = w_jp1 - w_j
        del_2 = w_j - w_jm1

        return(f.min_mod3(del_1, 0.5 * (del_1 + del_2), del_2))

    def get_flux_w_nph(w_n):
        A = jacobian_conserved(w_n[0], w_n[1], w_n[2], gamma)
        f = np.dot(A, w_n)
        w_nph = w_n - 0.5 * (dt / dx) * f
        [f0, f1, f2] = flux_cc(w_nph[0], w_nph[1], w_nph[2], gamma)

        return np.array([f0, f1, f2])

    def term1():
        return 0.25 * (W_n_jm1 + 2 * W_n_j + W_n_jp1)

    def term2():
        w_dash_jp1 = get_w_dash(W_n_jp2, W_n_jp1, W_n_j)
        w_dash_jm1 = get_w_dash(W_n_j, W_n_jm1, W_n_jm2)

        return -0.25 * 0.25 * (w_dash_jp1 - w_dash_jm1)

    def term3():
        A_jp1 = jacobian_conserved(W_n_jp1[0], W_n_jp1[1], W_n_jp1[2], gamma)
        A_jm1 = jacobian_conserved(W_n_jm1[0], W_n_jm1[1], W_n_jm1[2], gamma)

        f_dash_jp1 = np.dot(A_jp1, W_n_jp1)
        f_dash_jm1 = np.dot(A_jm1, W_n_jm1)

        w_jp1 = W_n_jp1 - 0.5 * (dt / dx) * f_dash_jp1
        w_jm1 = W_n_jm1 - 0.5 * (dt / dx) * f_dash_jm1

        [fp0, fp1, fp2] = flux_cc(w_jp1[0], w_jp1[1], w_jp1[2], gamma)
        [fm0, fm1, fm2] = flux_cc(w_jm1[0], w_jm1[1], w_jm1[2], gamma)

        return -0.5 * (dt / dx) * (np.array([fp0, fp1, fp2]) - np.array([fm0, fm1, fm2]))

    def term4():
        w_dash_jp2 = get_w_dash(W_n_jp3, W_n_jp2, W_n_jp1)
        w_dash_jp1 = get_w_dash(W_n_jp2, W_n_jp1, W_n_j)
        w_dash_j = get_w_dash(W_n_jp1, W_n_j, W_n_jm1)
        w_dash_jm1 = get_w_dash(W_n_j, W_n_jm1, W_n_jm2)
        w_dash_jm2 = get_w_dash(W_n_jm1, W_n_jm2, W_n_jm3)

        f_nph_jm2 = get_flux_w_nph(W_n_jm2)
        f_nph_jm1 = get_flux_w_nph(W_n_jm1)
        f_nph_j = get_flux_w_nph(W_n_j)
        f_nph_jp1 = get_flux_w_nph(W_n_jp1)
        f_nph_jp2 = get_flux_w_nph(W_n_jp2)

        del_w_jp1 = 0.5 * (W_n_jp2 - W_n_j) \
                    - 0.125 * (w_dash_jp2 - 2 * w_dash_jp1 + w_dash_j) \
                    - (dt / dx) * (f_nph_jp2 - 2 * f_nph_jp1 + f_nph_j)

        del_w_j = 0.5 * (W_n_jp1 - W_n_jm1) \
                  - 0.125 * (w_dash_jp1 - 2 * w_dash_j + w_dash_jm1) \
                  - (dt / dx) * (f_nph_jm1 - 2 * f_nph_j + f_nph_jp1)

        del_w_jm1 = 0.5 * (W_n_j - W_n_jm2) \
                    - 0.125 * (w_dash_j - 2 * w_dash_jm1 + w_dash_jm2) \
                    - (dt / dx) * (f_nph_j - 2 * f_nph_jm1 + f_nph_jm2)

        w_dash_jmh = f.min_mod2(del_w_j, del_w_jm1)
        w_dash_jph = f.min_mod2(del_w_jp1, del_w_j)

        return -0.125 * (w_dash_jph - w_dash_jmh)

    for j in range(ghost_nx, nx - ghost_nx):
        W_n_j = np.array([density[j], momentum[j], energy[j]])
        W_n_jm1 = np.array([density[j-1], momentum[j-1], energy[j-1]])
        W_n_jm2 = np.array([density[j-2], momentum[j-2], energy[j-2]])
        W_n_jm3 = np.array([density[j-3], momentum[j-3], energy[j-3]])
        W_n_jp1 = np.array([density[j+1], momentum[j+1], energy[j+1]])
        W_n_jp2 = np.array([density[j+2], momentum[j+2], energy[j+2]])
        W_n_jp3 = np.array([density[j+3], momentum[j+3], energy[j+3]])

        W_np1_j = term1() + term2() + term3() + term4()

        density_new[j] = W_np1_j[0]
        momentum_new[j] = W_np1_j[1]
        energy_new[j] = W_np1_j[2]

    return density_new, momentum_new, energy_new
