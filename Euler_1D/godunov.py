import numpy as np
import matplotlib.pyplot as plt
import initial_conds as ic
import functions as f
import reconstruction as r
import flux
import riemann

TEST = 1
gamma = 1.4
CFL = 0.9
[x_0, rho_l, rho_r, u_l, u_r, p_l, p_r, a_l, a_r] = ic.left_right_states(gamma, TEST)
t_out = ic.time_out(TEST)

domain_nx = 100
ghost_nx = 2
nx = domain_nx + 2 * ghost_nx
x_min = 0.
x_max = 1.
dx = (x_max - x_min) / domain_nx
shock_position = x_0 * (x_max + x_min) + x_min

x = np.linspace(x_min - ghost_nx * dx, x_max + ghost_nx * dx, nx)
[density, momentum, energy, velocity, pressure] = ic.initial_cond(x, rho_l, rho_r, u_l, u_r, p_l, p_r, gamma, nx , shock_position)
sound_speed = np.zeros(nx + 1)
length = nx + 1

time = 0.0
time_step = 0
# f.write(x, density, momentum, energy, time_step, 5, gamma, CFL, 0.0)

while time < t_out:
    # CALCULATE TIME STEP
    dt = f.time_step(density, momentum, energy, ghost_nx, nx, dx, CFL, gamma)

    # TIME EVOLUTION
    if dt < 1e-7:
        break
    time = time + dt
    time_step = time_step + 1

    sound_speed = np.sqrt(gamma * pressure / density)

    density_flux_left = np.zeros(nx - 2 * ghost_nx)
    density_flux_right = np.zeros(nx - 2 * ghost_nx)
    momentum_flux_left = np.zeros(nx - 2 * ghost_nx)
    momentum_flux_right = np.zeros(nx - 2 * ghost_nx)
    energy_flux_left = np.zeros(nx - 2 * ghost_nx)
    energy_flux_right = np.zeros(nx - 2 * ghost_nx)

    for i in range(ghost_nx, nx - ghost_nx):
        j = i - ghost_nx

        # RECONSTRUCTION
        density_left_l, density_left_r, density_right_l, density_right_r = r.reconstruction(density, i).constant()
        velocity_left_l, velocity_left_r, velocity_right_l, velocity_right_r = r.reconstruction(velocity, i).constant()
        pressure_left_l, pressure_left_r, pressure_right_l, pressure_right_r = r.reconstruction(pressure, i).constant()
        sound_speed_left_l, sound_speed_left_r, sound_speed_right_l, sound_speed_right_r = r.reconstruction(sound_speed, i).constant()

        # INTERFACE QUANTITIES
        [rho_left, u_left, p_left] = riemann.sample(0.0, dt, density_left_l, density_left_r, velocity_left_l,
                                                    velocity_left_r, pressure_left_l, pressure_left_r,
                                                    sound_speed_left_l, sound_speed_left_r, gamma)

        [rho_right, u_right, p_right] = riemann.sample(0.0, dt, density_right_l, density_right_r, velocity_right_l,
                                                       velocity_right_r, pressure_right_l, pressure_right_r,
                                                       sound_speed_right_l, sound_speed_right_r, gamma)

        # FLUXES OF CONSERVED INTERFACE QUANTITIES
        [rho_left, mom_left, en_left] = f.return_conserved(rho_left, u_left, p_left, gamma)
        [rho_right, mom_right, en_right] = f.return_conserved(rho_right, u_right, p_right, gamma)

        [density_flux_left[j], momentum_flux_left[j], energy_flux_left[j]] = flux.flux_cc(rho_left, mom_left, en_left, gamma)
        [density_flux_right[j], momentum_flux_right[j], energy_flux_right[j]] = flux.flux_cc(rho_right, mom_right, en_right, gamma)

    # UPDATE CONSERVED QUANTITIES
    density[ghost_nx : nx - ghost_nx] += (dt / dx) * (density_flux_left - density_flux_right)
    momentum[ghost_nx : nx - ghost_nx] += (dt / dx) * (momentum_flux_left - momentum_flux_right)
    energy[ghost_nx : nx - ghost_nx] += (dt / dx) * (energy_flux_left - energy_flux_right)

    # BOUNDARY CONDITIONS
    density = f.boundary_conditions(density, ghost_nx, nx)
    momentum = f.boundary_conditions(momentum, ghost_nx, nx)
    energy = f.boundary_conditions(energy, ghost_nx, nx)

    # CONSERVED TO PHYSICAL QUANTITIES
    [density, velocity, pressure] = f.return_primitive(density, momentum, energy, gamma)

    # f.write(x, density, momentum, energy, time_step, 5, gamma, CFL, 0.0)

# OBTAIN ANALYTIC SOLUTION
[x_analytic, density_analytic, velocity_analytic, pressure_analytic, momentum_analytic, energy_analytic] = \
    riemann.analytic_solution(riemann_analytic=True, plot_analytic=False, TEST=TEST)

f.plot_all(x, density, velocity, pressure, momentum, energy, x_analytic, density_analytic, velocity_analytic,
           pressure_analytic, momentum_analytic, energy_analytic, gamma)
