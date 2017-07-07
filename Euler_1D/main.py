import numpy as np

import initial_conds as ic
import functions as f
import flux
import riemann

TEST = 1
gamma = 1.4
CFL = 0.2
solver = 5
epsilon_artificial_viscosity = 0.0

[x_0, rho_l, rho_r, u_l, u_r, p_l, p_r, a_l, a_r] = ic.left_right_states(gamma, TEST)
t_out = ic.time_out(TEST)

domain_nx = 200
ghost_nx = 3
nx = domain_nx + 2 * ghost_nx
x_min = 0.
x_max = 1.
dx = (x_max - x_min) / domain_nx
shock_position = x_0 * (x_max + x_min) + x_min

x = np.linspace(x_min - ghost_nx * dx, x_max + ghost_nx * dx, nx)
[density, momentum, energy, velocity, pressure] = ic.initial_cond(x, rho_l, rho_r, u_l, u_r, p_l, p_r, gamma, nx, shock_position=x_0)

sound_speed = np.zeros(nx + 1)
length = nx + 1

time = 0.0
time_step = 0
# f.write(x, density, momentum, energy, time_step, 5, gamma, CFL, 0.0)

# Solve
while time < t_out:

    # time stepping
    dt = f.time_step(density, momentum, energy, ghost_nx, nx, dx, CFL, gamma)

    if dt < 1e-7:
        break
    time = time + dt
    time_step = time_step + 1

    # # apply artificial viscosity
    # density = f.artificial_viscosity(density, ghost_nx, nx, epsilon_artificial_viscosity)
    # momentum = f.artificial_viscosity(momentum, ghost_nx, nx, epsilon_artificial_viscosity)
    # energy = f.artificial_viscosity(energy, ghost_nx, nx, epsilon_artificial_viscosity)

    # calculate new quantities
    if solver == 2:
        [density_new, momentum_new, energy_new] = flux.LaxFriedrich(density, momentum, energy, ghost_nx, nx, gamma, dt, dx)
    elif solver == 3:
        [density_new, momentum_new, energy_new] = flux.MacCormack(density, momentum, energy, ghost_nx, nx, gamma, dt, dx)
    elif solver == 4:
        [density_new, momentum_new, energy_new] = flux.RichtMyer(density, momentum, energy, ghost_nx, nx, gamma, dt, dx)
    elif solver == 5:
        [density_new, momentum_new, energy_new] = flux.Nessyahu_Tadmor_non_staggered(density, momentum, energy, ghost_nx, nx, gamma, dt, dx)
    else:
        [density_new, momentum_new, energy_new] = flux.basic(density, momentum, energy, ghost_nx, nx, gamma, dt, dx)

    # apply boundary conditions and update states
    density = f.boundary_conditions(density_new, ghost_nx, nx)
    momentum = f.boundary_conditions(momentum_new, ghost_nx, nx)
    energy = f.boundary_conditions(energy_new, ghost_nx, nx)

    print(time, dt)
    # f.write(x, density, momentum, energy, time_step, solver, gamma, CFL, epsilon_artificial_viscosity)

print("final time step = " + str(time_step))

[x_analytic, density_analytic, velocity_analytic, pressure_analytic, momentum_analytic, energy_analytic] = \
    riemann.analytic_solution(riemann_analytic=True, plot_analytic=False, TEST=TEST)

density, velocity, pressure = f.return_primitive(density, momentum, energy, gamma)
f.plot_all(x, density, velocity, pressure, momentum, energy, x_analytic, density_analytic, velocity_analytic,
           pressure_analytic, momentum_analytic, energy_analytic, gamma)
