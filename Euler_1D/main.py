import numpy as np

import initial_conds as ic
import functions as f
import flux

domain_nx = 50
ghost_nx = 2
nx = domain_nx + 2*ghost_nx
x_min = -10.
x_max = 10.
dx = (x_max - x_min) / domain_nx

x = np.linspace(x_min - ghost_nx * dx, x_max + ghost_nx * dx, nx )

CFL = 0.4
gamma = 1.4

solver = 5
epsilon_artificial_viscosity = 0.0

# Initial Conditions
time_max = ic.time_out(test=1)
[rho_l, rho_r, u_l, u_r, p_l, p_r, a_l, a_r] = ic.left_right_states(gamma, test=1)
[density, momentum, energy, velocity, pressure] = ic.initial_cond(x, rho_l, rho_r, u_l, u_r, p_l, p_r, gamma, nx , shock_position=0.0)

time = 0.0; time_step = 0
f.write(x, density, momentum, energy, time_step, solver, gamma, CFL, epsilon_artificial_viscosity)

# Solve
while time < time_max:

    # time stepping
    dt = f.time_step(density, momentum, energy, ghost_nx, nx, dx, CFL, gamma)

    if dt < 1e-7:
        break
    time = time + dt
    time_step = time_step + 1

    # apply artificial viscosity
    density = f.artificial_viscosity(density, ghost_nx, nx, epsilon_artificial_viscosity)
    momentum = f.artificial_viscosity(momentum, ghost_nx, nx, epsilon_artificial_viscosity)
    energy = f.artificial_viscosity(energy, ghost_nx, nx, epsilon_artificial_viscosity)

    # calculate new quantities
    if solver == 2:
        [density_new, momentum_new, energy_new] = flux.LaxFriedrich(density, momentum, energy, ghost_nx, nx, gamma, dt, dx)
    elif solver == 3:
        [density_new, momentum_new, energy_new] = flux.MacCormack(density, momentum, energy, ghost_nx, nx, gamma, dt, dx)
    elif solver == 4:
        [density_new, momentum_new, energy_new] = flux.RichtMyer(density, momentum, energy, ghost_nx, nx, gamma, dt, dx)
    else:
        [density_new, momentum_new, energy_new] = flux.basic(density, momentum, energy, ghost_nx, nx, gamma, dt, dx)

    # apply boundary conditions
    density_new = f.boundary_conditions(density_new, ghost_nx, nx)
    momentum_new = f.boundary_conditions(momentum_new, ghost_nx, nx)
    energy_new = f.boundary_conditions(energy_new, ghost_nx, nx)

    # update quantities
    density = density_new
    momentum = momentum_new
    energy = energy_new

    print(time, dt)
    f.write(x, density, momentum, energy, time_step, solver, gamma, CFL, epsilon_artificial_viscosity)

print("final time step = " + str(time_step))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(x, density, 'k')
plt.figure()
plt.plot(x, momentum, 'b')
plt.figure()
plt.plot(x, energy, 'r')

plt.show()
