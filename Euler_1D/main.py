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

x = np.linspace(x_min - ghost_nx * dx, x_max + ghost_nx * dx, nx + 1)
time_max = 0.01
CFL = 0.3
gamma = 1.4
eps_AV = 0.0

solver = 4

### Initial Conditions
rho_L = 1.
rho_R = 0.125
u_L = 0.0
u_R = 0.0
p_L = 1.e5
p_R = 1.e4
shock_position = 0.5

[density, momentum, energy, velocity, pressure] = ic.initial_cond(x, rho_L, rho_R, u_L, u_R, p_L, p_R, gamma, nx, dx, shock_position)

### Solve
time = 0.0
time_step = 0
f.write(x, density, momentum, energy, time_step, solver, gamma)

while time < time_max:
    # time stepping
    dt = f.time_step(density, velocity, pressure, ghost_nx, nx, dx, CFL, gamma)
    dt = 4.276e-4
    if dt < 1e-7:
        break
    time = time + dt
    time_step = time_step + 1

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
    velocity = momentum / density
    pressure = (gamma - 1) * (energy - 0.5 * (momentum**2) / density)

    f.write(x, density, momentum, energy, time_step, solver, gamma)

print("final time step = " + str(time_step))

