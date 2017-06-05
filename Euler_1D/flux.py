import numpy as np
import functions as f


def flux_cc(density, momentum, energy, gamma):
    velocity = momentum / density
    kinetic_energy = 0.5 * momentum**2 / density
    internal_energy = energy - kinetic_energy
    pressure = (gamma-1)*internal_energy

    flux_density = density * velocity
    flux_momentum =  pressure + momentum * velocity
    flux_energy = (energy + pressure) * velocity

    return flux_density, flux_momentum, flux_energy


def flux_fc(density, momentum, energy, gamma):
    density_L = density[:-1]
    density_R = density[1:]

    momentum_L = momentum[:-1]
    momentum_R = momentum[1:]

    energy_L = energy[:-1]
    energy_R = energy[1:]

    velocity_L = momentum_L / density_L
    velocity_R = momentum_R / density_R

    kinetic_energy_L = 0.5 * momentum_L**2 / density_L
    kinetic_energy_R = 0.5 * momentum_R**2 / density_R

    internal_energy_L = energy_L - kinetic_energy_L
    internal_energy_R = energy_R - kinetic_energy_R

    pressure_L = (gamma - 1) * internal_energy_L
    pressure_R = (gamma - 1) * internal_energy_R

    flux_density_L = density_L * velocity_L
    flux_density_R = density_R * velocity_R
    flux_density = 0.5 * (flux_density_L + flux_density_R)

    flux_momentum_L = pressure_L + momentum_L * velocity_L
    flux_momentum_R = pressure_R + momentum_R * velocity_R
    flux_momentum = 0.5 * (flux_momentum_L + flux_momentum_R)

    flux_energy_L = (energy_L + pressure_L) * velocity_L
    flux_energy_R = (energy_R + pressure_R) * velocity_R
    flux_energy = 0.5 * (flux_energy_L + flux_energy_R)

    return flux_density, flux_momentum, flux_energy


# def flux_fc(density, momentum, energy, gamma):
#     flux_density_L = momentum[:-1]
#     flux_density_R = momentum[1:]
#     flux_density = 0.5 * (flux_density_L + flux_density_R)
#
#     flux_momentum_L = (gamma - 1) * energy[:-1] + 0.5 * (3 - gamma) * (momentum[:-1]**2) / density[:-1]
#     flux_momentum_R = (gamma - 1) * energy[1:] + 0.5 * (3 - gamma) * (momentum[1:]**2) / density[1:]
#     flux_momentum = 0.5 * (flux_momentum_L + flux_momentum_R)
#
#     flux_energy_L = (gamma*energy[:-1] - (gamma-1)*0.5*(momentum[:-1]**2)/density[:-1])*(momentum[:-1]/density[:-1])
#     flux_energy_R = (gamma*energy[1:] - (gamma-1)*0.5*(momentum[1:]**2)/density[1:])*(momentum[1:]/density[1:])
#     flux_energy = 0.5 * (flux_energy_L + flux_energy_R)
#
#     return flux_density, flux_momentum, flux_energy


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
    density_new = np.zeros(nx)
    momentum_new = np.zeros(nx)
    energy_new = np.zeros(nx)
    [flux_density, flux_momentum, flux_energy] = flux_fc(density, momentum, energy, gamma)

    for i in range(ghost_nx, nx - ghost_nx):
        density_new[i] = 0.5 * (density[i + 1] + density[i - 1]) - (dt/dx) * (flux_density[i] - flux_density[i-1])
        momentum_new[i] = 0.5 * (momentum[i + 1] + momentum[i - 1]) - (dt/dx) * (flux_momentum[i] - flux_momentum[i-1])
        energy_new[i] = 0.5 * (energy[i + 1] + energy[i - 1]) - (dt/dx) * (flux_energy[i] - flux_energy[i-1])

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
        density_predictor[i] = density[i]  - (dt/dx) * (flux1_density[i+1] - flux1_density[i])
        momentum_predictor[i] = momentum[i] - (dt/dx) * (flux1_momentum[i+1] - flux1_momentum[i])
        energy_predictor[i] = energy[i] - (dt/dx) * (flux1_energy[i+1] - flux1_energy[i])

    density_predictor = f.boundary_conditions(density_predictor, ghost_nx, nx)
    momentum_predictor = f.boundary_conditions(momentum_predictor, ghost_nx, nx)
    energy_predictor = f.boundary_conditions(energy_predictor, ghost_nx, nx)

    # (2) corrector step
    [flux2_density, flux2_momentum, flux2_energy] = flux_cc(density_predictor, momentum_predictor, energy_predictor, gamma)
    for i in range(ghost_nx, nx - ghost_nx):
        density_new[i] = 0.5 * ( density[i] + density_predictor[i] - (dt/dx) * (flux2_density[i] - flux2_density[i-1]) )
        momentum_new[i] = 0.5 * (momentum[i] + momentum_predictor[i] - (dt/dx) * (flux2_momentum[i] - flux2_momentum[i-1]))
        energy_new[i] = 0.5 * (energy[i] + energy_predictor[i] - (dt/dx) * (flux2_energy[i] - flux2_energy[i-1]))

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
        density_predictor[i] = 0.5 * (density[i + 1] + density[i - 1]) - 0.5 * (0.5 * dt/dx) * (flux1_density[i+1] - flux1_density[i-1])
        momentum_predictor[i] = 0.5 * (momentum[i + 1] + momentum[i - 1]) - 0.5 * (0.5 * dt/dx) * (flux1_momentum[i+1] - flux1_momentum[i-1])
        energy_predictor[i] = 0.5 * (energy[i + 1] + energy[i - 1]) - 0.5 * (0.5 * dt/dx) * (flux1_energy[i+1] - flux1_energy[i-1])

    density_predictor = f.boundary_conditions(density_predictor, ghost_nx, nx)
    momentum_predictor = f.boundary_conditions(momentum_predictor, ghost_nx, nx)
    energy_predictor = f.boundary_conditions(energy_predictor, ghost_nx, nx)

    # (2) corrector step

    [flux2_density, flux2_momentum, flux2_energy] = flux_cc(density_predictor, momentum_predictor, energy_predictor, gamma)
    for i in range(ghost_nx, nx - ghost_nx):
        density_new[i] = density[i] - 0.5 * (dt/dx) * (flux2_density[i+1] - flux2_density[i-1])
        momentum_new[i] = momentum[i] - 0.5 * (dt/dx) * (flux2_momentum[i+1] - flux2_momentum[i-1])
        energy_new[i] = energy[i] - 0.5 * (dt/dx) * (flux2_energy[i+1] - flux2_energy[i-1])

    # [flux2_density, flux2_momentum, flux2_energy] = flux_fc(density_predictor, momentum_predictor, energy_predictor, gamma)
    # for i in range(ghost_nx, nx - ghost_nx):
    #     density_new[i] = density[i] - 0.5 * (dt/dx) * (flux2_density[i] - flux2_density[i-1])
    #     momentum_new[i] = momentum[i] - 0.5 * (dt/dx) * (flux2_momentum[i] - flux2_momentum[i-1])
    #     energy_new[i] = energy[i] - 0.5 * (dt/dx) * (flux2_energy[i] - flux2_energy[i-1])

    return density_new, momentum_new, energy_new
