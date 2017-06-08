import numpy as np
import functions as f


def flux_cc(density, momentum, energy, gamma):
    velocity = momentum / density
    kinetic_energy = 0.5 * momentum**2 / density
    internal_energy = energy - kinetic_energy
    pressure = (gamma-1)*internal_energy

    flux_density = density * velocity
    flux_momentum = pressure + momentum * velocity
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
