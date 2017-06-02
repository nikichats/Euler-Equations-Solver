import numpy as np
import functions as f

def flux_cc(density, momentum, energy, gamma):
    flux_density = momentum
    flux_momentum = (gamma-1)*energy + 0.5*(3-gamma)*(momentum**2)/density
    flux_energy = (gamma*energy - (gamma-1)*0.5*(momentum**2)/density)*(momentum/density)

    return flux_density, flux_momentum, flux_energy


def flux_fc(density, momentum, energy, gamma):
    flux_density_L = momentum[:-1]
    flux_density_R = momentum[1:]
    flux_density = 0.5 * (flux_density_L + flux_density_R)

    flux_momentum_L = (gamma - 1) * energy[:-1] + 0.5 * (3 - gamma) * (momentum[:-1]**2) / density[:-1]
    flux_momentum_R = (gamma - 1) * energy[1:] + 0.5 * (3 - gamma) * (momentum[1:]**2) / density[1:]
    flux_momentum = 0.5 * (flux_momentum_L + flux_momentum_R)

    flux_energy_L = (gamma*energy[:-1] - (gamma-1)*0.5*(momentum[:-1]**2)/density[:-1])*(momentum[:-1]/density[:-1])
    flux_energy_R = (gamma*energy[1:] - (gamma-1)*0.5*(momentum[1:]**2)/density[1:])*(momentum[1:]/density[1:])
    flux_energy = 0.5 * (flux_energy_L + flux_energy_R)

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

    # (1) prediction step - LaxFriedrich
    [density_predictor, momentum_predictor, energy_predictor] = LaxFriedrich(density, momentum, energy, ghost_nx, nx, gamma, dt, dx)

    density_predictor = f.boundary_conditions(density_predictor, ghost_nx, nx)
    momentum_predictor = f.boundary_conditions(momentum_predictor, ghost_nx, nx)
    energy_predictor = f.boundary_conditions(energy_predictor, ghost_nx, nx)

    # (2) corrector step
    [flux_density, flux_momentum, flux_energy] = flux_cc(density_predictor, momentum_predictor, energy_predictor, gamma)
    for i in range(ghost_nx, nx - ghost_nx):
        density_new[i] = 0.5 * ( density_predictor[i] + density[i] - (dt/dx) * (flux_density[i] - flux_density[i-1]) )
        momentum_new[i] = 0.5 * (momentum[i] + momentum_predictor[i] - (dt/dx) * (flux_momentum[i] - flux_momentum[i-1]))
        energy_new[i] = 0.5 * (energy[i] + energy_predictor[i] - (dt/dx) * (flux_energy[i] - flux_energy[i-1]))

    return density_new, momentum_new, energy_new


def RichtMyer(density, momentum, energy, ghost_nx, nx, gamma, dt, dx):
    density_new = np.zeros(nx)
    momentum_new = np.zeros(nx)
    energy_new = np.zeros(nx)

    # (1) prediction step - LaxFriedrich
    [density_predictor, momentum_predictor, energy_predictor] = LaxFriedrich(density, momentum, energy, ghost_nx, nx, gamma, dt, dx)

    density_predictor = f.boundary_conditions(density_predictor, ghost_nx, nx)
    momentum_predictor = f.boundary_conditions(momentum_predictor, ghost_nx, nx)
    energy_predictor = f.boundary_conditions(energy_predictor, ghost_nx, nx)

    # (2) corrector step - Leapfrog scheme
    [flux_density, flux_momentum, flux_energy] = flux_cc(density_predictor, momentum_predictor, energy_predictor, gamma)

    for i in range(ghost_nx, nx - ghost_nx):
        density_new[i] = density[i] - (0.5 * dt / dx) * (flux_density[i+1] - flux_density[i - 1])
        momentum_new[i] = momentum[i] - (0.5 *dt / dx) * (flux_momentum[i+1] - flux_momentum[i - 1])
        energy_new[i] = energy[i] - (0.5* dt / dx) * (flux_energy[i+1] - flux_energy[i - 1])

    return density_new, momentum_new, energy_new


def RichtMyer2(density, momentum, energy, ghost_nx, nx, gamma, dt, dx):
    density_new = np.zeros(nx)
    momentum_new = np.zeros(nx)
    energy_new = np.zeros(nx)

    # (1) prediction step - LaxFriedrich
    [density_predictor, momentum_predictor, energy_predictor] = LaxFriedrich(density, momentum, energy, ghost_nx, nx, gamma, dt, dx)

    density_predictor = f.boundary_conditions(density_predictor, ghost_nx, nx)
    momentum_predictor = f.boundary_conditions(momentum_predictor, ghost_nx, nx)
    energy_predictor = f.boundary_conditions(energy_predictor, ghost_nx, nx)

    # (2) corrector step - Leapfrog scheme
    [flux_density, flux_momentum, flux_energy] = flux_fc(density_predictor, momentum_predictor, energy_predictor, gamma)

    for i in range(ghost_nx, nx - ghost_nx):
        density_new[i] = density[i] - (0.25 * dt / dx) * (flux_density[i] - flux_density[i - 1])
        momentum_new[i] = momentum[i] - (0.25 *dt / dx) * (flux_momentum[i] - flux_momentum[i - 1])
        energy_new[i] = energy[i] - (0.25* dt / dx) * (flux_energy[i] - flux_energy[i - 1])

    return density_new, momentum_new, energy_new