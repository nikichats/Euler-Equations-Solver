import matplotlib.pyplot as plt
import math as m
import csv


def sound_speed(density, momentum, energy, gamma):
    pressure = (energy - 0.5 * (momentum**2) / density) * (gamma - 1)
    sound_speed = m.sqrt(abs(gamma * pressure / density))

    return sound_speed


def time_step(density, momentum, energy, ghost_nx, nx, dx, CFL, gamma):
    """ Calculates dt required to satisfy the CFL condition using the maximum wave speed in the cell reference frame"""
    velocity = momentum / density
    u_max = 1e-37
    for i in range(ghost_nx, nx-ghost_nx):
        u_new = abs(velocity[i]) + sound_speed(density[i], momentum[i], energy[i], gamma)
        if u_new > u_max:
            u_max = u_new
    dt = CFL*dx/u_max
    print("sound speed = " + str(u_max))
    return dt


def boundary_conditions(quantity, ghost_nx, nx):
    for i in range(0, ghost_nx):
        quantity[ghost_nx-1-i] = quantity[ghost_nx]
        quantity[nx - ghost_nx + i] = quantity[nx - ghost_nx - 1]

    return quantity


def artificial_viscosity(quantity, ghost_nx, nx, epsilon):
    for i in range(ghost_nx, nx - ghost_nx):
        quantity[i] = quantity[i] + epsilon * (quantity[i+1] - 2 * quantity[i] + quantity[i-1])

    return quantity


def write(x, density, momentum, energy, time_step, solver, gamma, CFL, epsilon_artificial_viscosity):
    if solver == 1:
        filename = 'Euler_1D_basic_' + str(time_step) + '.csv'
    elif solver == 2:
        filename = 'Euler_1D_LaxF_' + str(time_step) + '.csv'
    elif solver == 3:
        filename = 'Euler_1D_MacC_' + str(time_step) + '.csv'
    elif solver == 4:
        filename = 'Euler_1D_RichtM_' + str(time_step) + '.csv'
    else:
        filename = 'Euler_1D_' + str(time_step) + '.csv'
    file = open(filename, 'w', newline='')
    w = csv.writer(file, delimiter=',')
    w.writerow(('x', 'density', 'momentum', 'energy', 'pressure', 'sound speed'))
    for i in range(len(x)-1):
        pressure = (energy[i] - 0.5 * momentum[i]**2 / density[i]) * (gamma - 1)
        sound_speed = m.sqrt(abs(gamma * pressure / density[i]))
        w.writerow((x[i], density[i], momentum[i], energy[i], pressure, sound_speed))
    file.close()
    return


def plot_all(x, density, momentum, energy):
    plt.figure()
    plt.plot(x, density, 'k', label='Density')
    plt.plot(x, momentum, 'r', label='Momentum')
    plt.plot(x, energy, 'b', label='energy')
    plt.legend()
    return
