import matplotlib.pyplot as plt
import numpy as np
import math as m
import csv


def sound_speed(density, momentum, energy, gamma):
    pressure = (energy - 0.5 * (momentum**2) / density) * (gamma - 1)
    sound_speed = np.sqrt(abs(gamma * pressure / density))

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
    # print("sound speed = " + str(u_max))
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


def return_primitive(density, momentum, energy, gamma):
    velocity = momentum / density
    pressure = (gamma - 1) * (energy - 0.5 * (momentum**2) / density)

    return density, velocity, pressure


def return_conserved(density, velocity, pressure, gamma):
    momentum = density * velocity
    energy = (pressure / (gamma - 1)) + 0.5 * density * velocity**2

    return density, momentum, energy


def write(x, density, momentum, energy, time_step, solver, gamma, CFL, epsilon_artificial_viscosity):
    if solver == 1:
        filename = 'Euler_1D_basic_' + str(time_step) + '.csv'
    elif solver == 2:
        filename = 'Euler_1D_LaxF_' + str(time_step) + '.csv'
    elif solver == 3:
        filename = 'Euler_1D_MacC_' + str(time_step) + '.csv'
    elif solver == 4:
        filename = 'Euler_1D_RichtM_' + str(time_step) + '.csv'
    elif solver == 5:
        filename = 'Euler_1D_Godunov_' + str(time_step) + '.csv'
    else:
        filename = 'Euler_1D_' + str(time_step) + '.csv'
    file = open(filename, 'w', newline='')
    w = csv.writer(file, delimiter=',')
    w.writerow(('x', 'density', 'momentum', 'energy', 'velocity', 'pressure', 'sound speed'))
    for i in range(len(x)-1):
        velocity = momentum[i] / density[i]
        pressure = (energy[i] - 0.5 * momentum[i]**2 / density[i]) * (gamma - 1)
        sound_speed = m.sqrt(abs(gamma * pressure / density[i]))
        w.writerow((x[i], density[i], momentum[i], energy[i], velocity, pressure, sound_speed))
    file.close()
    return


def plot_all(x, density, velocity, pressure, momentum, energy, x_analytic, density_analytic, velocity_analytic,
             pressure_analytic, momentum_analytic, energy_analytic, gamma):

    mach_number = velocity / sound_speed(density, momentum, energy, gamma)
    mach_number_analytic = velocity_analytic / sound_speed(density_analytic, momentum_analytic, energy_analytic, gamma)

    fr, axarr = plt.subplots(2, 2)
    # fr, axarr = plt.subplots(3, 2)

    axarr[0, 0].plot(x_analytic, density_analytic, 'k', x, density, 'ko', markersize=2)
    axarr[0, 0].set_title('Density')

    axarr[0, 1].plot(x_analytic, pressure_analytic, 'b', x, pressure, 'ko', markersize=2)
    axarr[0, 1].set_title('Pressure')

    axarr[1, 0].plot(x_analytic, momentum_analytic, 'g', x, momentum, 'ko', markersize=2)
    axarr[1, 0].set_title('Momentum')

    axarr[1, 1].plot(x_analytic, energy_analytic, 'c', x, energy, 'ko', markersize=2)
    axarr[1, 1].set_title('Energy')

    # axarr[2, 0].plot(x_analytic, velocity_analytic, 'm', x, velocity, 'ko', markersize=2)
    # axarr[2, 0].set_title('Velocity')
    #
    # axarr[2, 1].plot(x_analytic, mach_number_analytic, 'r', x, mach_number, 'ko', markersize=2)
    # axarr[2, 1].set_title('Mach Number')

    fr.subplots_adjust(hspace=0.5)
    plt.show()
