import matplotlib.pyplot as plt
import math as m
import csv


def time_step(density, velocity, pressure, ghost_nx, nx, dx, CFL, gamma):
    """ Calculates dt required to satisfy the CFL condition using the maximum wave speed in the cell reference frame"""
    u_max = 1e-37
    for i in range(ghost_nx, nx-ghost_nx):
        u_new = abs(velocity[i]) + m.sqrt(abs(gamma*pressure[i]/density[i]))
        if u_new > u_max:
            u_max = u_new
    dt = CFL*dx/u_max
    print ( "sound speed = " + str(u_max))
    return dt


def boundary_conditions(quantity, ghost_nx, nx):
    for i in range(0, ghost_nx):
        quantity[ghost_nx-1-i] = quantity[ghost_nx]
        quantity[nx - ghost_nx + i] = quantity[nx - ghost_nx -1]

    return quantity


def write(x, density, momentum, energy, time_step, solver):
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
    file = open(filename, 'w')
    w = csv.writer(file, delimiter=',')
    w.writerow(('x', 'density', 'momentum', 'energy'))
    for i in range(len(x)-1):
        w.writerow((x[i], density[i], momentum[i], energy[i]))
    file.close()
    return


def plot_all(x, density, momentum, energy):
    plt.figure()
    plt.plot(x, density, 'k', label='Density')
    plt.plot(x, momentum, 'r', label='Momentum')
    plt.plot(x, energy, 'b', label='energy')
    plt.legend()
    return
