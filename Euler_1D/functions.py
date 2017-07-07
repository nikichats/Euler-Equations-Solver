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


def convert_to_base(number, base):
    number_base_rep = []
    decimal_base_rep = []
    integer = int(number)
    decimal = abs(number - integer)

    if decimal == 0:
        decimal_base_rep.append(0)
    else:
        # while decimal != 0 or integer != 0:
        for i in range(40):
            decimal = base * decimal
            integer = int(decimal)
            decimal = abs(decimal - integer)
            decimal_base_rep.append(integer)
        decimal_base_rep = (decimal_base_rep[:-1])

    integer = int(number)
    if integer == 0:
        number_base_rep.insert(0, 0)
    else:
        while integer != 0:
            integer, remainder = divmod(integer, base)
            number_base_rep.append(remainder)
        number_base_rep = number_base_rep[:-1]

    return(number_base_rep, decimal_base_rep)


def convert_to_full(int_seq, dec_seq, base):
    a = 0.0
    l = len(int_seq)
    for i in range(l):
        j = l - 1 - i
        a += int_seq[j] * (base**i)

    l = len(dec_seq)
    for i in range(1,l+1):
        a += (dec_seq[i-1] * (base**(-i)))

    return(a)


def theta_vdc(k1, k2, n):
    for i in range(1, n+1):
        [int_seq, dec_seq] = convert_to_base(float(i/n), k1)
        a = convert_to_full(int_seq, dec_seq, k1)
        print(a)


def theta_numerica(k1, k2, n):
    theta = 0.0
    nn = 10.0
    i = 0
    while nn >= 1:
        l = nn%k1
        j = (k2*l)%k1
        theta += j/(k1**(i+1))
        i += 1

    print(theta)


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


def min_mod2(x, y):
    def MM(a, b):
        if a*b  < 0 :
            return 0
        elif abs(a) < abs(b):
            return a
        else:
            return b

    return np.array([MM(x[j], y[j]) for j in range(len(x))])


def min_mod3(x, y, z):

    def MM(a, b, c):
        mm = float((1./3.) * (np.sign(a) + np.sign(b) + np.sign(c)) * min(abs(a), min(abs(b), abs(c))))
        return mm

    if np.size(x) == 1 and np.size(y) == 1:
        return MM(x, y, z)
    else:
        assert(len(x) == len(y))
        return np.array([MM(x[j], y[j], z[j]) for j in range(len(x))])


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

    # mach_number = velocity / sound_speed(density, momentum, energy, gamma)
    # mach_number_analytic = velocity_analytic / sound_speed(density_analytic, momentum_analytic, energy_analytic, gamma)

    fr, axarr = plt.subplots(2, 2)

    axarr[0, 0].plot(x_analytic, density_analytic, 'k', x, density, 'ko-', markersize=2)
    axarr[0, 0].set_title('Density')

    axarr[0, 1].plot(x_analytic, pressure_analytic, 'b', x, pressure, 'ko-', markersize=2)
    axarr[0, 1].set_title('Pressure')

    axarr[1, 0].plot(x_analytic, velocity_analytic, 'g', x, velocity, 'ko-', markersize=2)
    axarr[1, 0].set_title('Velocity')

    # axarr[1, 0].plot(x_analytic, mach_number_analytic, 'r', x, mach_number, 'ko', markersize=2)
    # axarr[1, 0].set_title('Mach Number')

    axarr[1, 1].plot(x_analytic, energy_analytic, 'c', x, energy, 'ko-', markersize=2)
    axarr[1, 1].set_title('Energy')

    fr.subplots_adjust(hspace=0.5)
    plt.show()
