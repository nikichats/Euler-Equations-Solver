import numpy as np
import matplotlib.pyplot as plt
import copy
import csv

import flux
import initcond

class Wave:

    def __init__(self, time_max, x_max, nx):
        self.nx = nx
        self.dx = x_max / self.nx
        self.dt = 0.0
        self.time_max = time_max
        self.CFL = 1.0
        self.gamma = 1.4
        self.x = np.linspace(0, 1, self.nx+1)   # face centered x coordinates
        self.xc = self.x[:-1] + 0.5*self.dx     # cell centered x coordinates
        self.U_prim = np.zeros((3, self.nx))    # cell centered
        self.U = np.zeros((3, self.nx))         # cell centered
        self.U_next = np.zeros((3, self.nx))    # cell centered

    def init_cond(self, case):
        """Case 0: Sod problem / Case 2"""
        x_jump = 0.3
        if case == 1:
            U_0 = initcond.sod()
        elif case == 2:
            U_0 = initcond.toro_2()
        elif case == 3:
            U_0 = initcond.toro_3()
        elif case == 4:
            U_0 = initcond.toro_4()
        elif case == 5:
            U_0 = initcond.toro_5()

        for i in range(self.nx):
            if self.xc[i] < x_jump:
                self.U_prim[0, i] = U_0[0]
                self.U_prim[1, i] = U_0[2]
                self.U_prim[2, i] = U_0[4]
            else:
                self.U_prim[0, i] = U_0[1]
                self.U_prim[1, i] = U_0[3]
                self.U_prim[2, i] = U_0[5]

        self.U = flux.variables(self.U_prim, self.gamma)

    def solve(self, solver):
        time = 0.0
        time_step = 0
        write(self.xc, self.U_prim, time_step, time)

        while time < self.time_max:
            ### calculate time step
            self.dt = flux.time_step(self.U_prim, self.nx, self.dx, self.CFL, self.gamma)
            if self.dt < 10**(-7):
                break
            time = time + self.dt
            time_step = time_step + 1

            ### advance solution
            U_next = None
            if solver == 1:
                U_next = flux.basic(self.U, self.U_prim, self.nx, self.gamma, self.dt, self.dx, self.CFL)
            if solver == 2:
                U_next = flux.LaxFriedrich(self.U, self.U_prim, self.nx, self.gamma, self.dt, self.dx, self.CFL)
            if solver == 3:
                U_next = flux.MacCormack(self.U, self.U_prim, self.nx, self.gamma, self.dt, self.dx, self.CFL)

            ### apply boundary conditions
            U_next = flux.bc(U_next, self.nx, 1)

            ### update variables
            self.U = copy.copy(U_next)
            self.U_prim = flux.primitive( U_next, self.gamma)

            ### write solution to file
            write(self.xc, self.U_prim, time_step, time)

        # plt.show()
        print ("time step = " + str(time_step))
        return self.U

def write(x, U, time_step, time):
        filename = '1D_Euler_' + str(time_step) + '.csv'
        file = open(filename, 'w', newline='')
        w = csv.writer(file, delimiter=',')
        w.writerow(('x', 'rho', 'u', 'p'))
        for i in range(len(x)):
            w.writerow((x[i], U[0, i], U[1, i], U[2, i]))
        file.close()
        return

def plot_all(x, u_all):
    plt.figure()
    for i in range(len(u_all)):
        plt.plot(x, u_all[i], label='variable'+str(i))
        plt.draw()
    plt.legend()

w = Wave(0.2, 1.0, 200)     # max time, max length, number of spatial steps (nx)
w.init_cond(1)             # Toro test cases
w.solve(3)                 # 1: basic, 2: Lax Friedrichs, 3: example_code
plot_all(w.xc, w.U)
plt.show()
