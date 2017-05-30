import numpy as np
import matplotlib.pyplot as plt
import copy

import flux
import initcond

class Wave:

    def __init__(self, nt, nx, a):
        self.nt = nt
        self.nx = nx
        self.dx = 2.0 / self.nx
        self.dt = 0.8 * self.dx
        self.CFL = 0.5
        self.gamma = 1.4
        self.x = np.linspace(-1, 1, self.nx)
        self.U_prim = np.zeros((3, self.nx))
        self.U = np.zeros((3, self.nx))
        self.U_next = np.zeros((3, self.nx))
        self.F = np.zeros((3, self.nx))

    def init_cond(self, case):
        """Case 0: Sod problem / Case 2"""
        x_jump = 0.5
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
            if i*self.dx < x_jump:
                self.U_prim[0, i] = U_0[0]
                self.U_prim[1, i] = U_0[2]
                self.U_prim[2, i] = U_0[4]
            else:
                self.U_prim[0, i] = U_0[1]
                self.U_prim[1, i] = U_0[3]
                self.U_prim[2, i] = U_0[5]

        self.U = flux.variables(self.gamma, self.U_prim)

    def solve(self, solver):
        plt.figure()
        plt.plot(self.x, self.U_prim[1])
        plt.draw()

        for n in range(1, self.nt):
            self.dt = flux.time_step(self.U_prim, self.nx, self.dx, self.CFL, self.gamma)
            if solver == 1:
                U_next = flux.LaxF(self.U, self.U_prim, self.nx, self.gamma, self.dt, self.dx)
            elif solver == 2:
                U_next = flux.LaxW(self.U, self.U_prim, self.nx, self.gamma, self.dt, self.dx)
            elif solver == 3:
                U_next = flux.Godunov(self.U, self.U_prim, self.nx, self.gamma, self.dt, self.dx)

            U_next = flux.bc(U_next, self.nx, 1)
            self.U = copy.copy(U_next)
            self.U_prim = flux.primitive(self.gamma, U_next)
            plt.plot(self.x, self.U_prim[1])
            plt.draw()

        plt.show()

        return self.U

def plot_all(x, u_all):
    plt.figure()
    for i in range(len(u_all)):
        plt.plot(x, u_all[i])
        plt.draw()

w = Wave(20, 50, 1.0)
w.init_cond(1)              # Toro test cases
w.solve(3)                  # 1: LaxF, 2: LaxW, 3: Godunov
plot_all(w.x, w.U)
plt.show()