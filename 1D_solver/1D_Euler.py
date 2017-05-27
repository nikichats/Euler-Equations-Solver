import numpy as np
import math as m
import matplotlib.pyplot as plt
import copy
import flux

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
        """cases represent different jumps in primitive variables"""
        if case == 1:
            x_jump = 0.5
            rho_L = 1.0
            rho_R = 0.125
            u_L = 0.75
            u_R = 0.0
            p_L = 1.0
            p_R = 0.1

        if case == 2:
            x_jump = 0.5
            rho_L = 1.0
            rho_R = 1.0
            u_L = -2.0
            u_R = 2.0
            p_L = 0.4
            p_R = 0.4

        for i in range(self.nx):
            if i*self.dx < x_jump:
                self.U_prim[0, i] = rho_L
                self.U_prim[1, i] = u_L
                self.U_prim[2, i] = p_L
            else:
                self.U_prim[0, i] = rho_R
                self.U_prim[1, i] = u_R
                self.U_prim[2, i] = p_R

        self.U = variables(self.gamma, self.U_prim)

    # def solve(self, solver):
    #     if solver == 1:
    #         for n in range(1, self.nt):
    #             U = copy.copy(self.U)
    #             F_plus = flux.LF(self.gamma, U[:, 2:-1])
    #             F_minus = flux.LF(self.gamma, U[:, 0:-3])
    #             self.U[:, 1:-2] = 0.5*(self.U[:, 0:-3] + self.U[:, 2:-1]) + 0.5*(F_plus - F_minus)
    #             self.U[:, 0] = self.U[:, 1]
    #             self.U[:, self.nx-1] = self.U[:, self.nx-2]
    #     self.U_prim = primitive(self.gamma, self.U)

    def solve(self, solver):
        if solver == 1:
            plt.figure()
            plt.plot(self.x, self.U_prim[1])
            plt.draw()
            for n in range(1, self.nt):
                self.dt = time_step(self.U_prim, self.nx, self.dx, self.CFL, self.gamma)
                U_next = flux.LaxF(self.U, self.U_prim, self.nx, self.gamma, self.dt, self.dx)
                U_next = bc(U_next, self.nx, 1)
                self.U = copy.copy(U_next)
                self.U_prim = primitive(self.gamma, U_next)
                plt.plot(self.x, self.U_prim[1])
                plt.draw()
            plt.show()
        return self.U

def time_step(U_prim, nx, dx, CFL, gamma):
    u_max = 0.0
    for i in range(nx):
        u_loc = abs(U_prim[1, i]) + m.sqrt(abs(gamma*U_prim[2, i]/U_prim[0, i]))
        if u_loc > u_max:
            u_max = u_loc

    dt = dx*CFL/u_max
    return dt

def bc(U, nx, type):
    """ Type 1 : transmission boundary conditions, Type 2 : periodic"""
    if type == 2:
        U[:, 0] = U[:, nx-2]
        U[:, nx-1] = U[:, 2]
    else:
        U[:, 0] = U[:, 1]
        U[:, nx-1] = U[:, nx-2]
    return U

def variables(gamma, U_prim):
    """ Integration variables [ rho, rho*u, 0.5*rho*u^2 + p/(gamma-1)]"""
    U = np.zeros(U_prim.shape)
    U[0, :] = U_prim[0, :]
    U[1, :] = U_prim[0, :]*U_prim[1, :]
    U[2, :] = 0.5*U_prim[0, :]*(U_prim[1, :]**2) + U_prim[2, :]/(gamma - 1)
    return U

def primitive(gamma, U):
    """ Primitive variables [ rho, u, p] """
    U_prim = np.zeros(U.shape)
    U_prim[0, :] = U[0, :]
    U_prim[1, :] = U[1, :]/U[0, :]
    U_prim[2, :] = (gamma-1)*(U[2, :] - 0.5*(U[1, :]**2)/U[0, :])
    return U_prim

def plot_all(x, u_all):
    plt.figure()
    for i in range(len(u_all)):
        plt.plot(x, u_all[i])
        plt.draw()

w = Wave(5, 50, -1.0)
w.init_cond(1)
w.solve(1)
