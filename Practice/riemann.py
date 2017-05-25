###################
##  (1) using q_i^n, find q_L,i-1/2 and q_R,i-1/2 using volume averages
##  (2) calculate flux using Riemann solvers
##  (3) update q_i^n+1
##  (4) update time step, calculate new CFL condition based on wave speed
##  (5) repeat (1)-(4) until time criteria is met
###################

import numpy as np
import math as m
import matplotlib.pyplot as plt
import copy

class Euler:
    def __init__(self, xmax, xmin, nx, nt):
        self.dx = (xmax-xmin)/nx
        self.dt = 0.8*self.dx
        self.CFL = self.dt/self.dx
        self.nx = nx
        self.nt = nt
        self.gamma = 1.4

        self.x = np.linspace(xmin, xmax, nx)
        self.rho = np.zeros(nx)
        self.u = np.zeros(nx)
        self.p = np.zeros(nx)
        self.U = np.zeros((3,nx))
        self.F = np.zeros((3,nx))

    def initialise(self, rho_L, rho_R, u_L, u_R, p_L, p_R):
        for i in range(self.nx):
            if i < self.nx*0.5:
                self.rho[i] = rho_L
                self.u[i] = u_L
                self.p[i] = p_L
            else:
                self.rho[i] = rho_R
                self.u[i] = u_R
                self.p[i] = p_R

        for i in range(self.nx):
            U = u_vector(self.gamma, self.rho[i], self.u[i], self.p[i])
            self.U[0, i] = U[0]
            self.U[1, i] = U[1]
            self.U[2, i] = U[2]

    def burgers(self, scheme):
        if scheme == "LaxF":
            plt.figure()
            for n in range(self.nt):
                U = copy.copy(self.U)
                for i in range(1, self.nx-2):
                    F_minus = flux(self.gamma, U[0, i-1], U[1, i-1], U[2, i-1])
                    F_plus = flux(self.gamma, U[0, i+1], U[1, i+1], U[2, i+1])
                    self.U[:, i] = 0.5*(self.U[:, i+1] + self.U[:, i-1]) - 0.5*self.CFL*(F_plus[:] - F_minus[:])

                self.U[:, 0] = self.U[:, 1]
                self.U[:, self.nx-1] = self.U[:, self.nx-2]

                plt.plot(self.x, self.U[0])
                plt.draw()

            plt.show()

def u_vector(gamma, rho, u, p):
    U = np.array([rho, rho*u, p/(gamma-1)])
    return U

def flux1(gamma, rho, rhou, rhoE):
    p = (gamma-1)*(rhoE - 0.5*rho*rhou)
    F0 = rhou
    F1 = rho*rhou + p
    F2 = (rhou/rho)*(rhoE + p)
    F = np.array([F0, F1, F2])
    return F

def plot_all(x, u_all):
    plt.figure()
    for i in range(len(u_all)):
        plt.plot(x, u_all[i])
        plt.draw()

e = Euler(1., -1., 20, 5)
e.initialise(1.0, 0.125, 0.1, 0.1, 1.0, 0.1)
e.burgers("LaxF")
# plot_all(e.x, e.U)
# plt.show()
# adaptive scheme using flux limiters F = F1 + phi*F2, F1 ~ first order upwind, F2 - higher order
