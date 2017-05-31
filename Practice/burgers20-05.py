import numpy as np
import math as m
import matplotlib.pyplot as plt
import copy
import csv

class Wave:

    def __init__(self, nt, nx, a):
        self.nt = nt
        self.nx = nx
        self.dx = 1.0 / self.nx
        self.dt = 0.8 * self.dx
        self.a = a
        self.C = self.a * self.dt / self.dx
        self.x = np.linspace(-1, 1, self.nx)
        self.u_0 = np.zeros(self.nx)
        self.u_all = np.zeros((self.nt, self.nx))
        self.u = np.zeros(self.nx)

        self.U = np.zeros((3, self.nx))
        self.F = np.zeros((3,self.nx))

    def profile(self, profile):
        if profile == "sine":
            self.u_0 = [-m.sin(m.pi*self.x[i]) for i in range(self.nx)]

        elif profile == "step":
            for i in range(self.nx):
                if i < self.nx*0.5:
                    self.u_0[i] = -1.0
                elif i > self.nx*0.5:
                    self.u_0[i] = 1.0
                else:
                    self.u_0[i] = 0.125

        elif profile == "top_hat":
            for i in range(self.nx):
                if self.nx / 5 <= i <= self.nx * 2 / 5:
                    self.u_0[i] = 1
                else:
                    self.u_0[i] = 0

        elif profile == "test3_tororossothebest":
            for i in range(self.nx):
                if i < self.nx / 3:
                    self.u_0[i] = -0.5
                elif self.nx / 3 <= i <= self.nx * 2 / 3:
                    self.u_0[i] = 1.0
                else:
                    self.u_0[i] = 0
        else:
            print ("profile not recognised")
            return

        self.u_all[0] = self.u_0
        self.u_all[:,0] = self.u_0[0]
        self.u_all[:,-1] = self.u_0[-1]

        return

    def burgers(self, scheme):
        self.u = copy.copy(self.u_0)

        if scheme == "LaxF":
            for n in range(1, self.nt):
                filename = '1Dsolve_'+str(n)+'.csv.'
                file = open(filename, "w")
                w = csv.writer(file, delimiter=',')
                w.writerow(('x_i','u_i'))
                for i in range(1, self.nx-1):
                    # explicit
                    u = self.u_all[n-1, i]
                    u_plus = self.u_all[n-1, i+1]
                    u_minus = self.u_all[n-1, i-1]
                    self.u_all[n, i] = 0.5*(u_plus + u_minus) - 0.25*self.C*(u_plus**2 - u_minus**2)

                    # implicit : 0.5*(u_i+1 - u_i-1) - 0.5*dt/dx*(F_i+1 - F_i-1)
                    self.u[i] = 0.5*(self.u[i+1] + self.u[i-1]) - 0.25*self.C*(self.u[i+1]**2 - self.u[i-1]**2)

                    w.writerow((self.dx*i, self.u[i]))
                file.close()

        if scheme == "LaxW":
            for n in range(1, self.nt):
                for i in range(1, self.nx-1):
                    # explicit
                    u = self.u_all[n-1, i]
                    u_plus = self.u_all[n-1, i+1]
                    u_minus = self.u_all[n-1, i-1]
                    self.u_all[n, i] = u - 0.25*self.C*(u_plus**2 - u_minus**2) + 0.125*self.C*self.C*(((u_plus + u)*(u_plus**2 - u**2))-((u + u_minus)*(u**2 - u_minus**2)))
                    # implicit : u_i - 0.5*dt/dx*(F_i+1 - F_i-1) + 0.5*dt/dx^2*0.5*((ui+ - ui)*(Fi+ - Fi) - (ui - ui-)*(Fi - Fi-))
                    self.u[i] = self.u[i] - 0.25*self.C*(self.u[i+1]**2 - self.u[i-1]**2) + 0.125*self.C*self.C*(((self.u[i+1] + self.u[i])*(self.u[i+1]**2 - self.u[i]**2)) - ((self.u[i]+self.u[i+1])*(self.u[i]**2 - self.u[i-1]**2)))

        if scheme == "RichtM":
            for n in range(1, self.nt):
                for i in range(1, self.nx-1):
                    # explicit
                    u = self.u_all[n-1, i]
                    u_plus = self.u_all[n-1, i+1]
                    u_minus = self.u_all[n-1, i-1]
                    uhalf_plus = 0.5*(u_plus + u) - 0.25*self.C*(u_plus**2 - u**2)
                    uhalf_minus = 0.5*(u + u_minus) - 0.25*self.C*(u**2 - u_minus**2)

                    self.u_all[n, i] = u - 0.25*self.C*(uhalf_plus**2 - uhalf_minus**2)

                    # implicit
                    u = copy.copy(self.u)
                    self.u[i] = u[i] - 0.25*self.C*((0.5*(u[i+1] + u[i]) - 0.25*self.C*(u[i+1]**2 - u[i]**2))**2 - (0.5*(u[i] + u[i-1]) - 0.25*self.C*(u[i]**2 - u[i-1]**2))**2)

        if scheme == "Godunov":
            for n in range(1, self.nt):
                for i in range(1, self.nx-1):
                    # explicit
                    u = self.u_all[n-1, i]
                    u_plus = self.u_all[n-1, i+1]
                    u_minus = self.u_all[n-1, i-1]
                    self.u_all[n, i] = u - self.C*(nf(u, u_plus) - nf(u_minus, u))

                self.u_all[n, 0] = self.u_all[n, 1]
                self.u_all[n, self.nx-1] = self.u_all[n, self.nx-2]

def nf(u,v):
    if u == v or (u**2 - v**2)/(u-v) >= 0:
        F = 0.5*u*u
    elif (u**2 - v**2)/(u-v) < 0:
        F = 0.5*v*v

    return F

def plot_all(x, u_all):
    plt.figure()
    for i in range(len(u_all)):
        plt.plot(x, u_all[i])
        plt.draw()

w2 = Wave(10, 30, 1.0)      # initialise wave parameters
w2.profile("top_hat")       # assign initial profile
w2.burgers("LaxF")          # solve using assigned scheme

plot_all(w2.x, w2.u_all)    # plot density, velocity, pressure
plt.show()
