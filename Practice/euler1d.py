import numpy as np
import math as m
import matplotlib.pyplot as plt
import copy

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
        self.U_0 = np.zeros((3,self.nx))

    def profile(self, profile):
        if profile == "sine":
            self.u_0 = [-m.sin(m.pi*self.x[i]) for i in range(self.nx)]
        elif profile == "tophat":
            for i in range(self.nx):
                if self.nx / 3 <= i <= self.nx * 2 / 3:
                    self.u_0[i] = 1
                else:
                    self.u_0[i] = -1
        elif profile == "test3_toro":
            for i in range(self.nx):
                if i < self.nx/3:
                    self.u_0[i] = -0.5
                elif self.nx/3 <= i <= self.nx*2/3:
                    self.u_0[i] = 1.0
                else:
                    self.u_0[i] = 0
        else:
            print "profile not recognised"
            return

        self.u_all[0] = self.u_0
        return

    def linadv(self, scheme):
        for i in range(1,self.nt):
            u = self.u_all[i-1]
            u_plus = np.roll(self.u_all[i-1], 1)
            u_minus = np.roll(self.u_all[i-1], -1)

            if scheme == "LaxF":
                f_plus = self.a*u_plus
                f_minus = self.a*u_minus
                self.u_all[i] = 0.5 * (u_plus + u_minus) - (0.5 * self.C) * (f_plus - f_minus)

            elif scheme == "LaxW":
                self.u_all[i] = u - 0.5*self.C*(u_plus-u_minus) + 0.5*self.C*self.C*(u_plus - 2*u + u_minus)

        return

    def burgers(self, scheme):
        for i in range(1,self.nt):
            u = self.u_all[i-1]
            u_plus = np.roll(self.u_all[i-1], 1)
            u_minus = np.roll(self.u_all[i-1], -1)

            if scheme == "LaxF":
                f_plus = 0.5*u_plus*u_plus
                f_minus = 0.5*u_minus*u_minus
                self.u_all[i] = 0.5 * (u_plus + u_minus) - (0.5 * self.C) * (f_plus - f_minus)

            elif scheme == "LaxW":
                t1 = (u + u_plus)*(u_plus**2 - u**2)
                t2 = (u + u_minus)*(u**2 - u_minus**2)
                self.u_all[i] = u - 0.25*self.C*(u_plus**2-u_minus**2) + 0.125*(self.C**2)*(t1 - t2)

            elif scheme == "RichtM":
                uhalf_plus = 0.5*(u_plus + u) - 0.5*self.C*(u_plus**2 - u**2)
                uhalf_minus = 0.5*(u + u_minus) - 0.5*self.C*(u**2 - u_minus**2)
                self.u_all[i] = u - 0.5*self.C*(uhalf_plus**2 - uhalf_minus**2)

            elif scheme == "Godunov":
                self.u_all[i] = u + self.C*(nf(u, u_plus) - nf(u_minus, u))
        return

def nf(u,v):
    ustar = np.zeros(len(u))
    for i in range(len(u)):
        if u[i] >= v[i]:
            if (u[i]+v[i])*0.5 > 0:
                ustar[i] = u[i]
            else:
                ustar[i] = v[i]
        else:
            if u[i] > 0:
                ustar[i] = u[i]
            elif v[i] < 0:
                ustar[i] = v[i]
            else:
                ustar[i] = 0.0

    return ustar


w = Wave(10, 50, -1.0)
w.profile("test3_toro")
w.burgers("Godunov")
plt.plot(w.x, w.u_all[-1], 'ko')
plt.show()
