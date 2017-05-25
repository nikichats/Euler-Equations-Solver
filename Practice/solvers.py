import numpy as np
import matplotlib.pyplot as plt

def lin_adv(courant,u_0,n,u_all,scheme):
    """" Propagates a wave using the linear advection equation with velocity a """
    u_all[0] = u_0
    max_val = np.array([])
    for i in range(1,n):
        u_plus = np.roll(u_all[i-1],1)
        u_minus = np.roll(u_all[i-1],-1)

        if scheme == 2:
            u_all[i] = 0.5*(u_plus + u_minus) - courant*(u_all[-1] - u_minus) #First order upwind
        elif scheme == 3:
            u_all[i] = 0.5*(u_plus + u_minus) - 0.5*courant*(u_plus - u_minus) #Lax-Friedrich
        elif scheme == 4:
            u_all[i] = u_all[i-1] + courant*(u_plus - u_minus) - (courant**2)*(u_plus - 2*u_all[i-1] + u_minus)  #Lax-Wendroff
        else:
            u_all[i] = u_all[i - 1] - 0.5 * courant * (u_plus - u_minus) # Centered

        plt.plot(u_all[i],label='%i'%i)
        max_val = np.append(max_val,max(u_all[i]))
    return u_all, max_val

def burgers(courant,u_0,n,u_all):
    """ Propagates a wave using Burgers' equation """
    u_all[0] = u_0
    max_val = np.array([])
    for i in range(1,n):
        u_plus = np.roll(u_all[i-1],1)
        u_minus = np.roll(u_all[i-1],-1)

        # u_all[i] =  u_all[i-1] - 0.5*courant*(u_plus**2 - u_minus**2)
        u_all[i] = u_all[i-1] - 0.5*courant*u_all[i-1]*(u_plus - u_minus)

        plt.plot(u_all[i])
        max_val = np.append(max_val,max(u_all[i]))
    return u_all, max_val

def split_advection(a,dx,dt,n,u_0,u_all):
    u_all[0] = u_0

    if len(a) != len(u_0):
        print "a vector not the same length as velocity profile vector"
        return
    else:
        for i in range(1, n):
            u_plus = np.roll(u_all[i - 1], 1)
            u_minus = np.roll(u_all[i - 1], -1)
            lambda_plus = max(u_all[i-1], 0)
            lambda_minus = min(u_all[i-1], 0)

            u_all[i] = (1 + lambda_minus - lambda_plus)*u_all[i-1] + lambda_plus*u_minus - lambda_minus*u_plus

            plt.plot(u_all[i])

    return u_all
