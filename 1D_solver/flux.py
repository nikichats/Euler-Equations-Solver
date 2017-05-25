import numpy as np
import math as m
import copy

def LF(gamma, P):
    """ Returns fluxes for Lax Friedrich scheme (input is primitive variables)"""
    F0 = P[0, :]*P[1, :]
    F1 = P[0, :]*P[1, :]**2 + P[2, :]
    F2 = P[1, :]*(0.5*P[0, :]*P[1, :]**2 + (gamma/(gamma-1))*P[2, :])
    F = np.array([F0, F1, F2])
    return F

def Gdnv(gamma, P):
    """ Returns fluxes for Godunov scheme """
    F0 = P[0, :]*P[1, :]
    F1 = P[0, :]*P[1, :]**2 + P[2, :]
    F2 = P[1, :]*(0.5*P[0, :]*P[1, :]**2 + (gamma/(gamma-1))*P[2, :])
    F = np.array([F0, F1, F2])
    return F


