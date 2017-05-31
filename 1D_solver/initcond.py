# Initial conditions for the 1D_Euler.py program. Cases (1)-(5) are from Toro pg. 129

def sod():
    U_0 = [1.0, 0.125, 0.75, 0.0, 1.0, 0.1]
    return U_0

def toro_2():
    U_0 = [1.0, 1.0, -2.0, 2.0, 0.4, 0.4]
    return U_0

def toro_3():
    U_0 = [1.0, 1.0, 0.0, 0.0, 1000., 0.1]
    return U_0

def toro_4():
    U_0 = [1.0, 1.0, 0.0, 0.0, 0.01, 100.]
    return U_0

def toro_5():
    U_0 = [5.99924, 5.99924, 12.5975, -6.19633, 460.894, 46.095]
    return U_0