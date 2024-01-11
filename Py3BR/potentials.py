import numpy as np
from Py3BR.constants import *

def morse(r,de = 1., alpha = 1., re = 1.):
    '''
    Usage: V = morse(x,**kwargs)

    Return a one-dimensional Morse potential:
    V(r) = De*(1-exp(-a(r-re)))^2 - De

    Keyword arguments:
    de, float
        dissociation energy (depth)
    alpha, float
        alpha = we*sqrt(mu/2De)
    re, float
        equilibrium length
    '''
    V = de*(1-np.exp(-alpha*(r-re)))**2 - de
    return V

def dmorse(r, de = 1., alpha = 1., re = 1.):
       return 2*alpha*de*(1-np.exp(-alpha*(r-re)))*np.exp(-alpha*(r-re))

def LJ(r, m=12, n = 6, cm = 1., cn=1.):
    '''Usage:
        V = lj(**kwargs)
    
    Return a one-dimensional general Lennard-Jones potential:
    V(r) = cm/r^m - cn/r^n

    Default is 12-6 Lennard-Jones. 

    Keyword arguments:
    cn, float
        long-range parameter
    cm, float
        short-range parameter
    '''
    V = cm/r**(m)-cn/r**(n)
    return V

def dLJ(r, m=12,n=6, cm = 1., cn = 1.):
    return -m*cm/r**(m+1)+n*cn/r**(n+1)

def He2_V(x):
    '''
    HFD-B3-FCI1 potential from 10.1088/0026-1394/27/4/005, 
    parameters from10.1103/PhysRevLett.74.1586.
    '''
    rm = 0.29683e-9/Boh2m # position of min of V(r) : equilibrium length \approx 2.96 Angstrom
    eps = 10.956 * K2Har # epsilon /k_B: reduction factor of potential \approx well depth = 3.32e-5 hartree
    A = 1.86924404e5
    alpha = 10.5717543
    beta = -2.007758779
    c6 = 1.35186623
    c8 = 0.41495143
    c10 = 0.17151143
    D = 1.438
    R = x/rm
    # Piecewise needs to be run element-wise
    if isinstance(x,list) or isinstance(x,np.ndarray):
        V = []
        for i in R:
            if i < D:
                V.append(eps*(A*np.exp(-alpha*i + beta*i**2) - 
                                (c6*i**(-6) + c8*i**(-8) + c10*i**(-10))*np.exp(-(D/i-1)**2)))
            else:
                V.append(eps*(A*np.exp(-alpha*i + beta*i**2) - 
                                (c6*i**(-6) + c8*i**(-8) + c10*i**(-10))))
    else:
        if R < D:
            V = (eps*(A*np.exp(-alpha*R + beta*R**2) - 
                            (c6*R**(-6) + c8*R**(-8) + c10*R**(-10))*np.exp(-(D/R-1)**2)))
        else:
            V= (eps*(A*np.exp(-alpha*R + beta*R**2) - 
                            (c6*R**(-6) + c8*R**(-8) + c10*R**(-10))))
    return V

def He2_dV(x):
    '''
    HFD-B3-FCI1 potential from 10.1088/0026-1394/27/4/005, 
    parameters from10.1103/PhysRevLett.74.1586.
    '''
    rm = 0.29683e-9/Boh2m # position of min of V(r) : equilibrium length \approx 2.96 Angestrom
    eps = 10.956 * K2Har # epsilon /k_B: reduction factor of potential \approx well depth = 3.32e-5 hartree
    A = 1.86924404e5
    alpha = 10.5717543
    beta = -2.007758779
    c6 = 1.35186623
    c8 = 0.41495143
    c10 = 0.17151143
    D = 1.438
    R = x/rm
    if R < D:
        dV = 1/rm*eps*(A*(-alpha + 2*beta*R)*np.exp(-alpha*R + beta*R**2) - 
                        (2*D*R**(-2)*(D/R-1)*np.exp(-(D/R-1)**2))*
                        (c6*R**(-6) + c8*R**(-8) + c10*R**(-10)) + 
                        (6*c6*R**(-7) + 8*c8*R**(-9) + 10*c10*R**(-11))*np.exp(-(D/R-1)**2))
            
    else:
        dV = 1/rm*eps*(A*(-alpha + 2*beta*R)*np.exp(-alpha*R + beta*R**2) + 
                        (6*c6*R**(-7) + 8*c8*R**(-9) + 10*c10*R**(-11)))
    return dV

if __name__ == '__main__':
    v= lambda x: LJ(x)
    x = np.linspace(1,10,10)
    print(v(x))
