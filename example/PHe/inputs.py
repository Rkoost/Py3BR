from Py3BR.potentials import *
from Py3BR.constants import *

m1 = 4.002602*u2me  # Helium mass
m2 = 4.002602*u2me
m3 = 30.974 *u2me   # P

E0 = 0.1    # collision energy (K)
b0 = 0
R0 = 550
dR0 = 25

# Potential parameters in atomic units
v12 = lambda x: He2_V(x)
dv12 = lambda x: He2_dV(x)
v23 = lambda x: LJ(x,m=12,n=6,cm=9.9721e+5, cn=14.69)
dv23 = lambda x: dLJ(x,m=12,n=6,cm=9.9721e+5, cn=14.69)
v31 = lambda x: LJ(x,m=12,n=6,cm=9.9721e+5, cn=14.69)
dv31 = lambda x: dLJ(x,m=12,n=6,cm=9.9721e+5, cn=14.69)

input_dict = {'m1': m1, 'm2':m2, 'm3':m3,
              'E0': E0, 'b0': b0, 'R0':R0,
              'dR0': dR0, 
              'v12':v12, 'v23':v23,'v31':v31,
              'dv12':dv12, 'dv23':dv23, 'dv31':dv31,
              'seed': None, 
              'integ': {'t_stop': 10, 
                      'r_stop': 2,
                      'r_tol': 1e-10,
                      'a_tol': 1e-12}}

if __name__ == '__main__':
    # Find dissociation energy
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import fsolve
    root = fsolve(dv12,6)[0]
    print(v12(root)/K2Har)
    x = np.linspace(5,20,1000)
    plt.plot(x,v12(x))
    plt.plot(root,v12(root),'o')
    plt.show()
