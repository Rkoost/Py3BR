from Py3BR.potentials import *
from Py3BR.constants import *

m3 = 87.62 *u2me   # Sr
m2 = 132.91*u2me  # Cs
m1 = 132.91*u2me

E0 = 1         # collision energy
b0 = 0
R0 = 2000
ran = 50 # Randomization range
v23 = lambda x: LJ(x,n=4,m=8, cn = 200, cm =  1.67e+6) #Sr+Cs
dv23 = lambda x: dLJ(x,n=4,m=8, cn = 200, cm =  1.67e+6)
v12 = lambda x: LJ(x,n=6,m=12,cn=6.64e+3,cm=6.63e+8) # CsCs
dv12 = lambda x: dLJ(x,n=6,m=12,cn=6.64e+3,cm=6.63e+8)
v31 = lambda x: LJ(x,n=4,m=8, cn = 200, cm =  1.67e+6) # Sr+Cs
dv31 = lambda x: dLJ(x, n=4,m=8, cn = 200, cm =  1.67e+6)

input_dict = {'m1': m1, 'm2':m2, 'm3':m3,
              'E0': E0, 'b0': b0, 'R0':R0,
              'range': ran,
              'v12':v12, 'v23':v23,'v31':v31,
              'dv12':dv12, 'dv23':dv23, 'dv31':dv31,
              'seed': None,
              'integ': {'t_stop': 10,
                      'r_stop': 2,
                      'r_tol': 1e-10,
                      'a_tol': 1e-12}}