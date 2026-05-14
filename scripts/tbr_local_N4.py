import os
import sys
import time
import numpy as np

# Find the absolute path to the root directory (one level up from 'scripts')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insert it at the front of Python's search path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tbr.simulator import *
from tbr.constants import *
from tbr.potentials import *
from tbr.plotters import *

# N4p_de = 52.2 kcal/mol
N4p_de = 0.083186 # (Eh)
N4p_re = 1.382*ANG2BOH # (A -> a0)
N2_alpha = 1.710 # A^3
N4p_c4 = N2_alpha*(ANG2BOH**3)/2 # C4 = alpha/2
N4p_c8 = N4p_c4**2/(4*N4p_de)

# J. Chem. Phys., Vol. 95, No.8, 15 October 1991
# doi: 10.1063/1.461604
# ESMSV potential
N2He_de =  2.15*0.0000367493 # (meV -> Eh)
N2He_re = 3.66*ANG2BOH # (A -> a0)
N2He_c6 = 6093*(3.67493e-5)*ANG2BOH**6 # (meV A^6 -> Eh*a0^6)
N2He_c12 =  N2He_c6**2/(4*N2He_de)

N2pHe_de = 133.12*0.0000046 # cm^-1 -> Eh
N2pHe_re = 6.199
He_alpha = 0.208 # A^3
N2pHe_c4 = He_alpha*ANG2BOH**3/2 
N2pHe_c8 = N2pHe_c4**2/(4*N2pHe_de)

N4_de = 173.9 # kcal/mol
N4_re = 1.386*ANG2BOH # (A)
N4_c6 = 1
N4_c12 = 1

v_N4p = LJ(n=4, m=8, cn = N4p_c4, cm = N4p_c8)
dv_N4p = dLJ(n=4, m=8, cn = N4p_c4, cm = N4p_c8)

v_N2He = LJ(n=6, m=12, cn = N2He_c6, cm = N2He_c12)
dv_N2He = dLJ(n=6, m=12, cn = N2He_c6, cm = N2He_c12)

v_N2pHe = LJ(n=4, m=8, cn = N2pHe_c4, cm = N2pHe_c8)
dv_N2pHe = dLJ(n=4, m=8, cn = N2pHe_c4, cm = N2pHe_c8)

# N2 + N2p + He
v_funcs = (v_N4p, v_N2pHe, v_N2He)
dv_funcs = (dv_N4p, dv_N2pHe, dv_N2He)
masses = (2*14.007, 2*14.007, 4.0026) # N2, N2+, He

m1, m2, m3 = masses
E0 = 100 # Kelvin
R0 = 500.0
dR0 = 0.1*R0
b0 = 0

mu0 = np.sqrt(m1*m2*m3/(m1+m2+m3))*U2ME
P0 = np.sqrt(2*E0*K2HAR*mu0)
print(f'P0 = {P0:.2e} for E0 = {E0} K')
# seed = 1239823
# seed = 10964947 # Reactive
seed = int(np.random.random()*8923) + 102381
# seed = 108000 # Reactive 0.1 K
# seed = 105447 # Reactive 0.001 K
print(f'Running trajectory with seed {seed}...')

# print(np.sqrt(2*E0*K2HAR/mu0))
t_stop, r_stop, r_tol, a_tol = 3, 3, 1e-11, 1e-13
task_data = m1, m2, m3, E0, b0, R0, dR0, v_funcs, dv_funcs, t_stop, r_stop, r_tol, a_tol, seed
t0 = time.time()
solution = run_trajectory_worker(task_data=task_data)
tf = time.time()
print(f'Trajectory run in {tf-t0} s')
n_res = solution['n_res']
times = solution['times']
rho_vec = solution['positions_rho']
p_vec = solution['momenta_p']
if solution['success'] and len(solution['times']) > 0:

    r12, r23, r31 = get_distances_from_solution(
                np.vstack([rho_vec, p_vec]), m1, m2
            )
    
    data_block = np.vstack([
                solution['times'], 
                r12, r23, r31,
                solution['positions_rho'], 
                solution['momenta_p']
            ]).T
    
    # print(n_res)
    # print(rho_vec)
    plot_distances(data_block, seed, E0, b0)
    # plot_2d_motion(data_block, seed)
    plot_3d_motion(data_block, m1, m2, m3, seed)
    # plot_relative_e(solution, masses, v_funcs)
    plot_energy_trace(solution, masses, v_funcs)
    # animate_3d_forces(solution, masses, dv_funcs, scale = 0,
    #                   filename='trajectory_forces7.gif', duration_seconds=10,
    #                   azim=0,elev=30)
    plt.show()