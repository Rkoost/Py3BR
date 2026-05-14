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

# He2 parameters
rm = 0.29683e-9/BOH2M # position of min of V(r) : equilibrium length \approx 2.96 Angstrom
eps = 10.956 * K2HAR # epsilon /k_B: reduction factor of potential \approx well depth = 3.32e-5 hartree
A = 1.86924404e5
alpha = 10.5717543
beta = -2.007758779
c6 = 1.35186623
c8 = 0.41495143
c10 = 0.17151143
D = 1.438
v_He2 = He2_V(rm, eps, A, alpha, beta, c6, c8, c10, D)
dv_He2 = He2_dV(rm, eps, A, alpha, beta, c6, c8, c10, D)

De = 0.09027661
re = 2.04725623
beta = 1.09314253
b = 1.6059377
c = 0.00957858
As = 0.77802774
Bs = 6.15941548
Al = 2.03590553
Bl = 6.65881373
alpha = 1.3793
q = 1
v_He2plus = he2_plus(De, re, beta, b, c, As, Bs, Al, Bl, alpha, q)
dv_He2plus = dhe2_plus(De, re, beta, b, c, As, Bs, Al, Bl, alpha, q)

v_funcs = (v_He2, v_He2plus, v_He2plus)
dv_funcs = (dv_He2, dv_He2plus, dv_He2plus)

masses = (4.0026, 4.0026, 4.0026) # He, He, He+
m1, m2, m3 = masses
E0 = 0.1 # Kelvin
R0 = 2000.0
dR0 = 0.1*R0

# b_values = np.linspace(0, 20, 3)
# try:
#     run_b_scan(
#         b_range=b_values, num_traj_per_b=30,
#         masses=masses, E0=E0, R0=R0, v_funcs=v_funcs,
#         dv_funcs=dv_funcs,
#         summary_file='summary_results.csv',
#         save_detailed=True 
#     )
# except RuntimeError as e:
#     print(f'Multiprocessing error: {e}')


# import glob
# TARGET_SEED = 22606352
# search_pattern=f'results/*seed{TARGET_SEED}.npy'
# files = glob.glob(search_pattern)

# if files:
#     filename = files[0]
#     data = np.load(filename)]
#     plot_distances(data, TARGET_SEED, E0=0, b0=0)
#     plot_2d_motion(data, TARGET_SEED)
#     plot_3d_motion(data, m1, m2, m3, TARGET_SEED)
#     animate_3d_forces(data, masses, dv_funcs, scale=2000, filename=f'anim_seed{TARGET_SEED}.gif')
#     plt.show()
# else:
#     print("File not found.")


masses = (4.0026, 4.0026, 4.0026) # He, He, He+
m1, m2, m3 = masses
E0 = 10 # Kelvin
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
# t_stop, r_stop, r_tol, a_tol = 3, 3, 1e-11, 1e-13
task_data = m1, m2, m3, E0, b0, R0, v_funcs, dv_funcs, seed #, t_stop, r_stop, r_tol, a_tol, seed
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