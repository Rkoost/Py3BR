# Script for running Py3BR calculations
import time
import numpy as np

from tbr.simulator import *
from tbr.constants import *
from tbr.potentials import *
from tbr.plotters import *

v_He2 = He2_V()
dv_He2 = He2_dV()

v_He2plus = He2plus_V()
dv_He2plus = He2plus_dV()

v_funcs = (v_He2, v_He2plus, v_He2plus)
dv_funcs = (dv_He2, dv_He2plus, dv_He2plus)

masses = (4.0026, 4.0026, 4.0026) # He, He, He+
m1, m2, m3 = masses
E0 = 0.1 # Kelvin
R0 = 2000.0
dR0 = 0.1*R0

###----- Run full b scan for E0 -----###
# b_values = np.linspace(0, 20, 3)
# try:
#     run_b_scan(
#         b_range=b_values, num_traj_per_b=30,
#         masses=masses, E0=E0, R0=R0, v_funcs=v_funcs,
#         dv_funcs=dv_funcs,
#         summary_file='summary_results.csv',
#         save_detailed=True,
#         bare=True
#     )
# except RuntimeError as e:
#     print(f'Multiprocessing error: {e}')

###----- Replot a previously run trajectory -----###
# import glob
# TARGET_SEED = 22606352
# search_pattern=f'trajectories/*seed{TARGET_SEED}.npy'
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


###----- Run one trajectory -----###
seed = int(np.random.random()*8923) + 102381
print(f'Running trajectory with seed {seed}...')

b0 = 0 # Impact parameter
task_data = m1, m2, m3, E0, b0, R0, v_funcs, dv_funcs, seed 

# Bare coordinates
t0 = time.time()
solution_bare = run_trajectory_worker(task_data=task_data, bare=True) # Default is bare=True
tf = time.time()
print(f'Bare trajectory run in {tf-t0} s')

n_res = solution_bare['n_res']
times = solution_bare['times']
rho_vec = solution_bare['positions_rho']
p_vec = solution_bare['momenta_p']


if solution_bare['success'] == True:
    r12, r23, r31 = get_distances_from_solution(
                np.vstack([rho_vec, p_vec]), m1, m2
            )

    data_block = np.vstack([
                solution_bare['times'], 
                r12, r23, r31,
                solution_bare['positions_rho'], 
                solution_bare['momenta_p']
            ]).T
    print(f'Result bare (n12, n23, n31, nd, nc): {n_res}')
    plot_distances(solution_bare, masses)
    plot_3d_motion(solution_bare, masses)

# # Mass weighted coordinates
# t0 = time.time()
# solution_mw = run_trajectory_worker(task_data=task_data, bare=False)
# tf = time.time()
# print(f'MW trajectory run in {tf-t0} s')
# n_res_mw = solution_mw['n_res']
# times_mw = solution_mw['times']
# rho_vec_mw = solution_mw['positions_rho']
# p_vec_mw = solution_mw['momenta_p']
# if solution_mw['success'] == True:
#     r12_mw, r23_mw, r31_mw = get_distances_from_solution(
#                 np.vstack([rho_vec_mw, p_vec_mw]), m1, m2
#             )
#     data_block_mw = np.vstack([
#                 solution_mw['times'], 
#                 r12_mw, r23_mw, r31_mw,
#                 solution_mw['positions_rho'], 
#                 solution_mw['momenta_p']
#             ]).T

#     print(f'Result mw (n12, n23, n31, nd, nc): {n_res_mw}')
#     plot_distances(solution_mw, masses)
#     plot_3d_motion(solution_mw, masses)

