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
#         save_detailed=True 
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

t0 = time.time()
solution = run_trajectory_worker(task_data=task_data)
tf = time.time()
print(f'Trajectory run in {tf-t0} s')
n_res = solution['n_res']
times = solution['times']
rho_vec = solution['positions_rho']
p_vec = solution['momenta_p']
if solution['success'] == True:
    r12, r23, r31 = get_distances_from_solution(
                np.vstack([rho_vec, p_vec]), m1, m2
            )

    data_block = np.vstack([
                solution['times'], 
                r12, r23, r31,
                solution['positions_rho'], 
                solution['momenta_p']
            ]).T
    print(f'Result (n12, n23, n31, nd, nc): {n_res}')
    plot_distances(solution, masses)
    plot_3d_motion(solution, masses)
    # plot_relative_e(solution, masses, v_funcs)