import numpy as np
import numba
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import warnings
import os

from tbr.constants import *
from tbr.physics import *

warnings.filterwarnings("ignore", category=numba.NumbaWarning)

def get_distances_from_solution(y, m1, m2):
    C1 = m1 / (m1 + m2)
    C2 = m2 / (m1 + m2)
    
    rho1 = y[0:3, :]
    rho2 = y[3:6, :]
    
    r12 = np.sqrt(np.sum(rho1**2, axis=0))
    
    v23 = rho2 - C1 * rho1
    r23 = np.sqrt(np.sum(v23**2, axis=0))
    
    v31 = rho2 + C2 * rho1
    r31 = np.sqrt(np.sum(v31**2, axis=0))
    
    return r12, r23, r31

def run_trajectory_worker(task_data, t_stop=3, r_stop=3, r_tol=1e-10, a_tol=1e-12):
    (m1, m2, m3, E0, b0, R0, v_funcs, dv_funcs, seed) = task_data
    
    sim = TbrSimulator(
        m1=m1, m2=m2, m3=m3, E0=E0, b0=b0, R0=R0, 
        v_funcs=v_funcs, dv_funcs=dv_funcs, t_stop=t_stop, r_stop=r_stop, r_tol=r_tol, a_tol=a_tol
    )
    
    solution = sim.run_trajectory(seed=seed)

    # Initialize 'rejected' flag
    rejected = 0

    
    # Handling different return types (dict vs ODESolution)
    # If it's a dict, it might be a rejection (IC or Energy)
    is_dict = isinstance(solution, dict)

    msg = solution.get('message', None) if is_dict else None
    
    if is_dict and solution.get('rejected_ic', False):
        rejected = 1
        n_res = np.array([0,0,0,0,0])
        times, rho_vec, p_vec = np.array([]), np.array([]), np.array([])
        success = False
    elif is_dict and solution.get('rejected_energy', False):
        # NEW: Handle Energy Rejection
        rejected = 1
        print(f"Seed {seed} rejected due to energy drift: {solution.get('E_drift', 0):.2e}")
        n_res = np.array([0,0,0,0,0])
        times, rho_vec, p_vec = np.array([]), np.array([]), np.array([])
        success = False
    elif not hasattr(solution, 'success') or not solution.success:
        # Standard integration failure
        rejected = 1 
        n_res = np.array([0,0,0,0,0])
        if hasattr(solution, 't'):
             times = solution.t
             rho_vec = solution.y[:6, :]
             p_vec = solution.y[6:, :]
        else:
             times, rho_vec, p_vec = np.array([]), np.array([]), np.array([])
        success = False
    else:
        # Successful Run
        n_res = sim.analyze_categorize(solution)
        times = solution.t
        rho_vec = solution.y[:6, :]
        p_vec = solution.y[6:, :]
        success = True

    return {
        'seed': seed,
        'E0': E0,
        'b0': b0,
        'n_res': n_res, # [n12, n23, n31, nd, nc]
        'rejected': rejected,
        'message': msg,
        'success': success,
        'times': times,
        'positions_rho': rho_vec,
        'momenta_p': p_vec
    }

def run_parallel_batch(num_trajectories, masses, E0, b0, R0, v_funcs, dv_funcs, 
                       num_cores=None, save_detailed=False, output_dir='trajectories',
                       file_fmt='npy'):
    
    m1, m2, m3 = masses

    if num_cores is None:
        num_cores = mp.cpu_count()

    if 'SLURM_CPUS_PER_TASK' in os.environ:
        num_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    elif 'PBS_NUM_PPN' in os.environ:
        num_cores = int(os.environ['PBS_NUM_PPN'])
    else:
        num_cores = mp.cpu_count()

    np.random.seed(int(time.time()) + int(b0*1000)) 
    seeds = np.random.randint(0, 100000000, num_trajectories)

    tasks = [
        (m1, m2, m3, E0, b0, R0, v_funcs, dv_funcs, int(s))
        for s in seeds
    ]

    start_time = time.time()
    
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(run_trajectory_worker, tasks)

    rejections = [r['message'] for r in results if not r['success'] and r.get('message') is not None]
    if rejections:
        print(f"\n--- Total rejections due to V(R0) > 0.01*E0 for b0={b0}: {len(rejections)} ---")
        for msg in rejections[:5]: # Print first 5 to verify
            print(msg)

    end_time = time.time()
    total_runtime = end_time - start_time

    # Aggregation variables
    total_stats = np.zeros(5, dtype=int) # n12, n23, n31, nd, nc
    total_rejected = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for res in results:
        total_stats += res['n_res']
        total_rejected += res['rejected']

        if np.any(res['n_res'][:3]) == 1:
            print(f'Reaction found at E: {E0:.2e} K, b: {b0:.2f} a0, seed: {res["seed"]}')
        
        if save_detailed and res['success'] and len(res['times']) > 0:
            fname = os.path.join(output_dir, f"traj_E{E0:.2e}_b{b0:.2f}_seed{res['seed']}.csv")
            
            r12, r23, r31 = get_distances_from_solution(
                np.vstack([res['positions_rho'], res['momenta_p']]), m1, m2
            )
            
            # 16 Columns: [t, r12, r23, r31, rho(6), p(6)]
            data_block = np.vstack([
                res['times'], 
                r12, r23, r31,
                res['positions_rho'], 
                res['momenta_p']
            ]).T
            
            base_name = f"traj_E{E0:.2e}_b{b0:.2f}_seed{res['seed']}"
            
            if file_fmt == 'npy':
                fname = os.path.join(output_dir, base_name + ".npy")
                np.save(fname, data_block)
            else:
                fname = os.path.join(output_dir, base_name + ".csv")
                header_str = "Time,r12,r23,r31,rho1x,rho1y,rho1z,rho2x,rho2y,rho2z,p1x,p1y,p1z,p2x,p2y,p2z"
                np.savetxt(fname, data_block, delimiter=',', header=header_str, comments='')

    return {
        'E0': E0, 'b0': b0, 'counts': total_stats, 
        'rejected': total_rejected, 'runtime': total_runtime, 'raw_results': results 
    }

def run_b_scan(b_range, num_traj_per_b, masses, E0, R0, v_funcs, dv_funcs, 
               num_cores = None, summary_file='summary.csv', save_detailed=False):
    
    '''
    Run trajectories over a range of impact parameters for a given collision energy. 
    Saves output in summary_file. 
    Inputs:
        b_range: Range of impact parameter values to simulate over
        num_traj_per_b: Number of trajectories per impact parameter. 
        masses: List of masses (m1, m2, m3)
        E0: Collision energy (K)
        R0: Initial hyperradius (a0)
        v_funcs: List of interatomic potentials (V12, V23, V31)
        dv_funcs: List of interatomic potential derivatives w.r.t. distance (dV12, dV23, dV31)
        num_cores: Number of cpus to use for parallel computing
        summary_file: String containing file name for storing results
        save_detailed: If True, saves each trajectory as a numpy file for further analysis.
    
    '''
    print(f"Starting Scan over b0: {b_range}")
    print(f"Summary will be saved to: {summary_file}")
    
    for b0 in b_range:
        print(f"\n--- Running batch for b0 = {b0:.2f} ---")
        
        batch_res = run_parallel_batch(
            num_trajectories=num_traj_per_b,
            masses=masses, E0=E0, b0=b0, R0=R0,
            v_funcs=v_funcs, dv_funcs=dv_funcs,
            num_cores=num_cores, # Adjust cores here
            save_detailed=save_detailed
        )
        if summary_file:
            file_exists = os.path.isfile(summary_file)

            with open(summary_file, 'a') as f:
                if not file_exists:
                    f.write("e,b,n12,n23,n31,nd,nc,rej,time\n")
                 
                # Unpack stats
                n12, n23, n31, nd, nc = batch_res['counts']
                rej = batch_res['rejected']
                rt = batch_res['runtime']
                
                f.write(f'{E0:.4e},{b0:.4f},{n12},{n23},{n31},{nd},{nc},{rej},{rt:.4f}\n')

        print(f"  Finished b0={b0:.2f}. Runtime: {rt:.2f}s. Rej: {rej}")
        print(f"  Stats: n12={n12}, n23={n23}, n31={n31}, nd={nd}, nc={nc}")
    print('b-scan complete!')


class TbrSimulator:
    def __init__(self, m1, m2, m3, E0, b0, R0,
                 v_funcs, dv_funcs, t_stop = 3, r_stop = 3, r_tol = 1e-11, a_tol = 1e-13):
        
        self.m1, self.m2, self.m3 = m1*U2ME, m2*U2ME, m3*U2ME
        self.mtot = self.m1 + self.m2 + self.m3
        self.mu12 = self.m1 * self.m2 / (self.m1 + self.m2)
        self.mu23 = self.m2 * self.m3 / (self.m2 + self.m3)
        self.mu31 = self.m3 * self.m1 / (self.m3 + self.m1)
        self.mu312 = self.m3 * (self.m1 + self.m2) / self.mtot
        self.mu0 = np.sqrt(self.m1 * self.m2 * self.m3 / self.mtot)
        self.C1 = self.m1 / (self.m1 + self.m2)
        self.C2 = self.m2 / (self.m1 + self.m2)
        self.E0, self.b0, self.R0 = E0*K2HAR, b0, R0
        self.dR0 = self.R0*0.1

        self.v_funcs = v_funcs # (v12, v23, v31)
        self.dv_funcs = dv_funcs # (dv12, dv23, dv31)

        self.t_stop = t_stop
        self.r_stop = r_stop
        self.rtol = r_tol
        self.atol = a_tol

        self.t_max = self.R0 / np.sqrt(2 * self.E0 / self.mu312) * self.t_stop

    def get_total_energy(self, state):
        """
        Calculates Total Energy (T + V) for a single state vector (shape 12,).
        Used for conservation checks.
        """
        rho1 = state[0:3]
        rho2 = state[3:6]
        p1   = state[6:9]
        p2   = state[9:12]

        # --- Kinetic Energy (T) ---
        T = (np.sum(p1**2) / (2 * self.mu12)) + (np.sum(p2**2) / (2 * self.mu312))

        # --- Potential Energy (V) ---
        r12, r23, r31 = jac2cart_jit(state[:6], self.m1, self.m2)

        v12, v23, v31 = self.v_funcs
        V = v12(r12) + v23(r23) + v31(r31)

        return T + V

    def iCond(self, seed):
        v12, v23, v31 = self.v_funcs
        seeded_rng = np.random.default_rng(seed)
        w0, R, status, V_rej, R_rej, angles = iCond_jit(self.E0, self.b0, self.R0, self.dR0, 
                                          self.m1, self.m2, self.m3, v12, v23, v31, seeded_rng)

        return w0, R, status, V_rej, R_rej
    
    def avg_rel_dist(self, state):
        r12, r23, r31 = jac2cart_jit(state[:6], self.m1, self.m2)
        avg_rij = (r12 + r23 + r31)/3
        return avg_rij
    
    def avg_rel_p(self, state):
        p1x, p1y, p1z, p2x, p2y, p2z = state[6:]

        p12_x = p1x; p12_y = p1y; p12_z = p1z
        p23_x = self.mu23*p2x/self.mu312 - self.mu23*p1x/self.m2
        p23_y = self.mu23*p2y/self.mu312 - self.mu23*p1y/self.m2
        p23_z = self.mu23*p2z/self.mu312 - self.mu23*p1z/self.m2
        p31_x = self.mu31*p2x/self.mu312 + self.mu31*p1x/self.m1
        p31_y = self.mu31*p2y/self.mu312 + self.mu31*p1y/self.m1
        p31_z = self.mu31*p2z/self.mu312 + self.mu31*p1z/self.m1

        p12 = np.sqrt(p12_x**2 + p12_y**2 + p12_z**2)
        p23 = np.sqrt(p23_x**2 + p23_y**2 + p23_z**2)
        p31 = np.sqrt(p31_x**2 + p31_y**2 + p31_z**2)
        avg_pij = (p12 + p23 + p31)/3
        return avg_pij
    
    def run_trajectory(self, seed=None):
        w0, R_init, status, V_rej, R_rej = self.iCond(seed)

        # 1. Check Initial Conditions Status
        if np.any(np.isnan(w0)):
            if status == 1:
                return {'t': np.array([0.0]), 'y': np.full((12, 1), np.nan), 'success': False,'rejected_ic': True, 'rejected_energy': False}
            if status == 2:
                rejection_msg = f"Rejected: V(R0) > 0.01*E0, increase R0."
                return {
                    'success': False, 
                    'rejected_flag': 1, 
                    'message': rejection_msg,
                    'V_rej': V_rej
                }
            return {'t': np.array([0.0]), 'y': np.full((12, 1), np.nan), 'success': False,'rejected_ic': True, 'rejected_energy': False}
        # --- NEW: Calculate Initial Energy ---

        v12, v23, v31 = self.v_funcs
        dv12, dv23, dv31 = self.dv_funcs
        
        ivp_args = (
            self.mu12, self.mu312, self.m1, self.m2, 
            v12, v23, v31, 
            dv12, dv23, dv31, 
            self.R0, self.r_stop 
        )

        rho6D_0 = w0[:6]
        P6D_0 = w0[6:]
        u1 = np.sqrt(self.mu12/self.mu0)
        u2 = np.sqrt(self.mu312/self.mu0)

        # Bare representation
        y0 = np.concatenate((rho6D_0,
                             P6D_0[:3]*u1, P6D_0[3:]*u2))
        avg_r = self.avg_rel_dist(y0)
        avg_p = self.avg_rel_p(y0)
        t_max = avg_r*self.mu0/avg_p*self.t_stop
        t_span = [0, t_max]
        solution = solve_ivp(
            hamEq_jit, t_span, y0, method='RK45',events=stop_event_jit, 
            rtol=self.rtol, atol=self.atol, args=ivp_args 
        )

        if not solution.success or solution.y.size == 0:
             return solution

        initial_state = solution.y[:, 0]
        E_start = self.get_total_energy(initial_state)

        final_state = solution.y[:, -1]
        E_final = self.get_total_energy(final_state)
        
        energy_drift = abs(E_final - E_start)
        print(f'Energy drift {energy_drift} Hartree')

        if energy_drift > 1e-5:
            return {
                't': solution.t,
                'y': solution.y,
                'success': False,
                'rejected_ic': False,
                'rejected_energy': True,
                'E_drift': energy_drift
            }

        return solution
    
    def analyze_categorize(self, solution):
        if not solution['success'] or solution.y is None or solution.y.size == 0 or np.any(np.isnan(solution.y)):
            return np.array([0,0,0,0,0])
        
        final_w = solution.y[:,-1]
        v12, v23, v31 = self.v_funcs
        dv12, dv23, dv31 = self.dv_funcs

        result = checkBound_jit(final_w, self.m1, self.m2, self.m3, self.C1, self.C2, v12, v23, v31, dv12, dv23, dv31)

        return result
    
    


    
if __name__ == '__main__':
    from potentials import *
    from plotters import *

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

    b_values = np.linspace(0, 20, 3)

    masses = (4.0026, 4.0026, 4.0026) # He, He, He+
    m1, m2, m3 = masses
    E0 = 1e-02 # Kelvin
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
    task_data = m1, m2, m3, E0, b0, R0, dR0, v_funcs, dv_funcs, seed
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
        
        print(n_res)
        print(rho_vec)
        plot_distances(data_block, seed, E0, b0)
        # plot_2d_motion(data_block, seed)
        plot_3d_motion(data_block, m1, m2, m3, seed)
        plot_relative_e(solution, masses, v_funcs)
        # plot_energy_trace(solution, masses, v_funcs)
        # animate_3d_forces(solution, masses, dv_funcs, scale = 0,
        #                   filename='trajectory_forces7.gif', duration_seconds=10,
        #                   azim=0,elev=30)
        plt.show()