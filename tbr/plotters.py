import numpy as np
import matplotlib.pyplot as plt


def plot_distances(data, seed=None, E0=None, b0=None, savefig=None):
    """
    Plots the inter-particle distances (r12, r23, r31) over time.
    Assumes data columns: [Time, r12, r23, r31, ...]
    """
    time = data[:, 0]
    r12  = data[:, 1]
    r23  = data[:, 2]
    r31  = data[:, 3]

    plt.figure(figsize=(10, 6))
    plt.plot(time, r12, label='$r_{12}$', color='blue')
    plt.plot(time, r23, label='$r_{23}$', color='green')
    plt.plot(time, r31, label='$r_{31}$', color='red')
    
    plt.title(f"Trajectory Distances") #(Seed: {seed})\n$E_0={E0}, b={b0}$")
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Distance (a.u.)")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if savefig:
        plt.savefig(f'{savefig}.png', dpi=300)


def plot_3d_motion(data, m1, m2, m3, seed, savefig=None):
    time=data[:,0]
    rho1 = data[:,4:7]
    rho2 = data[:,7:10]

    # print(rho1[0], rho2[0])

    C1 = m1/(m1+m2)
    C2 = m2/(m1+m2)
    mtot = m1 + m2 + m3
    mu3 = m3/mtot
    mu12 = (m1+m2)/mtot

    r1 = (-C2*rho1 - mu3*rho2).T
    r2 = (C1*rho1 - mu3*rho2).T
    r3 = (mu12*rho2).T

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Time trace
    ax.plot(r1[0], r1[1], r1[2], label=f'Particle 1', color='blue', alpha = 0.6)
    ax.plot(r2[0], r2[1], r2[2], label=f'Particle 2', color='red', alpha = 0.6)
    ax.plot(r3[0], r3[1], r3[2], label=f'Particle 3', color='green', alpha = 0.6)

    # Start points (Circles)
    ax.scatter(r1[0,0], r1[1,0], r1[2,0], color='blue', marker='o', s=50)
    ax.scatter(r2[0,0], r2[1,0], r2[2,0], color='red', marker='o', s=50)
    ax.scatter(r3[0,0], r3[1,0], r3[2,0], color='green', marker='o', s=50)
    
    # End points (X)
    ax.scatter(r1[0,-1], r1[1,-1], r1[2,-1], color='blue', marker='x', s=50)
    ax.scatter(r2[0,-1], r2[1,-1], r2[2,-1], color='red', marker='x', s=50)
    ax.scatter(r3[0,-1], r3[1,-1], r3[2,-1], color='green', marker='x', s=50)
    
    ax.set_xlabel('X (Bohr)')
    ax.set_ylabel('Y (Bohr)')
    ax.set_zlabel('Z (Bohr)')
    ax.set_title(f'3D Trajectory (Seed: {seed})')
    ax.legend()
    
    # Set equal aspect ratio 
    # Calculate bounds
    all_x = np.concatenate((r1[0], r2[0], r3[0]))
    all_y = np.concatenate((r1[1], r2[1], r3[1]))
    all_z = np.concatenate((r1[2], r2[2], r3[2]))
    
    max_range = np.array([all_x.max()-all_x.min(), 
                          all_y.max()-all_y.min(), 
                          all_z.max()-all_z.min()]).max()/2.0
    
    mid_x = (all_x.max()+all_x.min())*0.5
    mid_y = (all_y.max()+all_y.min())*0.5
    mid_z = (all_z.max()+all_z.min())*0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    if savefig:
        plt.savefig(f'{savefig}.png', dpi=300)

def plot_opac(df, energies, suffix='BB', fmt = '.', label = None, save_path=None):
    # plt.figure()
    colors = plt.cm.viridis(np.linspace(0,1,len(energies)))
    
    for i, e_val in enumerate(energies):
        subset = df[df['e'] == e_val].sort_values('b')

        if subset.empty:
            print(f'Skipping E = {e_val}: No data found.')
            continue
        
        plt.errorbar(subset['b'], subset[f'p_{suffix}'], yerr=subset[f'p_{suffix}_err'],
                        fmt = fmt, capsize=3, label=f'{label}, E = {e_val} K', color = colors[i])
        
        if f'bmax_{suffix}' in df.columns:
            bmax = df[df['e'] == e_val].sort_values('b')[f'bmax_{suffix}'].values[0]
            plt.axvline(bmax, 0, 1, color = colors[i])
    
    plt.xlabel('Impact Parameter ($a_0$)')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout
    if save_path:
        plt.savefig(save_path)
        print(f'Plot saved: {save_path}')
    



def animate_3d_forces(result_dict, masses, dv_funcs, 
                      scale=2000, filename="trajectory_forces.mp4", 
                      fps=30, duration_seconds=10,
                      zoom=1.0,
                      azim=45,
                      elev=30):
    """
    Animates the 3D trajectory and force vectors.
    """
    if not result_dict['success']:
        print("Cannot animate: Trajectory was unsuccessful.")
        return
    
    def get_cartesian_vectors(rho_vec, m1, m2, m3):
        """
        Converts Jacobi coordinates (rho1, rho2) to Cartesian coordinates (r1, r2, r3)
        in the Center of Mass frame.
        """
        # Total Mass
        M = m1 + m2 + m3
        
        # Unpack Jacobi vectors (Shape: 3)
        rho1 = rho_vec[0:3]
        rho2 = rho_vec[3:6]
        
        # Mass ratios
        # r2 - r1 = rho1
        # r3 - COM_12 = rho2
        
        # Derived from geometric definitions of Jacobi coords:
        r3 = (m1 + m2) / M * rho2
        r1 = -(m2 / (m1 + m2)) * rho1 - (m3 / M) * rho2
        r2 = (m1 / (m1 + m2)) * rho1 - (m3 / M) * rho2
        
        return r1, r2, r3
    
    # Extract Data
    raw_times = result_dict['times']
    raw_rho = result_dict['positions_rho'].T  # Shape (T, 6)
    m1, m2, m3 = masses
    dv12_func, dv23_func, dv31_func = dv_funcs

    # Calculate total number of frames
    num_frames = int(fps*duration_seconds)

    # Create a linear time grid 
    t_min, t_max = raw_times[0], raw_times[-1]
    t_uniform = np.linspace(t_min, t_max, num_frames)

    # Interpolate the position data onto new time grid
    # Iterate over 6 columns of rho
    rho_uniform = np.zeros((num_frames, 6))
    for i in range(6):
        rho_uniform[:, i] = np.interp(t_uniform, raw_times, raw_rho[:, i])
    
    r1_hist = np.zeros((num_frames, 3))
    r2_hist = np.zeros((num_frames, 3))
    r3_hist = np.zeros((num_frames, 3))

    for i in range(num_frames):
        r1_hist[i], r2_hist[i], r3_hist[i] = get_cartesian_vectors(rho_uniform[i], m1, m2, m3)
    
    # Setup Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    
    # Calculate limits based on the full Cartesian history
    all_x = np.concatenate([r1_hist[:,0], r2_hist[:,0], r3_hist[:,0]])
    all_y = np.concatenate([r1_hist[:,1], r2_hist[:,1], r3_hist[:,1]])
    all_z = np.concatenate([r1_hist[:,2], r2_hist[:,2], r3_hist[:,2]])

    # x0 = np.concatenate([r1_hist[:,0][0], r2_hist[:,0][0], r3_hist[:,0][0]])
    # y0 = np.concatenate([r1_hist[:,1][0], r2_hist[:,1][0], r3_hist[:,1][0]])
    # z0 = np.concatenate([r1_hist[:,2][0], r2_hist[:,2][0], r3_hist[:,2][0]])
    
    max_range = np.array([max(all_x)-min(all_x), max(all_y)-min(all_y), max(all_z)-min(all_z)]).max() / 2.0
    mid_x, mid_y, mid_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
    
    visible_range = (max_range*1.1)/zoom
    # Add a little margin
    # ax.set_xlim(mid_x - visible_range, mid_x + visible_range)
    # ax.set_ylim(mid_y - visible_range, mid_y + visible_range)
    # ax.set_zlim(mid_z - visible_range, mid_z + visible_range)
    # ax.set_xlim(mid_x - visible_range, mid_x + visible_range)
    # ax.set_ylim(mid_y - visible_range, mid_y + visible_range)
    # ax.set_zlim(mid_z - visible_range, mid_z + visible_range)



    ax.set_xlabel('X (a.u.)'); ax.set_ylabel('Y (a.u.)'); ax.set_zlabel('Z (a.u.)')
    
    # --- Initialize Plot Elements ---
    
    # 1. Particles (Scatter) - Initialize with frame 0
    colors = ['r', 'g', 'b']
    scat = ax.scatter([r1_hist[0,0], r2_hist[0,0], r3_hist[0,0]], 
                      [r1_hist[0,1], r2_hist[0,1], r3_hist[0,1]], 
                      [r1_hist[0,2], r2_hist[0,2], r3_hist[0,2]], 
                      s=100, c=colors, depthshade=False)
    
    # 2. Traces (Lines) - Initialize empty lines
    trails = []
    for i in range(3):
        # Create empty line with matching color, slightly transparent
        line, = ax.plot([], [], [], color=colors[i], lw=1.5, alpha=0.6)
        trails.append(line)

    # 3. Text and Quivers
    txt = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    # txt2 = ax.text2D(0.05, 0.85, "", transform=ax.transAxes)
    # txt3 = ax.text2D(0.05, 0.75, "", transform=ax.transAxes)
    # txt4 = ax.text2D(0.05, 0.65, "", transform=ax.transAxes)
    quivers = []
    def update(frame_idx):
        # 1. Clear old quivers (Matplotlib 3D quivers are hard to update in-place)
        nonlocal quivers
        for q in quivers:
            q.remove()
        quivers = []
        
        t = t_uniform[frame_idx]
        # Get current positions from pre-calculated history
        r1, r2, r3 = r1_hist[frame_idx], r2_hist[frame_idx], r3_hist[frame_idx]
        
        # --- A. Update Particles ---
        scat._offsets3d = ([r1[0], r2[0], r3[0]], [r1[1], r2[1], r3[1]], [r1[2], r2[2], r3[2]])
        txt.set_text(f"Time: {t:.2f} a.u.")

        # Slice history up to current frame. 
        # We use .T (transpose) because plot() expects separate arrays for x, y, z.
        # P1 Trace
        trails[0].set_data(r1_hist[:frame_idx+1, 0], r1_hist[:frame_idx+1, 1])
        trails[0].set_3d_properties(r1_hist[:frame_idx+1, 2])
        # P2 Trace
        trails[1].set_data(r2_hist[:frame_idx+1, 0], r2_hist[:frame_idx+1, 1])
        trails[1].set_3d_properties(r2_hist[:frame_idx+1, 2])
        # P3 Trace
        trails[2].set_data(r3_hist[:frame_idx+1, 0], r3_hist[:frame_idx+1, 1])
        trails[2].set_3d_properties(r3_hist[:frame_idx+1, 2])
        
        # --- C. Calculate Forces (Same logic) ---
        v12 = r2 - r1; d12 = np.linalg.norm(v12); u12 = v12/d12 if d12 > 1e-9 else np.zeros(3)
        v23 = r3 - r2; d23 = np.linalg.norm(v23); u23 = v23/d23 if d23 > 1e-9 else np.zeros(3)
        v31 = r1 - r3; d31 = np.linalg.norm(v31); u31 = v31/d31 if d31 > 1e-9 else np.zeros(3)
        
        f12 = dv12_func(d12) * u12 * scale
        f23 = dv23_func(d23) * u23 * scale
        f31 = dv31_func(d31) * u31 * scale

        # txt2.set_text(f"F12: {f12} a.u.")
        # txt3.set_text(f"F23: {f23} a.u.")
        # txt4.set_text(f"F31: {f31} a.u.")
        
        
        # --- D. Draw Arrows ---
        # Pair 1-2 (Magenta)
        quivers.append(ax.quiver(r1[0], r1[1], r1[2], f12[0], f12[1], f12[2], color='m', linestyle='--'))
        quivers.append(ax.quiver(r2[0], r2[1], r2[2], -f12[0], -f12[1], -f12[2], color='m', linestyle='--'))
        # Pair 2-3 (Cyan)
        quivers.append(ax.quiver(r2[0], r2[1], r2[2], f23[0], f23[1], f23[2], color='c', linestyle='-'))
        quivers.append(ax.quiver(r3[0], r3[1], r3[2], -f23[0], -f23[1], -f23[2], color='c', linestyle='-'))
        # Pair 3-1 (Orange)
        quivers.append(ax.quiver(r3[0], r3[1], r3[2], f31[0], f31[1], f31[2], color='orange', linestyle=':'))
        quivers.append(ax.quiver(r1[0], r1[1], r1[2], -f31[0], -f31[1], -f31[2], color='orange', linestyle=':'))
        
        # Return changed artists (needed if blit=True, good practice anyway)
        return scat, txt, *quivers, *trails

    print(f"Generating animation ({num_frames} frames over {duration_seconds}s)...")
    ani = FuncAnimation(fig, update, frames=range(num_frames), interval=1000/fps, blit=True)
    
    # Save
    if filename.endswith('.gif'):
        ani.save(filename, writer='pillow', fps=fps)
    else:
        # Requires ffmpeg installed
        try:
            ani.save(filename, writer='ffmpeg', fps=fps, extra_args=['-vcodec', 'libx264'])
        except Exception as e:
            print(f"FFMpeg error: {e}. Saving as GIF instead.")
            ani.save(filename.replace('.mp4', '.gif'), writer='pillow', fps=fps)
            
    print(f"Saved to {filename}")
    plt.close()


def plot_energy_trace(result_dict, masses, v_funcs):
    """
    Plots the Total Energy, Kinetic Energy, and Potential Energy over time.
    Also plots the relative energy error to diagnose integrator stability.
    
    Parameters:
    - result_dict: The dictionary returned by the worker.
    - masses: Tuple (m1, m2, m3).
    - potential_func: The function V(r) used in the simulation (e.g., He2_V).
    """
    if not result_dict['success']:
        print("Cannot plot energy: Trajectory was unsuccessful.")
        return

    #Extract data
    times = result_dict['times']
    # Shape: (N_steps, 6) -> We transpose for easier unpacking if needed
    rho_traj = result_dict['positions_rho'].T 
    p_traj = result_dict['momenta_p'].T
    
    print(p_traj.shape)
    m1, m2, m3 = masses
    M_tot = m1 + m2 + m3
    
    # Reduced Masses for Jacobi Coordinates
    mu12 = (m1 * m2) / (m1 + m2)
    mu23 = (m2 * m3) / (m2 + m3)
    mu31 = (m3 * m1) / (m3 + m1)
    mu312 = ((m1 + m2) * m3) / M_tot

    # T12 = np.zeros(len(times))
    # T23 = np.zeros(len(times))
    # T31 = np.zeros(len(times))
    
    # V12 = np.zeros(len(times))
    # V23 = np.zeros(len(times))
    # V31 = np.zeros(len(times))
    
    # E12 = np.zeros(len(times))
    # E23 = np.zeros(len(times))
    # E31 = np.zeros(len(times))
    
    # for i in range(len(times)):
    #     p1x, p1y, p1z = p_vec[0:3]


    # Arrays to store energy terms
    T_vals = np.zeros(len(times))
    V_vals = np.zeros(len(times))
    E_vals = np.zeros(len(times))

    # print(f"\n--- DEBUGGING ENERGY TRACE (First 5 steps) ---")
    # print(f"{'Time':<10} | {'r12':<10} {'r23':<10} | {'V(r12)':<12} {'V_tot':<12} | {'T':<12}")
    
    print("Calculating energy trace...")
    
    # Iterate through trajectory
    for i in range(len(times)):
        # -- Kinetic Energy (T) --
        # T = p1^2 / (2*mu1) + p2^2 / (2*mu2)
        p_vec = p_traj[i]
        p1_sq = np.sum(p_vec[0:3]**2)
        p2_sq = np.sum(p_vec[3:6]**2)
        
        T = (p1_sq / (2 * mu12)) + (p2_sq / (2 * mu312))
        
        # -- Potential Energy (V) --
        # We need distances r12, r23, r31
        rho_vec = rho_traj[i]
        rho1 = rho_vec[0:3]
        rho2 = rho_vec[3:6]
        
        # r12 is just magnitude of rho1
        r12 = np.linalg.norm(rho1)
        
        # Helper conversions for r23 and r31
        # mass ratios
        c1 = m1 / (m1 + m2)
        c2 = m2 / (m1 + m2)
        
        v23 = rho2 - c1 * rho1
        r23 = np.linalg.norm(v23)
        
        v31 = rho2 + c2 * rho1
        r31 = np.linalg.norm(v31)
        
        vf12, vf23, vf31 = v_funcs
        # Sum pairwise potentials
        V = vf12(r12) + vf23(r23) + vf31(r31)
        
        T_vals[i] = T
        V_vals[i] = V
        E_vals[i] = T + V

        # if i % 100 == 0:
        #     print(f"{times[i]:<10.4f} | {r12:<10.4f} {r23:<10.4f} | {vf12(r12):<12.4e} {V:<12.4e} | {T:<12.4e}")

    # Plotting
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot A: Total, Kinetic, Potential
    ax1.plot(times, E_vals, label='Total Energy ($E$)', color='black', linewidth=2)
    # ax1.plot(times, T_vals, label='Kinetic ($T$)', color='red', alpha=0.6, linestyle='--')
    # ax1.plot(times, V_vals, label='Potential ($V$)', color='blue', alpha=0.6, linestyle='--')
    ax1.set_ylabel('Energy (Hartree)')
    ax1.set_title(f"System Energy vs Time (Seed: {result_dict.get('seed', 'N/A')})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # # Plot B: Relative Error (Log scale is often useful here)
    # # Error = |(E(t) - E0) / E0|
    # E0 = E_vals[0]
    # # Add a small epsilon to avoid divide by zero if E0 is exactly 0
    # rel_error = np.abs((E_vals - E0)/E0)
    
    # ax2.plot(times, rel_error, color='purple')
    # ax2.set_ylabel('Relative Energy Error $|(E-E_0)/E_0|$')
    # ax2.set_xlabel('Time (a.u.)')
    # ax2.set_yscale('log') # Log scale helps see small errors
    # ax2.grid(True, alpha=0.3)
    # ax2.set_title(f"Conservation Accuracy (Mean Error: {np.mean(rel_error):.2e})")

    plt.tight_layout()
    
    # # Save or Show
    # filename = f"energy_trace_{result_dict.get('seed', 'test')}.png"
    # plt.savefig(filename, dpi=150)
    # print(f"Energy plot saved to {filename}")
    # plt.close()

    # time = data[:, 0]
    # r12  = data[:, 1]
    # r23  = data[:, 2]
    # r31  = data[:, 3]

    # data_block = np.vstack([
    #                 solution['times'], 
    #                 r12, r23, r31,
    #                 solution['positions_rho'], 
    #                 solution['momenta_p']
    #             ]).T

def plot_relative_e(result_dict, masses, vfuncs, title='Relative E'):
    if not result_dict['success']:
        print("Cannot plot energy: Trajectory was unsuccessful.")
        return

    # 1. Extract Data
    t = result_dict['times']
    # Shape: (N_steps, 6) -> We transpose for easier unpacking if needed
    rho_traj = result_dict['positions_rho'] 
    p_traj = result_dict['momenta_p']
    
    m1, m2, m3 = masses
    v12, v23, v31 = vfuncs
    
    rho1x, rho1y, rho1z, rho2x, rho2y, rho2z = rho_traj
    p1x, p1y, p1z,p2x, p2y, p2z = p_traj

    mu12 = m1*m2/(m1+m2)
    mu23 = m2*m3/(m2+m3)
    mu31 = m3*m1/(m3+m1)
    mu312 = m3*(m1+m2)/(m1+m2+m3)

    C1 = m1/(m1+m2)
    C2 = m2/(m1+m2)

    r12 = np.sqrt(rho1x**2+rho1y**2+rho1z**2)
    r23 = np.sqrt((rho2x - C1*rho1x)**2
                  + (rho2y - C1*rho1y)**2 
                  + (rho2z - C1*rho1z)**2)
    r31 = np.sqrt((rho2x + C2*rho1x)**2
                  + (rho2y + C2*rho1y)**2 
                  + (rho2z + C2*rho1z)**2)

    # Momenta components
    p23_x = mu23*p2x/mu312-mu23*p1x/m2
    p23_y = mu23*p2y/mu312-mu23*p1y/m2
    p23_z = mu23*p2z/mu312-mu23*p1z/m2
    p31_x = mu31*p2x/mu312+mu31*p1x/m1
    p31_y = mu31*p2y/mu312+mu31*p1y/m1
    p31_z = mu31*p2z/mu312+mu31*p1z/m1

    p12 = np.sqrt(p1x**2+p1y**2+p1z**2)
    p23 = np.sqrt(p23_x**2 + p23_y**2 + p23_z**2)
    p31 = np.sqrt(p31_x**2 + p31_y**2 + p31_z**2)

    K12 = p12**2/2/mu12
    K23 = p23**2/2/mu23
    K31 = p31**2/2/mu31

    V12 = [v12(n) for n in r12]
    V23 = [v23(n) for n in r23]
    V31 = [v31(n) for n in r31]

    # Convert to arrays
    V12_vals = np.array(V12)
    V23_vals = np.array(V23)
    V31_vals = np.array(V31)
    
    E12_vals = K12 + V12_vals
    E23_vals = K23 + V23_vals
    E31_vals = K31 + V31_vals


    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))

    # Plot Zero Line (Threshold for binding)
    plt.axhline(0, color='black', linewidth=1.5, linestyle='-', label='Unbound Threshold (E=0)')

    # Plot Pairs
    plt.plot(t, E12_vals, label='Pair 1-2', color='m', alpha=0.8)
    plt.plot(t, E23_vals, label='Pair 2-3', color='c', alpha=0.8)
    plt.plot(t, E31_vals, label='Pair 3-1', color='orange', alpha=0.8)

    # Formatting
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Pairwise Internal Energy (Hartree)')
    plt.title(f'Version: {title}')
    # plt.title(f"Pairwise Binding Energies (Seed: {result_dict.get('seed')})\nNegative = Bound State")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()

# def plot_2d_motion(data, seed):
#     """
#     Plots the motion of the relative coordinates in the XY plane.
#     """
#     # Columns 4,5,6 are rho1 (internuclear vector r)
#     # Columns 7,8,9 are rho2 (particle 3 relative to CM of 1-2)
#     rho1_x = data[:, 4]
#     rho1_y = data[:, 5]
#     rho1_z = data[:, 6]
    
#     rho2_x = data[:, 7]
#     rho2_y = data[:, 8]
#     rho2_z = data[:, 9]

#     plt.figure(figsize=(8, 8))
    
#     # Plot Path of Vector r (Diatomic vibration/rotation)
#     plt.plot(rho1_x, rho1_y, label='Diatomic Vector ($r_{12}$)', alpha=0.6, lw=1)
    
#     # Plot Path of Vector R (Incoming particle trajectory)
#     plt.plot(rho2_x, rho2_y, label='Incoming Particle ($R_{3-12}$)', color='black', lw=2)
    
#     # Mark Start and End
#     plt.scatter([rho2_x[0]], [rho2_y[0]], color='green', marker='o', label='Start')
#     plt.scatter([rho2_x[-1]], [rho2_y[-1]], color='red', marker='x', label='End')
#     plt.scatter([rho1_x[0]], [rho1_y[0]], color='green', marker='o')
#     plt.scatter([rho1_x[-1]], [rho1_y[-1]], color='red', marker='x')

#     plt.title(f"2D Relative Motion (Seed: {seed})")
#     plt.xlabel("X (a.u.)")
#     plt.ylabel("Y (a.u.)")
#     plt.axis('equal') # Crucial to see true geometry
#     plt.legend()
#     plt.grid(True)