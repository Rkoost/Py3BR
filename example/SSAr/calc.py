import numpy as np
from Py3BR import threebodies, analysis, potentials, constants

m1 = 32.06*constants.u2me  # S
m2 = 32.06*constants.u2me
m3 = 39.948*constants.u2me   # Ar

E0 = 300    # collision energy (K)
b0 = 0
R0 = 1150
dR0 = 115

# S-S
# c6 from 10.1021/acs.jctc.6b00361 
# de from J. Chem. Phys. 128, 204306
SS_c6 = 140
SS_de = 101.79/627.503 # Hartree
SS_rd = (SS_c6/2/SS_de)**(1/6) 
SS_c12 = SS_c6*SS_rd**6/2

# S-Ar
# c6 from 10.1021/acs.jctc.6b00361 
# de from J. Chem. Phys. 128, 204306
SAr_c6 = 95.27
SAr_de = 0.424/627.503 # Hartree
SAr_rd = (SAr_c6/2/SAr_de)**(1/6) 
SAr_c12 = SAr_c6*SAr_rd**6/2

# # Potential parameters in atomic units
v12 = lambda x: potentials.LJ(x,m=12,n=6,cm=SS_c12,cn=SS_c6)
dv12 = lambda x: potentials.dLJ(x,m=12,n=6,cm=SS_c12,cn=SS_c6)
v23 = lambda x: potentials.LJ(x,m=12,n=6,cm=SAr_c12, cn=SAr_c6)
dv23 = lambda x: potentials.dLJ(x,m=12,n=6,cm=SAr_c12, cn=SAr_c6)
v31 = lambda x: potentials.LJ(x,m=12,n=6,cm=SAr_c12, cn=SAr_c6)
dv31 = lambda x: potentials.dLJ(x,m=12,n=6,cm=SAr_c12, cn=SAr_c6)

input_dict = {'m1': m1, 'm2':m2, 'm3':m3,
              'E0': E0, 'b0': b0, 'R0':R0,
              'dR0': dR0, 
              'v12':v12, 'v23':v23,'v31':v31,
              'dv12':dv12, 'dv23':dv23, 'dv31':dv31,
              'seed': None, 
              'integ': {'t_stop': 3, 
                      'r_stop': 3,
                      'r_tol': 1e-10,
                      'a_tol': 1e-12}}

if __name__ == '__main__':
    short_out = 'results/sample/sample_short.txt' # Output file for analysis
    long_out = 'results/sample/sample_long.txt'
    bi = np.arange(0,20,5) # Range of impact parameters
    ntraj = 200
    # Run over range of impact parameters
    for b in bi:
        input_dict['b0'] = b # Update b
        # Run N trajectories and write to short, long output files
        threebodies.runN(ntraj, input_dict, attrs=['delta_e'], short_out = short_out, long_out = long_out)

    #-------------Analysis------------------#
    mu0 = np.sqrt(m1*m2*m3/(m1+m2+m3))
    # # Write analysis files
    analysis.opacity(short_out,output = 'results/sample/sample_opacity.txt') # opacity
    bmax_AB,bmax_BB = analysis.bmax(short_out,tol_AB=0.01,n_AB=2,tol_BB=0.01,n_BB=2) # estimate bmax
    analysis.cross_section(short_out, bmax_AB,bmax_BB,output = 'results/sample/sample_sigma.txt') # cross section
    rate = analysis.k3(short_out,mu0,bmax_AB,bmax_BB,output = 'results/sample/k3.txt') # rates