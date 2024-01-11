import numpy as np
from Py3BR import threebodies, analysis
from inputs import input_dict

short_out = 'results/sample/sample_short.txt' # Output file for analysis
long_out = 'results/sample/sample_long.txt'
bi = np.arange(0,1000,50) # Range of impact parameters
ntraj = 200
# Run over range of impact parameters
for b in bi:
    input_dict['b0'] = b # Update b
    # Run N trajectories and write to short, long output files
    threebodies.runN(ntraj, input_dict, attrs=['delta_e'], short_out = short_out, long_out = long_out)

#-------------Analysis------------------#
m1,m2,m3 = input_dict['m1'],input_dict['m2'],input_dict['m3']
mu0 = np.sqrt(m1*m2*m3/(m1+m2+m3))
# # Write analysis files
analysis.opacity(short_out,output = 'results/sample/sample_opacity.txt') # opacity
bmax_AB,bmax_BB = analysis.bmax(short_out,tol_AB=0,n_AB=1,tol_BB=0,n_BB=1) # estimate bmax
analysis.cross_section(short_out, bmax_AB,bmax_BB,output = 'results/sample/sample_sigma.txt') # cross section
rate = analysis.k3(short_out,mu0,bmax_AB,bmax_BB,output = 'results/sample/sample_k3.txt') # rates
# print(rate)