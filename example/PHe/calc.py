import numpy as np
from Py3BR import threebodies, analysis
from inputs import input_dict

short_out = 'results/sample/sample_short.txt' # Output file for analysis
long_out = 'results/sample/sample_long.txt'
bi = np.arange(0,50,5) # Range of impact parameters
ntraj = 200
# Run over range of impact parameters
for b in bi:
    input_dict['b0'] = b # Update b
    # Run N trajectories and write to short, long output files
    long,short = threebodies.runN(ntraj, input_dict, short_out = short_out, long_out = long_out)

#-------------Analysis------------------#
m1,m2,m3 = input_dict['m1'],input_dict['m2'],input_dict['m3']
mu0 = np.sqrt(m1*m2*m3/(m1+m2+m3))
# Write analysis files
analysis.opacity(short_out,mode = 'w',output = 'results/sample/sample_opacity.txt') # opacity
analysis.cross_section(short_out, mode = 'w',output = 'results/sample/sample_sigma.txt') # cross section
analysis.k3(short_out,mu0, mode = 'w',output = 'results/sample/sample_k3.txt') # rates