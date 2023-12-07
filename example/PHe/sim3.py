import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
import pandas as pd
import time
import os
from Py3BR import threebodies, plotters, analysis, util
from Py3BR.constants import *
from inputs import *

if __name__ == '__main__':
    # t0 = time.time()
    n_traj = 10
    result = []
    long_out = 'results/it_long.txt'
    short_out = 'results/it_short.txt'
    
    # results = run1(**input_dict)
    # print(results)
    #--------------For Loop-----------------#
    b = np.arange(0,4,2)
    attrs = ['wn','t'] # Keep track of attributes
    for i in b:
        input_dict['b0'] = i
        for j in range(n_traj):
            t0 = time.time()
            output = threebodies.runOneT(*attrs,**input_dict)
            output['time'] = time.time() - t0
            result.append(output)
    df = pd.DataFrame(result)
    print(df)
    cols = ['e','b','r12','r23','r31','nd','nc','rej','time'] # Attributes to sum over
    counts = df.loc[:,cols].groupby(['e','b']).sum() # Sum over e,b
    print(counts)

    #----------------Parallel----------------#
    # t0 = time.time()
    # with mp.Pool(processes=os.cpu_count()) as p:
    #     event = [p.apply_async(threebodies.runOneT, args = ('wn','t',),kwds=(input_dict)) for i in range(n_traj)]
    #     for res in event:
    #         result.append(res.get())
    #         df = pd.DataFrame([res.get()])
    # df = pd.DataFrame(result)
    # counts = df.loc[:,:'rej'].groupby(['e','b']).sum() # sum counts
    # counts['time'] = time.time() - t0
    # df.to_csv(long_out, mode = 'a',
    #           header = os.path.isfile(short_out) == False or os.path.getsize(short_out) == 0)
    # # counts.to_csv(short_out, mode = 'a',
    # #               header = os.path.isfile(short_out) == False or os.path.getsize(short_out) == 0)

    #------------Iterate b------------------#
    # b_int = 2
    # # while True:
    # while True:
    #     with mp.Pool(processes=os.cpu_count()) as p:
    #         event = [p.apply_async(threebodies.runOneT, kwds=(input_dict)) for i in range(n_traj)]
    #         for res in event:
    #             result.append(res.get())
    #             df = pd.DataFrame([res.get()])
    #     df_short = pd.DataFrame(result)
    #     counts = df_short.loc[:,:'rej'].groupby(['e','b']).sum() # sum counts
    #     counts['time'] = time.time() - t0
    #     counts.xs(input_dict['b0'], level=1, drop_level=False).to_csv(short_out, mode = 'a',
    #                     header = os.path.isfile(short_out) == False or os.path.getsize(short_out) == 0)
    #     opac = analysis.opacity(short_out).reset_index(level=1)
    #     if opac[opac['b'] == input_dict['b0']]['pAB'].values[-1] <= 1e-2:
    #         break
    #     input_dict['b0']+=b_int # Iterate b 
    #-------------Analysis------------------#
    mu0 = 6504.062019864895 # PHe2 reduced mass
    # analysis.opacity('../../Tests/PHe/short.txt',mode = 'w',output = '../../Tests/PHe/k3.txt')
    # analysis.cross_section('../../Tests/PHe/short.txt', mode = 'w',output = '../../Tests/PHe/k3.txt')
    # analysis.k3('../../Tests/PHe/short.txt',mu0, mode = 'w',output = '../../Tests/PHe/k3.txt') # Rates
    
    # #----------Run one and plot-----------#
    # traj = threebodies.TBR(**input_dict)
    # traj.iCond()
    # print(traj.mu0)
    # a = traj.runT()
    # #2-d Trajectory Plot
    # fig,ax = plt.subplots()
    # plotters.traj_plt(traj)
    # # 3-d Trajectory Plot
    # plt.figure(2)
    # plotters.traj_3d(traj)
    # plt.show()
