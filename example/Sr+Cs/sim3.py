import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp
import pandas as pd
import time
import os
from Py3BR import threebodies, plotters
from inputs import *


if __name__ == '__main__':
    t0 = time.time()
    n_traj = 100
    result = []
    long_out = '1k/long.csv'
    short_out = '1k/short.csv'
    
    #----------------Parallel----------------#
    # with mp.Pool(processes=os.cpu_count()) as p:
    #     event = [p.apply_async(threebodies.runOneT, kwds=(input_dict)) for i in range(n_traj)]
    #     for res in event:
    #         result.append(res.get())
    #         df = pd.DataFrame([res.get()])
    #         df.to_csv(long_out, mode = 'a', index=False,
    #                     header = os.path.isfile(long_out) == False or os.path.getsize(long_out) == 0)
    # df_short = pd.DataFrame(result)
    # counts = df_short.loc[:,:'rej'].groupby(['e','b']).sum() # sum counts
    # counts['time'] = time.time() - t0
    # counts.to_csv(short_out, mode = 'a',
    #               header = os.path.isfile(short_out) == False or os.path.getsize(short_out) == 0)

    #-------------Run one and plot-----------#
    traj = threebodies.TBR(**input_dict)
    a = traj.runT()
    #2-d Trajectory Plot
    fig,ax = plt.subplots()
    plotters.traj_plt(traj)
    # 3-d Trajectory Plot
    plt.figure(2)
    plotters.traj_3d(traj)
    plt.show()
