import numpy as np
import multiprocess as mp
import pandas as pd
import time
import os
from Py3BR import threebodies, plotters, analysis, util
from Py3BR.constants import *

def runN(n_traj,input_dict, mode = 'parallel',cpus=os.cpu_count(), attrs = None, short_out = None, long_out = None):
    '''
    Run N trajectories.
    Inputs:
    n_traj, int
        number of trajectories
    input_dict, dictionary
        input dictionary containing all input data
    mode, str: 'parallel' or 'serial' (optional)
        default to parallel
    cpus, int (optional)
        number of cpus for parallel computing.
        Defaults to os.cpu_count().
    attrs, list of strings (optional)
        extra attributes to store in long output file. 
    short_out, string (optional)
        path to short output file
    long_out, string (optional)
        path to long output file. Store extra data here.
    Returns:
    df, pandas dataframe
        long output with one line per trajectory
    counts, pandas dataframe
        short output for analysis
    '''
    t0 = time.time()
    result = []
    if mode == 'parallel':
        with mp.Pool(processes=cpus) as p:
            if attrs:
                event = [p.apply_async(threebodies.runOneT, args = (*attrs,),kwds=(input_dict)) for i in range(n_traj)]
            else:
                event = [p.apply_async(threebodies.runOneT,kwds=(input_dict)) for i in range(n_traj)]
            for res in event:
                result.append(res.get())
    if mode == 'serial':
        for i in range(n_traj):
            if attrs:
                output = threebodies.runOneT(*attrs,**input_dict)
            else:
                output = threebodies.runOneT(**input_dict)
            result.append(output)
    df = pd.DataFrame(result)
    counts = df.loc[:,:'rej'].groupby(['e','b']).sum() # sum counts
    counts['time'] = time.time() - t0
    # Long output to track attributes of each trajectory
    if long_out:
        df.to_csv(long_out, mode = 'a',
                header = os.path.isfile(long_out) == False or os.path.getsize(long_out) == 0)
    # Short output for Py3BR.analysis files
    if short_out:
        counts.to_csv(short_out, mode = 'a',
                    header = os.path.isfile(short_out) == False or os.path.getsize(short_out) == 0)
    return df, counts
