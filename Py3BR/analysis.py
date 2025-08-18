import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
from itertools import groupby
from operator import itemgetter
from Py3BR.constants import *

fact_sig = (Boh2m*1e2)**5 # Sigma factor
fact_k3 = (Boh2m*1e2)/ttos # k3 factor

def opacity(input,  sep = None, output = None,mode = 'a',):
    '''
    Calculate P(E,b) of a three-body recombination event.
    Input Parameters:
    input, str
        Path to input file. Should be in the form of short output from Py3BR, 
        with header (e,b,n12,n23,n31,nd,nc,rej,time)
    sep, str (optional)
        delimeter, input to pandas read_csv()
    output, str (optional)
        Path to output file, if saving. 
    '''
    cols = ['e','b','n12','n23','n31','nd','nc','rej']
    if isinstance(input, pd.DataFrame):
        df = input
    elif sep:
        df = pd.read_csv(input, sep = sep, usecols=cols)
    else:
        df = pd.read_csv(input, usecols=cols)
    stats = df.groupby(['e','b']).sum()
    nTraj = stats.sum(axis=1) - stats['rej']
    pAB = (stats['n23']+stats['n31'])/nTraj
    pAB_err = np.sqrt(stats['n23'] + stats['n31'])/nTraj*np.sqrt((nTraj-(stats['n23']+stats['n31']))/nTraj)
    pBB = (stats['n12'])/nTraj
    pBB_err = np.sqrt(stats['n12'])/nTraj*np.sqrt((nTraj-(stats['n12']))/nTraj)
    opacity = pd.DataFrame([pAB,pAB_err,pBB,pBB_err], index=['pAB','pAB_err','pBB','pBB_err']).T
    if output:
        if sep:
            opacity.to_csv(f'{output}', mode = mode, sep = sep)
        else:
            opacity.to_csv(f'{output}', mode = mode)
    return opacity

def bmax(input, tol_AB = 1e-3, n_AB = 3, tol_BB = 1e-3, n_BB=3):
    '''
    Function to suggest bmax for each collision energy. Inspect each value against the opacity
    function to ensure a good bmax, which should capture most non-zero probabilities. 
    bmax is found by P(b=bmax)/P(b=0) <= tolerance, such that all P(b) up to P(b=bmax+n-1)/P(b=0) <= tolerance also. 
    Input Parameters:
    input, str
        Path to input file. Should be in the form of short output from Py3BR, 
        with header (e,b,n12,n23,n31,nd,nc,rej,time)
    tol_AB, float (optional)
        Minimum value for P_AB(b=bmax)/P_AB(b=0), should be close to 0.
    n_AB, int (optional)
        number of times in a row tol_AB is satisfied
    tol_BB, float (optional)
        Minimum value of P_BB(b=bmax)/P_BB(b=0), should be close to 0.
    n_BB, int (optional)
        number of times in a row tol_BB is satisfied
    Outputs:
    bmax_AB, dict
        Dictionary mapping bmax of P_AB for each energy ({Ec:bmax})
    bmax_BB, dict
        Dictionary mapping bmax of P_BB for each energy ({Ec:bmax})
    '''
    # If input is an opacity dataframe
    if isinstance(input, pd.DataFrame):
        if 'pAB' in input.columns: # If probability, opacity dataframe
            opac = input.set_index('e')
    else:
        opac = opacity(input).reset_index(level=1)
    # Find bmax of pAB for each energy
    bmax_AB = {}
    # Iterate over energy
    for i in opac.index.unique():
        try:
            df = opac.loc[i].copy()
            # Find indices satisfying tolerance
            data = np.where((df['pAB']/(max(df['pAB']))).values<=tol_AB)[0]
            # If data is empty
            if (len(data) == 0):
                bmax_AB[i] = df['b'].values[-1]
            for k, g in groupby(enumerate(data), lambda ix:ix[0]-ix[1]):
                # Find lists of consecutive indices satisfying P(b)<tolerance
                bl_AB = list(map(itemgetter(1), g))
                # If number of consecutive indices > n_AB
                if len(bl_AB) >= n_AB: 
                    # Use first instance
                    bmax_AB[i] = df['b'].values[bl_AB[0]]
                    break
                # If not found, use last impact parameter
                else:
                    bmax_AB[i] = df['b'].values[-1]
        except IndexError:
            print(f"Max impact parameter not reached for E = {i} K. Increase impact parameter until P_AB(b=bmax)/P_AB(b=0) <= {tol_AB}, {n_AB} times in a row.")

    # Find bmax of pBB for each energy
    bmax_BB = {}
    for i in opac.index.unique():
        try:
            df = opac.loc[i].copy()
            data = np.where((df['pBB']/(max(df['pBB']))).values<=tol_BB)[0]
            # If data is empty
            if (len(data) == 0):
                bmax_BB[i] = df['b'].values[-1]
            for k, g in groupby(enumerate(data), lambda ix:ix[0]-ix[1]):
                bl_BB = list(map(itemgetter(1), g))
                if len(bl_BB) >= n_BB:
                    bmax_BB[i] = df['b'].values[bl_BB[0]]
                    break
                # If not found, use last impact parameter
                else:
                    bmax_BB[i] = df['b'].values[-1]
        except IndexError:
                print(f"Max impact parameter not reached for E = {i} K. Increase impact parameter until P_BB(b=bmax)/P_BB(b=0) <= {tol_BB}, {n_BB} times in a row.")
    return bmax_AB,bmax_BB

def cross_section(input, bmax_AB, bmax_BB, mode = 'w', sep = None, output = None):
    '''
    Calculate cross section sigma(E) of a three-body recombination event.
    Units: cm^5
    Input Parameters:
    input, str
        Path to input file. Should be in the form of short output from Py3BR, 
        with header (e,b,n12,n23,n31,nd,nc,rej,time)
    bmax_AB, dict
        Dictionary mapping bmax of P_AB for each energy ({Ec:bmax})
    bmax_BB, dict
        Dictionary mapping bmax of P_BB for each energy ({Ec:bmax})
    sep, str (optional)
        delimeter, input to pandas read_csv()
    output, str (optional)
        Path to output file, if saving. 
    '''
    # If input is an opacity dataframe
    if isinstance(input, pd.DataFrame):
        if 'pAB' in input.columns: # If probability, opacity dataframe
            opac = input.set_index('e')
    else:
        opac = opacity(input,sep=sep).reset_index(level=1)
    # Create new dataframe for AB up to bmax_AB
    data_AB = []
    for k,v in bmax_AB.items():
        data_AB.append(opac.loc[k][opac.loc[k]['b'] <= v])
    integ_AB = pd.concat(data_AB)

    # Create new dataframe for BB up to bmax_BB
    data_BB = []
    for k,v in bmax_BB.items():
        data_BB.append(opac.loc[k][opac.loc[k]['b'] <= v])
    integ_BB = pd.concat(data_BB)

    # Trapezoidal integrals
    sig_AB = integ_AB.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapezoid(g.pAB*g.b**4, x=g.b))*fact_sig
    sig_AB_err = integ_AB.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapezoid(g.pAB_err*g.b**4, x=g.b))*fact_sig
    sig_BB = integ_BB.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapezoid(g.pBB*g.b**4, x=g.b))*fact_sig
    sig_BB_err = integ_BB.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapezoid(g.pBB_err*g.b**4, x=g.b))*fact_sig
    sigma = pd.DataFrame([sig_AB,sig_AB_err,sig_BB,sig_BB_err], index=['sig_AB','sig_AB_err','sig_BB','sig_BB_err']).T
    if output:
        if sep:
            sigma.to_csv(f'{output}', mode = mode, sep = sep)
        else:
            sigma.to_csv(f'{output}', mode = mode)
    return sigma

def k3(input, mu0,  bmax_AB, bmax_BB,mode = 'w',sep = None, output = None):
    '''
    Calculate rate k3(E) of a three-body recombination event.
    Units: cm^6/s
    Input Parameters:
    input, str
        Path to input file. Should be in the form of short output from Py3BR, 
        with header (e,b,n12,n23,n31,nd,nc,rej,time)
    mu0, float
        m3*m1*m2/(m1+m2+m3)
    sep, str (optional)
        delimeter, input to pandas read_csv()
    output, str (optional)
        Path to output file, if saving. 
    '''
    sigma = cross_section(input,bmax_AB, bmax_BB, sep=sep)
    rate = sigma.copy()
    rate['P0'] = np.sqrt(rate.index*K2Har*2*mu0) # P = sqrt(2*mu*E), E is index of sigma
    rate = rate.loc[:,:'sig_BB_err'].multiply(rate['P0']/mu0*fact_k3, axis = 'index')
    rate = rate.rename(columns = {'sig_AB':'k3_AB','sig_AB_err':'k3_AB_err',
                           'sig_BB':'k3_BB','sig_BB_err':'k3_BB_err'})
    if output:
        if sep:
            rate.to_csv(f'{output}', mode = mode, sep = sep)
        else:
            rate.to_csv(f'{output}', mode = mode)
    return rate

if __name__ == '__main__':
    # mu0 = 6504.062019864895 # PHe
    # mu0 = 120631.7241 #SrCs

    short_path = 'C:/Users/Rian/Documents/research/jesusgroup/Py3BR_mw/Py3BR/example/I_Noble/INe/results/short.txt'
    # opac = opacity('../example/Sr+Cs/results/short.txt')
    bmax_AB,bmax_BB=bmax(short_path,tol_AB=1e-2,n_AB=2,tol_BB=1e-3,n_BB=1)
    # bmax_AB[40000]=39.1
    # sigma = cross_section('../example/Sr+Cs/results/short.txt',bmax_AB=bmax_AB, bmax_BB=bmax_BB).reset_index()
    # rate = k3('../example/Sr+Cs/results/short.txt',mu0,bmax_AB=bmax_AB, bmax_BB=bmax_BB).reset_index()

    # # Plot opacities
    # opac = opac.reset_index(level=1)
    # el = [20000,40000,65000]
    # for i in el:
    #     df = opac.loc[i]
    #     plt.figure(1)
    #     plt.scatter(df['b'], df['pAB'], marker='.',label = f'{i} K')
    #     plt.vlines(bmax_AB[i],0,df['pAB'].values.max())
    # plt.legend()
    # plt.title(r'$P_{AB}$')
    # # plt.figure(2)
    # # plt.errorbar(df['b'], df['pBB'], df['pBB_err'], fmt='.', label = f'{i} K')
    # # plt.legend()
    # # plt.title(r'$P_{BB}$')
    # # plt.show()
    
    # # Plot rates
    # plt.figure(3)
    # plt.errorbar(rate['e'],rate['k3_AB'],rate['k3_AB_err'], capsize = 3,fmt = '_', label='py')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r'E$(E_H)$')
    # plt.ylabel(r'$k_3 (cm^6/s)$')
    # plt.title('Rate AB')
    # plt.legend()
    # plt.figure(4)
    # plt.errorbar(rate['e'],rate['k3_BB'],rate['k3_BB_err'],  capsize = 3,fmt = '_')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('Rate BB')
    # plt.xlabel(r'E$(E_H)$')
    # plt.ylabel(r'$k_3 (cm^6/s)$')
    # plt.show()
