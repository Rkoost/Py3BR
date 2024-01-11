import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
from itertools import groupby
from operator import itemgetter
from Py3BR.constants import *

fact_sig = (Boh2m*1e2)**5 # Sigma factor
fact_k3 = (Boh2m*1e2)/ttos # k3 factor

def opacity(input, mode = 'a', sep = None, output = None):
    '''
    Calculate P(E,b) of a three-body recombination event.
    Input Parameters:
    input, str
        Path to input file. Should be in the form of short output from Py3BR, 
        with header (e,b,r12,r23,r31,nd,nc,rej,time)
    sep, str (optional)
        delimeter, input to pandas read_csv()
    output, str (optional)
        Path to output file, if saving. 
    '''
    if sep:
        df = pd.read_csv(input, sep = sep)
    else:
        df = pd.read_csv(input)
    cols = ['e','b','r12','r23','r31','nd','nc','rej']
    stats = df.loc[:,cols].groupby(['e','b']).sum()
    nTraj = stats.sum(axis=1) - stats['rej']
    pAB = (stats['r23']+stats['r31'])/nTraj
    pAB_err = np.sqrt(stats['r23'] + stats['r31'])/nTraj*np.sqrt((nTraj-(stats['r23']+stats['r31']))/nTraj)
    pBB = (stats['r12'])/nTraj
    pBB_err = np.sqrt(stats['r12'])/nTraj*np.sqrt((nTraj-(stats['r12']))/nTraj)
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
    bmax is found by P(bmax) <= tolerance, such that all P(b) up to P(bmax + n-1) <= tolerance also. 
    Input Parameters:
    input, str
        Path to input file. Should be in the form of short output from Py3BR, 
        with header (e,b,r12,r23,r31,nd,nc,rej,time)
    tol_AB, float (optional)
        Minimum value for P_AB(b), should be close to 0.
    n_AB, int (optional)
        number of times in a row `P_AB(b) <= tolerance_AB` is satisfied
    tol_BB, float (optional)
        Minimum value of P_BB(b), should be close to 0.
    n_BB, int (optional)
        number of times in a row `P_BB(b) <= tolerance_BB` is satisfied
    Outputs:
    bmax_AB, dict
        Dictionary mapping bmax of P_AB for each energy ({Ec:bmax})
    bmax_BB, dict
        Dictionary mapping bmax of P_BB for each energy ({Ec:bmax})
    '''
    opac = opacity(input).reset_index(level=1)
    # Find bmax of pAB for each energy
    bmax_AB = {}
    for i in opac.index.unique():
        try:
            df = opac.loc[i].copy()
            data = np.where(df['pAB'].values<=tol_AB)[0]
            for k, g in groupby(enumerate(data), lambda ix:ix[0]-ix[1]):
                # Find lists of consecutive indices satisfying P(b)<tolerance
                bl_AB = list(map(itemgetter(1), g))
                if len(bl_AB) >= n_AB: 
                    bmax_AB[i] = df['b'].values[bl_AB[0]]
                    break
        except IndexError:
            print(f"Max impact parameter not reached for E = {i} K. Increase impact parameter until P_AB(b) <= {tolerance_AB}, {n_AB} times in a row.")

    # Find bmax of pBB for each energy
    bmax_BB = {}
    for i in opac.index.unique():
        try:
            df = opac.loc[i].copy()
            data = np.where(df['pBB'].values<=tol_BB)[0]
            for k, g in groupby(enumerate(data), lambda ix:ix[0]-ix[1]):
                bl_BB = list(map(itemgetter(1), g))
                if len(bl_BB) >= n_BB:
                    bmax_BB[i] = df['b'].values[bl_BB[0]]
                    break
        except IndexError:
                print(f"Max impact parameter not reached for E = {i} K. Increase impact parameter until P_BB(b) <= {tolerance_BB}, {n_BB} times in a row.")
    return bmax_AB,bmax_BB

def cross_section(input, bmax_AB, bmax_BB, mode = 'w', sep = None, output = None):
    '''
    Calculate cross section sigma(E) of a three-body recombination event.
    Units: cm^5
    Input Parameters:
    input, str
        Path to input file. Should be in the form of short output from Py3BR, 
        with header (e,b,r12,r23,r31,nd,nc,rej,time)
    bmax_AB, dict
        Dictionary mapping bmax of P_AB for each energy ({Ec:bmax})
    bmax_BB, dict
        Dictionary mapping bmax of P_BB for each energy ({Ec:bmax})
    sep, str (optional)
        delimeter, input to pandas read_csv()
    output, str (optional)
        Path to output file, if saving. 
    '''
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
    sig_AB = integ_AB.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapz(g.pAB*g.b**4, x=g.b))*fact_sig
    sig_AB_err = integ_AB.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapz(g.pAB_err*g.b**4, x=g.b))*fact_sig
    sig_BB = integ_BB.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapz(g.pBB*g.b**4, x=g.b))*fact_sig
    sig_BB_err = integ_BB.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapz(g.pBB_err*g.b**4, x=g.b))*fact_sig
    sigma = pd.DataFrame([sig_AB,sig_AB_err,sig_BB,sig_BB_err], index=['sig_AB','sig_AB_err','sig_BB','sig_BB_err']).T
    if output:
        if sep:
            sigma.to_csv(f'{output}', mode = mode, sep = sep)
        else:
            sigma.to_csv(f'{output}', mode = mode)
    return sigma

def k3(input,mu0,  bmax_AB, bmax_BB,mode = 'w',sep = None, output = None):
    '''
    Calculate rate k3(E) of a three-body recombination event.
    Units: cm^6/s
    Input Parameters:
    input, str
        Path to input file. Should be in the form of short output from Py3BR, 
        with header (e,b,r12,r23,r31,nd,nc,rej,time)
    sep, str (optional)
        delimeter, input to pandas read_csv()
    output, str (optional)
        Path to output file, if saving. 
    '''
    sigma = cross_section(input,bmax_AB, bmax_BB, sep=sep,output=None)
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
    mu0 = 6504.062019864895 # PHe
    # mu0 = 120631.7241 #SrCs
    bmax_AB,bmax_BB=bmax('../example/PHe/results/short.txt',tolerance_AB=0,n_AB=2,tolerance_BB=0,n_BB=2)
    rate = k3('../example/PHe/results/short.txt',mu0,bmax_AB=bmax_AB, bmax_BB=bmax_BB)
   
    # # Plot opacities
    # opac = opac.reset_index(level=1)
    # for i in opac.index.unique()[:5]:
    # i=1000
    # df = opac.loc[i]
    # plt.figure(1)
    # plt.errorbar(df['b'], df['pAB'], df['pAB_err'], fmt='.',label = f'{i} K')
    # plt.legend()
    # plt.title(r'$P_{AB}$')
    # plt.figure(2)
    # plt.errorbar(df['b'], df['pBB'], df['pBB_err'], fmt='.', label = f'{i} K')
    # plt.legend()
    # plt.title(r'$P_{BB}$')
    # plt.show()
    
    # Plot rates
    plt.figure(3)
    plt.errorbar(rate.reset_index(0)['e'],rate.reset_index(0)['k3_AB'],rate.reset_index(0)['k3_AB_err'], capsize = 3,fmt = '_')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'E$(E_H)$')
    plt.ylabel(r'$k_3 (cm^6/s)$')
    plt.title('Rate AB')
    plt.figure(4)
    plt.errorbar(rate.reset_index(0)['e'],rate.reset_index(0)['k3_BB'],rate.reset_index(0)['k3_BB_err'],  capsize = 3,fmt = '_')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Rate BB')
    plt.xlabel(r'E$(E_H)$')
    plt.ylabel(r'$k_3 (cm^6/s)$')
    plt.show()