import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
from Py3BR.constants import *

fact_sig = (Boh2m*1e2)**5 # Sigma factor
fact_k3 = (Boh2m*1e2)/ttos # k3 factor

def plot_opacity(input, sep = None, ax = None, output= False):    
    if ax == None:
        ax = plt.gca()
    if sep:
        df = pd.read_csv(input, sep = sep)
    else:
        df = pd.read_csv(input)
    stats = df.groupby(['e','b']).sum()
    nTraj = stats.sum(axis=1) - stats['rej']
    pAB = (stats['r23']+stats['r31'])/nTraj
    pAB_err = np.sqrt(stats['r23'] + stats['r31'])/nTraj*np.sqrt((nTraj-(stats['r23']+stats['r31']))/nTraj)
    pBB = (stats['r12'])/nTraj
    pBB_err = np.sqrt(stats['r12'])/nTraj*np.sqrt((nTraj-(stats['r12']))/nTraj)
    ax.errorbar(df['b'], pAB, pAB_err, fmt='.',label = f'{pAB}')
    ax.errorbar(df['b'], pBB, pBB_err, fmt='.', label = f'{pBB}')
    opacity = pd.DataFrame([pAB,pBB], index=['pAB','pBB'])
    if output:
        if sep:
            opacity.to_csv(f'{output}', mode = 'a', sep = sep)
        else:
            opacity.to_csv(f'{output}', mode = 'a')
        print(f'Opacity function saved to {output}.')
    return ax, opacity

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
    stats = df.groupby(['e','b']).sum()
    nTraj = stats.sum(axis=1) - stats['rej']
    pAB = (stats['r23']+stats['r31'])/nTraj
    pAB_err = np.sqrt(stats['r23'] + stats['r31'])/nTraj*np.sqrt((nTraj-(stats['r23']+stats['r31']))/nTraj)
    pBB = (stats['r12'])/nTraj
    pBB_err = np.sqrt(stats['r12'])/nTraj*np.sqrt((nTraj-(stats['r12']))/nTraj)
    opacity = pd.DataFrame([pAB,pAB_err,pBB,pBB_err], index=['pAB','pAB_err','pBB','pBB_err'])
    if output:
        if sep:
            opacity.to_csv(f'{output}', mode = mode, sep = sep)
        else:
            opacity.to_csv(f'{output}', mode = mode)
    return opacity.T

def cross_section(input, mode = 'a',sep = None, output = None):
    '''
    Calculate cross section sigma(E) of a three-body recombination event.
    Input Parameters:
    input, str
        Path to input file. Should be in the form of short output from Py3BR, 
        with header (e,b,r12,r23,r31,nd,nc,rej,time)
    sep, str (optional)
        delimeter, input to pandas read_csv()
    output, str (optional)
        Path to output file, if saving. 
    '''
    stats = opacity(input,sep=sep,output=None)
    # Trapezoidal integrals
    stats = stats.reset_index(level=1)
    sig_AB = stats.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapz(g.pAB*g.b**4, x=g.b))*fact_sig
    sig_AB_err = stats.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapz(g.pAB_err*g.b**4, x=g.b))*fact_sig
    sig_BB = stats.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapz(g.pBB*g.b**4, x=g.b))*fact_sig
    sig_BB_err = stats.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapz(g.pBB_err*g.b**4, x=g.b))*fact_sig
    sigma = pd.DataFrame([sig_AB,sig_AB_err,sig_BB,sig_BB_err], index=['sig_AB','sig_AB_err','sig_BB','sig_BB_err']).T
    if output:
        if sep:
            sigma.to_csv(f'{output}', mode = mode, sep = sep)
        else:
            sigma.to_csv(f'{output}', mode = mode)
    return sigma

def k3(input,mu0, mode = 'a', sep = None, output = None):
    '''
    Calculate rate k3(E) of a three-body recombination event.
    Input Parameters:
    input, str
        Path to input file. Should be in the form of short output from Py3BR, 
        with header (e,b,r12,r23,r31,nd,nc,rej,time)
    sep, str (optional)
        delimeter, input to pandas read_csv()
    output, str (optional)
        Path to output file, if saving. 
    '''
    sigma = cross_section(input,sep=sep,output=None)
    rate = sigma.copy()
    rate['P0'] = np.sqrt(rate.index*K2Har*2*mu0)
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
    # mu0 = 6504.062019864895
    mu0 = 120631.7241 #SrCs
    # sig = cross_section('../Tests/PHe/short.txt', output = '../Tests/PHe/sigma.txt')
    k3('../Tests/Sr+Cs/short.txt',mu0, mode = 'w', output = '../Tests/Sr+Cs/k3.txt')
