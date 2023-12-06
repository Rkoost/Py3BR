import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
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
    stats = opacity(input,sep=sep,output=None).reset_index(level=1)
    # Find bmax
    bmax = {}

    for i in stats.index.unique():
        try:
            bmax[i] = stats.loc[i][stats.loc[i]['pAB'] < 1e-3]['b'].iloc[0] 
        except IndexError:
            print(f"Max impact parameter not reached for E = {i} K. Increase impact parameter until P(b) < 1e-3. ")
    # Create new dataframe up to bmax
    data_all = []
    for k,v in bmax.items():
        data_all.append(stats.loc[k][stats.loc[k]['b'] <= v])
    integ = pd.concat(data_all)
    # Trapezoidal integrals
    sig_AB = integ.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapz(g.pAB*g.b**4, x=g.b))*fact_sig
    sig_AB_err = integ.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapz(g.pAB_err*g.b**4, x=g.b))*fact_sig
    sig_BB = integ.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapz(g.pBB*g.b**4, x=g.b))*fact_sig
    sig_BB_err = integ.groupby('e').apply(lambda g: 8*np.pi**2/3*integrate.trapz(g.pBB_err*g.b**4, x=g.b))*fact_sig
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
    mu0 = 120631.7241 #SrCs
    opac = opacity('../example/Sr+Cs/results/short2.txt')#, mode = 'w', output = '../example/Sr+Cs/results/opacity.txt')
    sigma = cross_section('../example/Sr+Cs/results/short2.txt')#, mode = 'w', output = '../example/Sr+Cs/results/sigma.txt')
    rate = k3('../example/Sr+Cs/results/short2.txt', mu0)#, mode = 'w', output = '../example/Sr+Cs/results/k3.txt')

    # Plot opacities
    opac = opac.reset_index(level=1)
    for i in opac.index.unique()[:5]:
        df = opac.loc[i]
        plt.figure(1)
        plt.errorbar(df['b'], df['pAB'], df['pAB_err'], fmt='.',label = f'{i} K')
        plt.legend()
        plt.title(r'$P_{AB}$')
        plt.figure(2)
        plt.errorbar(df['b'], df['pBB'], df['pBB_err'], fmt='.', label = f'{i} K')
        plt.legend()
        plt.title(r'$P_{BB}$')
    # plt.show()
    
    # Plot rates
    plt.figure(3)
    plt.errorbar(rate.reset_index(0)['e'],rate.reset_index(0)['k3_AB'],rate.reset_index(0)['k3_AB_err'], fmt = '.')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'E$(E_H)$')
    plt.ylabel(r'$k_3 (cm^6/s)$')
    plt.title('Rate AB')
    plt.figure(4)
    plt.errorbar(rate.reset_index(0)['e'],rate.reset_index(0)['k3_BB'],rate.reset_index(0)['k3_BB_err'], fmt = '.')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Rate BB')
    plt.xlabel(r'E$(E_H)$')
    plt.ylabel(r'$k_3 (cm^6/s)$')
    plt.show()