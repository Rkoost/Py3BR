import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.integrate import trapezoid
from tbr.constants import BOH2M, K2HAR, T2S

    
def opacity(input_data, reactants, output_csv=None):
    '''
    Probability of all processes for reaction A + A + B, 
    where n12 represents A2 molecule formation.
    Input:
        input_data, csv with header ('e', 'b', 'n12', 'n23', 'n31', 'nd', 'nc')
    Output:
        Pandas DataFrame of probabilities as a function of collision 
        energy and impact parameter. 'p_AB' represents the probability of 
        AB formation, and 'p_AA' represents probability of A2 formation.
    '''
    if isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        df = pd.read_csv(input_data)

    grouped = df.groupby(['e','b']).agg({
        'n12': 'sum',
        'n23': 'sum',
        'n31': 'sum',
        'nd': 'sum',
        'nc': 'sum'
    }).reset_index()

    grouped['nT'] = grouped['n12'] + grouped['n23'] + grouped['n31'] + grouped['nd']
    nT = grouped['nT'].replace(0, np.nan)

    def get_channels(reactants, tol=1e-2):
        '''
        Identifies possible reaction channels based on reactants.

        For A + A + B: returns{'AA': ['n12'], 'AB': ['n23', 'n31']}
        For A + B + C: returns{'AB': ['n12'], 'BC': ['n23'], 'CA': ['n31']}
        For A + A + A: returns{'AA': ['n12', 'n23', 'n31']}
        '''
        letters = ['A', 'B', 'C']
        unique_ids = []
        labels = []
        
        for item in reactants:
            match_index = -1
            for i, uid in enumerate(unique_ids):
                # Use float tolerance for numbers, exact match for strings/labels
                if isinstance(item, (int, float)) and isinstance(uid, (int, float)):
                    if np.isclose(item, uid, atol=tol):
                        match_index = i
                        break
                else:
                    if item == uid:
                        match_index = i
                        break
                        
            if match_index != -1:
                labels.append(letters[match_index])
            else:
                unique_ids.append(item)
                labels.append(letters[len(unique_ids)-1])
                
        l1, l2, l3 = labels
        
        # Sort alphabetically to combine identical channels (e.g., 'BA' -> 'AB')
        c12 = "".join(sorted([l1, l2]))
        c23 = "".join(sorted([l2, l3]))
        c31 = "".join(sorted([l1, l3]))
        
        mapping = {}
        for col, channel in zip(['n12', 'n23', 'n31'], [c12, c23, c31]):
            mapping.setdefault(channel, []).append(col)
            
        return mapping

    channels = get_channels(reactants)
    
    for ch, cols in channels.items():
        grouped[f'n_{ch}'] = grouped[cols].sum(axis=1)
        grouped[f'p_{ch}'] = grouped[f'n_{ch}'] / nT
        grouped[f'p_{ch}_err'] = (np.sqrt(grouped[f'n_{ch}'])/nT) * np.sqrt((nT-grouped[f'n_{ch}'])/nT)

    df_results = grouped.fillna(0)

    if output_csv:
        df_results.to_csv(output_csv, index=False)
        print(f'Results saved to {output_csv}')
    
    return df_results

def cross_section(input_data, reactants, bmax_dict = None, windows=None, tolerances=None, output_csv=None):
    '''
    Calculate cross section sigma(E) of a three-body recombination event.
    Units: cm^5
    Input Parameters:
    input_data, str
        Path to input file. Should be in the form of short output from Py3BR, 
        with header (e,b,n12,n23,n31,nd,nc,rej,time)
    bmax_dict, dict
        Dictionary mapping bmax of each process for each energy ({'AA': {Ec:bmax}, 'AB': {Ec:bmax}})
    windows, dict
        Maps process names to window sizes ({'AA':3, 'AB':2})
    toleracnes, dict
        Maps process names to tolerances ({'AA': 10, 'AB': 25}) (see opac_bmax())
    output, str (optional)
        Path to output file, if saving. 
    '''
    fact_sig = (BOH2M*1e2)**5 # Sigma factor

    bmax_dict = bmax_dict or {}
    windows = windows or {}
    tolerances = tolerances or {}

    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        df = opacity(input_data, reactants)

    channels = [c.split('_')[1] for c in df.columns if c.startswith('p_') and not c.endswith('_err')]

    def get_limit(limit_arg, energy):
        if isinstance(limit_arg, dict):
            return limit_arg.get(energy)
        elif isinstance(limit_arg, (float, int)):
            return limit_arg
        return None
    
    def find_bmax(p_vals, b_vals, window_size=3, frac=0):
        '''
        Function to find bmax.
        P(bmax) = max(opac)*F
        where F is a fraction of the maximum value set by the user. 
        We find the best agreement with the data comes from frac=10_successful_trajectories(b=bmax)/total_#_trajectories(b=0)
        '''
        opac = np.array(p_vals)
        b_values = np.array(b_vals)

        if len(opac) == 0:
            return 0.0

        ref_val = np.max(opac) if len(opac) > 0 else 1.0
        cutoff = ref_val*frac

        if len(opac) < window_size:
            return b_values[-1] # Data too short, return max b

        windows = np.lib.stride_tricks.sliding_window_view(opac, window_size)

        if frac > 0:
            mask = np.all(windows < cutoff, axis=1)
        else:
            mask = np.all(windows == 0, axis=1)

        indices = np.where(mask)[0]

        if indices.size > 0:
            first_index = indices[0]
            bmax = b_values[first_index]
            return bmax
        else:
            # Integrate everything
            return b_values[-1]
        
    def integrate_process(group, manual_limit, suffix, threshold, window_size):
        group = group.sort_values('b')
        b = group['b'].values
        p = group[f'p_{suffix}'].values
        p_err = group[f'p_{suffix}_err'].values

        if manual_limit is not None:
            limit = manual_limit
        else:
            limit = find_bmax(p,b, window_size, threshold)

        mask = b <= limit
        b_sub = b[mask].astype(np.float64)
        p_sub = p[mask].astype(np.float64)
        p_err_sub = p_err[mask].astype(np.float64)
        
        # Need at least 2 points to integrate
        if len(b_sub) < 2:
            return 0.0, 0.0, limit
            
        # Trapezoidal weights 
        diffs = np.diff(b_sub)
        weights = np.zeros_like(b_sub)
        weights[1:-1] = 0.5 * (b_sub[2:] - b_sub[:-2])
        weights[0] = 0.5 * diffs[0]
        weights[-1] = 0.5 * diffs[-1]
        
        # Integrate
        factor = (8 * np.pi**2) / 3
        
        # Sigma = factor * sum(weights * p * b^4)
        integrand = p_sub * (b_sub**4)
        sigma = factor * np.sum(weights * integrand)*fact_sig
        
        weighted_errors = weights * p_err_sub * (b_sub**4)
        sigma_err = factor * np.sqrt(np.sum(weighted_errors**2))*fact_sig
        
        return sigma, sigma_err, limit
    
    def apply_integration(group):
        energy = group.name

        count_cols = ['n12', 'n23', 'n31', 'nd', 'nc']

        if all(col in group.columns for col in count_cols):
            first_row = group.iloc[0]
            N = sum(first_row[col] for col in count_cols)
        elif all(col in group.columns for col in ['Ntot']):
            last_row = group.iloc[-1]
            N = sum(last_row[col] for col in ['Ntot'])
        else:
            N = 1.0
            
        res = {}
        for ch in channels:
            # Default to windows=3, tolerance=10 if undefined
            win = windows.get(ch, 3)
            tol = tolerances.get(ch, 10)

            threshold = tol/N if N > 0 else 0.0
            manual_limit = get_limit(bmax_dict.get(ch), energy)

            s_val, s_err, used_b = integrate_process(group, manual_limit, ch, threshold, win)

            res[f'sig_{ch}'] = s_val
            res[f'sig_{ch}_err'] = s_err
            res[f'bmax_{ch}'] = used_b
            res[f'threshold_{ch}'] = threshold

        return pd.Series(res)
    
    df_sig = df.groupby('e').apply(apply_integration).reset_index()

    if output_csv:
        df_sig.to_csv(output_csv, index=False)
        print(f'Cross section results saved to {output_csv}')
        
    return df_sig


def rate(input_data, mu0, reactants, bmax_dict = None, 
         windows=None, tolerances=None, output_csv=None):
    
    fact_k3 = (BOH2M*1e2)/T2S # k3 factor
    sigma = cross_section(input_data=input_data,
                          reactants=reactants,
                          bmax_dict=bmax_dict,
                          windows=windows,
                          tolerances=tolerances
    )

    df = sigma.copy()
    v = np.sqrt(2*(df['e']*K2HAR).astype(np.float64)/mu0)

    channels = [c.split('_')[1] for c in df.columns if c.startswith('sig_') and not c.endswith('_err')]

    for ch in channels:
        sig_col = f'sig_{ch}'
        sig_err_col = f'sig_{ch}_err'
        k_col = f'k_{ch}'
        k_err_col = f'k_{ch}_err'

        df[k_col] = df[sig_col]*v*fact_k3
        df[k_err_col] = df[sig_err_col]*v*fact_k3

        
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f'Rates saved to {output_csv}')

    return df

def thermal_rate(rate_df, T_vals_K, process='AA'):
    '''
    Calculates temperature-dependent rate coefficient of a given process.
    Inputs:
        rate_df (DataFrame): Energy-dependent rates from rate() 
        T_vals_K (list or array): Range of temperature values to calculate 
        process (string): Either 'AA' or 'AB'.
    '''

    rate_df = rate_df.sort_values('e', ascending=True)
    e_vals = rate_df['e'].values
    k_col = f'k_{process}'
    k_err_col = f'k_{process}_err'
    if k_col not in rate_df.columns:
        raise ValueError(f'Column "{k_col}" not found in DataFrame.')
    
    k_vals = rate_df[k_col].values
    k_err_vals = rate_df[k_err_col].values
    thermal_rates = []
    thermal_rates_err = []

    for T_K in T_vals_K:

        if T_K == 0:
            thermal_rates.append(0.0)
            thermal_rates_err.append(0.0)
            continue

        exp_term = np.exp(-e_vals/T_K)

        num = e_vals**2 * k_vals * exp_term
        den = 2*T_K**3

        num_err = e_vals**2 * k_err_vals * exp_term

        integrand = num/den
        integrand_err = num_err/den
        
        k_thermal = trapezoid(y=integrand, x=e_vals)
        k_thermal_err = trapezoid(y=integrand_err, x=e_vals)

        thermal_rates.append({'T_K': T_K.round(10), 
                              f'k_thermal_{process}': k_thermal,
                              f'k_thermal_{process}_err': k_thermal_err})
        
    return pd.DataFrame(thermal_rates)

def rate_thres_ionatom(rate, mu0, atom_polarizability, suffix='AB'):
    '''
    Threshold law for energy dependent atom-ion recombination rate
    based on Langevin capture model.
    Assumes the reaction A + A + B^+, where B is the ion. 
    bmax = (2a/E)^(1/4)
    sigma = 8pi^2/15*(bmax)^5
    rate = sigma*v 
    Inputs:
        rate_df (DataFrame): Energy-dependent rates from rate() 
        mu0 (float): reduced mass in atomic units
        atom_polarizability (float): polarizability of the neutral species A
        suffix (str): 'AB' (default) or 'AA'.
    Outputs:
        Pandas DataFrame of threshold-law rates

    '''
    fact_rate = (BOH2M*1e2)**6/T2S

    df = rate.copy()
    e_val = df['e']*K2HAR
    k_th = (8*np.pi**2/15)*(2*atom_polarizability)**(5/4)*np.sqrt(2/mu0)*(e_val)**(-3/4)*fact_rate
    k_thres_col = f'k_{suffix}_thres'

    df[k_thres_col] = k_th
    return k_th

def analytical_temp_ionatom(temps, mu0, alpha):
    '''
    Threshold law for temperature dependent ion-atom recombination, based on a
    MB distribution of energies and long-range (1/r^4) interaction. 
    Useful for low temperatures. 
    '''
    fact_rate = (BOH2M*1e2)**6/T2S

    C = (8*np.pi**2/15) * ((2*alpha)**(5/4))*np.sqrt(2/mu0)*fact_rate
    A = C*gamma(9/4)/2

    T_Hartree = temps*K2HAR
    k_analytical = A*(T_Hartree**(-3/4))
    return k_analytical
