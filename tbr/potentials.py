import numpy as np
import numba
from tbr.constants import BOH2M, K2HAR

def LJ(n=6, m=12, cn = 1, cm = 1):
    @numba.njit
    def lenjones(x):
        return cm/(x**m) - cn/(x**n)
    return lenjones

def dLJ(n=6, m=12, cn=1, cm=1):
    @numba.njit
    def dlenjones(x):
        return - m*cm/(x**(m+1)) + n*cn/(x**(n+1))
    return dlenjones


def He2_V(rm = 0.29683e-9/BOH2M ,eps = 10.956 * K2HAR ,A = 1.86924404e5,
          alpha = 10.5717543,beta = -2.007758779,c6 = 1.35186623,
          c8 = 0.41495143,c10 = 0.17151143,D = 1.438):
        '''
        HFD-B3-FCI1 potential from 10.1088/0026-1394/27/4/005, 
        parameters from 10.1103/PhysRevLett.74.1586.
        '''
        _rm = float(rm)
        _eps = float(eps)
        _A = float(A)
        _alpha = float(alpha)
        _beta = float(beta)
        _c6 = float(c6)
        _c8 = float(c8)
        _c10 = float(c10)
        _D = float(D)

        @numba.njit
        def he2(x):
            R = x/_rm

            # if R < _D:
            #     F_x =  np.exp(-(_D/R-1)**2)
            # else:
            #     F_x = 1.0
            F_x = np.where(R < _D, np.exp(-(_D/R-1)**2), 1.0)
            V = _eps*(_A*np.exp(-_alpha*R + _beta*R**2) - 
                                          (_c6*R**(-6) + _c8*R**(-8) + _c10*R**(-10))*F_x)
            return V
        return he2
    
def He2_dV(rm = 0.29683e-9/BOH2M ,eps = 10.956 * K2HAR ,A = 1.86924404e5,
          alpha = 10.5717543,beta = -2.007758779,c6 = 1.35186623,
          c8 = 0.41495143,c10 = 0.17151143,D = 1.438):
    '''
    HFD-B3-FCI1 potential from 10.1088/0026-1394/27/4/005, 
    parameters from 10.1103/PhysRevLett.74.1586.
    '''
    _rm = float(rm)
    _eps = float(eps)
    _A = float(A)
    _alpha = float(alpha)
    _beta = float(beta)
    _c6 = float(c6)
    _c8 = float(c8)
    _c10 = float(c10)
    _D = float(D)

    @numba.njit
    def dhe2(x):
        R = x/_rm

        if R < _D:
            F_x =  np.exp(-(_D/R-1)**2)
            dF_dx = 2*(_D/R**2)*(_D/R-1)*F_x
        else:
            F_x = 1.0
            dF_dx = 0.0

        exp_arg = -_alpha*R + _beta*R**2
        repulsive_term = _A*np.exp(exp_arg)
        d_repulsive_dR = repulsive_term*(-_alpha + 2*_beta*R)

        C_disp = _c6*R**(-6) + _c8*R**(-8) + _c10*R**(-10)
        d_C_disp_dR = -6*_c6*R**(-7) - 8*_c8*R**(-9) - 10*_c10*R**(-11)

        d_attractive_dR = (d_C_disp_dR*F_x) + (C_disp*dF_dx)
        
        dV_dR = _eps * (d_repulsive_dR - d_attractive_dR)

        dV_dx = dV_dR/_rm
        return dV_dx
    return dhe2

# He2+ parameters
def He2plus_V(De = 0.09027661,re = 2.04725623,beta = 1.09314253,
             b = 1.6059377,c = 0.00957858,As = 0.77802774,
             Bs = 6.15941548,Al = 2.03590553,Bl = 6.65881373,
             alpha = 1.3793,q = 1):
    @numba.njit
    def he2(r):
        # Prevent overflow by imposing
        r_max = 20.0
        if r < r_max:
            r_re_internal = r - re
            exp_negbeta = np.exp(-beta*r_re_internal)
            exp_neg2beta = np.exp(-2*beta*r_re_internal)

            poly_term = c*beta**3*r_re_internal**3 *(1 + b*beta*r_re_internal)


            bracket_term = ((1-exp_negbeta)**2 + poly_term*exp_neg2beta)

            V_morse_mod = De*(bracket_term) - De
            
            # Switching functions
            switch_s = 0.5*(1-np.tanh(As*(r-Bs)))
            switch_l = 0.5*(1+np.tanh(Al*(r-Bl)))
            v_long = -q**2*alpha/(2*r**4)

            return switch_s*V_morse_mod + switch_l*v_long
        
        else:
            switch_l = 0.5 * (1 + np.tanh(Al*(r-Bl)))
            v_long = -q**2*alpha/(2*r**4)
            return switch_l*v_long
    return he2

#NEW
def He2plus_dV(De = 0.09027661,re = 2.04725623,beta = 1.09314253,
             b = 1.6059377,c = 0.00957858,As = 0.77802774,
             Bs = 6.15941548,Al = 2.03590553,Bl = 6.65881373,
             alpha = 1.3793,q = 1):
    @numba.njit
    def dhe2(r):
        r_max = 20.0

        if r < r_max:
            x = r - re
            exp_negbeta = np.exp(-beta*x)
            exp_neg2beta = np.exp(-2*beta*x)

            d_exp_negbeta = -beta*exp_negbeta
            d_exp_neg2beta = -2*beta*exp_neg2beta

            poly_term = c*beta**3*x**3 *(1 + b*beta*x)
            d_poly_term = 3*c*beta**3*x**2 + 4*b*c*beta**4*x**3

            bracket = (1 - exp_negbeta)**2 + poly_term*exp_neg2beta
            d_bracket = -2*(1-exp_negbeta)*d_exp_negbeta + d_poly_term*exp_neg2beta + poly_term*d_exp_neg2beta
            
            V_morse_mod = De * (bracket) - De
            dV_morse_mod = De * (d_bracket)
            
            tanh_s = np.tanh(As * (r - Bs))
            switch_s = 0.5 * (1 - tanh_s)
            d_switch_s = 0.5 * (-As) * (1 - tanh_s**2)
            
            tanh_l = np.tanh(Al * (r - Bl))
            switch_l = 0.5 * (1 + tanh_l)
            d_switch_l = 0.5 * (Al) * (1 - tanh_l**2)
            
            V_long = -q**2 * alpha / (2 * r**4)
            dV_long = 2 * q**2 * alpha / r**5
            
            term1 = d_switch_s * V_morse_mod + switch_s * dV_morse_mod
            term2 = d_switch_l * V_long + switch_l * dV_long
            
            return term1 + term2
        
        else:

            tanh_l = np.tanh(Al * (r - Bl))
            switch_l = 0.5 * (1 + tanh_l)
            d_switch_l = 0.5 * (Al) * (1 - tanh_l**2)
            
            V_long = -q**2 * alpha / (2 * r**4)
            dV_long = 2 * q**2 * alpha / r**5
            
            return d_switch_l * V_long + switch_l * dV_long
        
    return dhe2


# # He2 parameters
# rm = 0.29683e-9/BOH2M # position of min of V(r) : equilibrium length \approx 2.96 Angstrom
# eps = 10.956 * K2HAR # epsilon /k_B: reduction factor of potential \approx well depth = 3.32e-5 hartree
# A = 1.86924404e5
# alpha = 10.5717543
# beta = -2.007758779
# c6 = 1.35186623
# c8 = 0.41495143
# c10 = 0.17151143
# D = 1.438

# # He2+ params
# De = 0.09027661
# re = 2.04725623
# beta = 1.09314253
# b = 1.6059377
# c = 0.00957858
# As = 0.77802774
# Bs = 6.15941548
# Al = 2.03590553
# Bl = 6.65881373
# alpha = 1.3793
# q = 1