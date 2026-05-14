import numpy as np
import numba
import warnings
from tbr.constants import *

# Suppress all Numba warnings
warnings.filterwarnings("ignore", category=numba.NumbaWarning)


@numba.njit
def bisection_jit(f, a, b, u, tol=1e-8, max_iter=500):
    """
    Numba-compatible Bisection Method (replacement for scipy.optimize.brentq).
    """
    fa, fb = f(a, u), f(b, u)
    if np.sign(fa) == np.sign(fb):
        return np.nan 

    for _ in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c, u)

        if np.abs(fc) < tol:
            return c

        if np.sign(fa) == np.sign(fc):
            a, fa = c, fc
        else:
            b, fb = c, fc
    
    return (a + b) / 2.0

@numba.njit
def find_barrier(v, dv, j, mu):
    """
    Numba-compatible fsolve (replacement for scipy.optimize.fsolve).
    Returns (r_max, v_max) 
    """
    # If there's no rotational momentum, the barrier is 0 at infinity
    if j <= 0:
        return np.inf, 0.0
    
    r = 0.1
    dr = 0.05
    # dv_eff(r) = dv(r) - j*(j+1)/(mu * r**3)
    prev_dv = dv(r) - j*(j+1)/(mu * r**3)
    
    found_min = False
    for _ in range(10000): # Scan up to r=500
        r += dr
        curr_dv = dv(r) - j*(j+1)/(mu * r**3)
        
        if not found_min:
            # Look for the minimum (dv_eff crosses from - to +)
            if prev_dv < 0 and curr_dv >= 0:
                found_min = True
        else:
            # Look for the maximum/barrier (dv_eff crosses from + to -)
            if prev_dv > 0 and curr_dv <= 0:
                # Barrier root bracketed! Refine with bisection
                a = r - dr
                b = r
                fa = dv(a) - j*(j+1)/(mu * a**3)
                for _ in range(50):
                    c = (a + b) / 2.0
                    fc = dv(c) - j*(j+1)/(mu * c**3)
                    if abs(fc) < 1e-8:
                        break
                    if np.sign(fa) == np.sign(fc):
                        a = c
                        fa = fc
                    else:
                        b = c
                        
                barrier_r = (a + b) / 2.0
                barrier_v = v(barrier_r) + j*(j+1)/(2 * mu * barrier_r**2)
                return barrier_r, barrier_v
                
        prev_dv = curr_dv
        
    return np.inf, 0.0


@numba.njit
def jac2cart_jit(x, m1, m2):
    '''
    Jacobian to Cartesian coordinate system for distances.
    :param x: [rho1x, rho1y, rho1z, rho2x, rho2y, rho2z]
    '''
    C1 = m1/(m1+m2)
    C2 = m2/(m1+m2)

    rho1x, rho1y, rho1z, rho2x, rho2y, rho2z = x
    
    # Internuclear distances 
    r12 = np.sqrt(rho1x**2+rho1y**2+rho1z**2)
    r23 = np.sqrt((rho2x - C1*rho1x)**2
                  + (rho2y - C1*rho1y)**2 
                  + (rho2z - C1*rho1z)**2)
    r31 = np.sqrt((rho2x + C2*rho1x)**2
                  + (rho2y + C2*rho1y)**2 
                  + (rho2z + C2*rho1z)**2)
    return r12, r23, r31

@numba.njit
def hamEq_jit(t, w, mu12, mu312, m1, m2, v12, v23, v31, dv12, dv23, dv31, R0_unused, r_stop_unused):
    '''
    Hamilton's equations of motion
    '''
    C1 = m1/(m1+m2)
    C2 = m2/(m1+m2)

    rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = w

    drho1x = p1x/mu12
    drho1y = p1y/mu12 
    drho1z = p1z/mu12 
    drho2x = p2x/mu312
    drho2y = p2y/mu312
    drho2z = p2z/mu312

    r12, r23, r31 = jac2cart_jit(w[:6], m1, m2)

    dP1x = - (dv12(r12)*rho1x/r12 + dv23(r23)*(-C1*rho2x + C1**2*rho1x)/r23 + 
             dv31(r31)*(C2*rho2x + C2**2*rho1x)/r31)
    dP1y = - (dv12(r12)*rho1y/r12 + dv23(r23)*(-C1*rho2y + C1**2*rho1y)/r23 + 
             dv31(r31)*(C2*rho2y + C2**2*rho1y)/r31)
    dP1z = - (dv12(r12)*rho1z/r12 + dv23(r23)*(-C1*rho2z + C1**2*rho1z)/r23 + 
             dv31(r31)*(C2*rho2z + C2**2*rho1z)/r31)
    dP2x = - (dv23(r23)*(rho2x-C1*rho1x)/r23 + 
             dv31(r31)*(rho2x +C2*rho1x)/r31)
    dP2y = - (dv23(r23)*(rho2y-C1*rho1y)/r23 + 
             dv31(r31)*(rho2y +C2*rho1y)/r31)
    dP2z = - (dv23(r23)*(rho2z-C1*rho1z)/r23 + 
             dv31(r31)*(rho2z +C2*rho1z)/r31)
    
    f = [drho1x, drho1y, drho1z, drho2x, drho2y, drho2z,
         dP1x, dP1y, dP1z, dP2x, dP2y, dP2z] 
    
    return f

@numba.njit
def stop_event_jit(t, w, mu12_unused, mu312_unused, m1, m2, v12u, v23u, v31u, dv12u, dv23u, dv31u,  R0, r_stop):
    r12, r23, r31 = jac2cart_jit(w[:6], m1, m2)
    r_lim = R0 * r_stop
    return np.min(np.array([r_lim - r12, r_lim - r23, r_lim - r31]))

stop_event_jit.terminal = True
stop_event_jit.direction = -1


@numba.njit
def checkBound_jit(final_w, m1, m2, m3, C1, C2, v12, v23, v31, dv12, dv23, dv31):
    rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = final_w

    mu12 = m1*m2/(m1+m2)
    mu23 = m2*m3/(m2+m3)
    mu31 = m3*m1/(m3+m1)
    mu312= m3*(m1+m2)/(m1+m2+m3)

    r12_x = rho1x; r12_y = rho1y; r12_z = rho1z
    r23_x = rho2x - C1*rho1x
    r23_y = rho2y - C1*rho1y
    r23_z = rho2z - C1*rho1z
    r31_x = rho2x + C2*rho1x
    r31_y = rho2y + C2*rho1y
    r31_z = rho2z + C2*rho1z

    r12 = np.sqrt(r12_x**2 + r12_y**2 + r12_z**2)
    r23 = np.sqrt(r23_x**2 + r23_y**2 + r23_z**2)
    r31 = np.sqrt(r31_x**2 + r31_y**2 + r31_z**2)

    p12_x = p1x; p12_y = p1y; p12_z = p1z
    p23_x = mu23 * p2x / mu312-mu23 * p1x / m2
    p23_y = mu23 * p2y / mu312-mu23 * p1y / m2
    p23_z = mu23 * p2z / mu312-mu23 * p1z / m2
    p31_x = mu31 * p2x / mu312+mu31 * p1x / m1
    p31_y = mu31 * p2y / mu312+mu31 * p1y / m1
    p31_z = mu31 * p2z / mu312+mu31 * p1z / m1

    p12 = np.sqrt(p12_x**2 + p12_y**2 + p12_z**2)
    p23 = np.sqrt(p23_x**2 + p23_y**2 + p23_z**2)
    p31 = np.sqrt(p31_x**2 + p31_y**2 + p31_z**2)
    
    E12 = v12(r12) + p12**2 / (2 * mu12)
    E23 = v23(r23) + p23**2 / (2 * mu23)
    E31 = v31(r31) + p31**2 / (2 * mu31)

    # # Angular momentum components
    j12_x = rho1y*p1z - rho1z*p1y
    j12_y = rho1z*p1x - rho1x*p1z
    j12_z = rho1x*p1y - rho1y*p1x

    j23_x = r23_y*p23_z - r23_z*p23_y
    j23_y = r23_z*p23_x - r23_x*p23_z
    j23_z = r23_x*p23_y - r23_y*p23_x

    j31_x = r31_y*p31_z - r31_z*p31_y
    j31_y = r31_z*p31_x - r31_x*p31_z
    j31_z = r31_x*p31_y - r31_y*p31_x

    j12 = np.round(-0.5 + 0.5*np.sqrt(1 + 4*(j12_x**2 + j12_y**2 + j12_z**2)))
    j23 = np.round(-0.5 + 0.5*np.sqrt(1 + 4*(j23_x**2 + j23_y**2 + j23_z**2)))
    j31 = np.round(-0.5 + 0.5*np.sqrt(1 + 4*(j31_x**2 + j31_y**2 + j31_z**2)))
    
    bdx12, bdry12 = find_barrier(v12, dv12, j12, mu12)
    bdx23, bdry23 = find_barrier(v23, dv23, j23, mu23)
    bdx31, bdry31 = find_barrier(v31, dv31, j31, mu31)

    is_bound12 = (E12 < bdry12) and (r12 < bdx12)
    is_bound23 = (E23 < bdry23) and (r23 < bdx23)
    is_bound31 = (E31 < bdry31) and (r31 < bdx31)

    # V_barrier12 = 0.0
    # V_barrier23 = 0.0
    # V_barrier31 = 0.0
    
    # is_bound12 = E12 < V_barrier12
    # is_bound23 = E23 < V_barrier23
    # is_bound31 = E31 < V_barrier31

    N_true = is_bound12 + is_bound23 + is_bound31

    n_12, n_23, n_31, n_d, n_c = 0, 0, 0, 0, 0

    if N_true >= 2:
        n_c += 1 # Complex
        return np.array([n_12, n_23, n_31, n_d, n_c])
    elif N_true == 0:
        n_d += 1
        return np.array([n_12, n_23, n_31, n_d, n_c])
    else: # N_true == 1
        if is_bound12 and not is_bound23 and not is_bound31:
            n_12 +=1
            return np.array([n_12, n_23, n_31, n_d, n_c])
        elif is_bound23 and not is_bound31 and not is_bound12:
            n_23 +=1
            return np.array([n_12, n_23, n_31, n_d, n_c])
        elif is_bound31 and not is_bound12 and not is_bound23:
            n_31 +=1
            return np.array([n_12, n_23, n_31, n_d, n_c])
        else:
            return np.array([n_12, n_23, n_31, n_d, n_c])
    
@numba.njit
def root_function_aP5(x, uP5):
    return 12*x - 8*np.sin(2*x) + np.sin(4*x) - 12*np.pi*uP5

@numba.njit
def root_function_ab3(x, ub3):
    return 2 * x - np.sin(2 * x) - 2 * np.pi * ub3

@numba.njit
def iCond_jit(E0, b0, R0, dR0, m1,m2,m3, v12, v23, v31, seeded_rng):
    '''
    Generate initial conditions in atomic units.
    Returns:
        w0, R, status, V_rej, R_rej, angles
        
    '''
    mu0 = np.sqrt(m1*m2*m3/(m1+m2+m3))
    C1 = m1/(m1+m2)
    C2 = m2/(m1+m2)

    P0 = np.sqrt(2*E0*mu0)

    R = R0 + dR0*(2.0 * seeded_rng.random() - 1)

    aP1 = 2*np.pi*seeded_rng.random() # 2d
    uP2 = 1-2*seeded_rng.random() # 3d
    aP2 = np.arccos(uP2)
    uP5 = seeded_rng.random()
    aP5 = bisection_jit(root_function_aP5, 0, np.pi, uP5)
    
    P6D_0 = np.array([P0*np.sin(aP1)*np.sin(aP2)*np.sin(aP5),
                      P0*np.cos(aP1)*np.sin(aP2)*np.sin(aP5),
                      P0*np.cos(aP2)*np.sin(aP5),
                      0,
                      0,
                      P0*np.cos(aP5)])

    # Impact parameter angles
    ab1 = 2*np.pi*seeded_rng.random() 
    ub2 = 1-2*seeded_rng.random()
    ab2 = np.arccos(ub2)
    ub3 = seeded_rng.random()
    ab3 = bisection_jit(root_function_ab3, 0.0, np.pi, ub3)


    ub4r = seeded_rng.random() # generate random number for ub4
    ub4 = (2*ub4r + 2*np.sqrt(ub4r**2 - ub4r + 0j) - 1)**(1/3) # Add 0j for negative sqrt
    ab4 = (np.arccos((-(1j*np.sqrt(3)+1)*ub4**2+1j*np.sqrt(3)-1)/(2*ub4))).real

    bnn = np.array([np.sin(ab1)*np.sin(ab2)*np.sin(ab3)*np.sin(ab4),
                    np.cos(ab1)*np.sin(ab2)*np.sin(ab3)*np.sin(ab4),
                    np.cos(ab2)*np.sin(ab3)*np.sin(ab4),
                    np.cos(ab3)*np.sin(ab4),
                    np.cos(ab4),
                    0])
    
    angles = (aP1,uP2, aP2, uP5, aP5, ab1, ub2, ab2, ub3, ab3, ub4, ab4)

    P_dot_P = np.dot(P6D_0, P6D_0)
    if P_dot_P < 1e-15:
        b_vec = b0 * bnn / np.linalg.norm(bnn)
    else:
        P_dot_bnn = np.dot(bnn, P6D_0)
        bn = bnn - P_dot_bnn * P6D_0 / P_dot_P

        bn_norm = np.linalg.norm(bn)
        if bn_norm < 1e-15:
            # print('Error: bn == 0, so b is parallel to P6D_0')
            return np.full(12, np.nan), R, 1, 0.0, 0.0, angles
        
        b_vec = b0 * bn / bn_norm

    R_b0_sq = R**2 - b0**2
    if R_b0_sq < 0:
        # print('Error: R**2 < b0**2')
        return np.full(12, np.nan), R, 1, 0.0, 0.0, angles
    
    scale_factor = np.sqrt(R_b0_sq)/P0
    rho6D_0 = b_vec - P6D_0*scale_factor

    rho1x,rho1y,rho1z,rho2x,rho2y,rho2z = rho6D_0

    r12_0 = np.sqrt(rho1x**2 + rho1y**2 + rho1z**2)
    r23_0 = np.sqrt((rho2x - C1*rho1x)**2 
                    +(rho2y - C1*rho1y)**2 
                    +(rho2z - C1*rho1z)**2)
    r31_0 = np.sqrt((rho2x + C2*rho1x)**2 
                    +(rho2y + C2*rho1y)**2 
                    +(rho2z + C2*rho1z)**2)
    
    if abs(v12(r12_0)) > 0.01*E0:
        print(f'Rejected: v12(r12_0) > 0.01*E0 ')
        return np.full(12, np.nan), R, 2, v12(r12_0), r12_0, angles
    if abs(v23(r23_0)) > 0.01*E0:
        print('Rejected: v23(r23) > 0.01*E0')
        return np.full(12, np.nan), R, 2, v23(r23_0), r23_0, angles
    if abs(v31(r31_0)) > 0.01*E0:
        print('Rejected: v31(r31) > 0.01*E0')
        return np.full(12, np.nan), R, 2, v31(r31_0), r31_0, angles

    w0 = np.concatenate((rho6D_0, P6D_0))

    return w0, R, 0, 0.0, 0.0, angles