import numpy as np
from Py3BR.constants import *

def jac2cart(x,C1,C2):
    '''
    Jacobian to Cartesian coordinate system. 
    x contains all Jacobi position vectors,
    C1 = m1/(m1+m2)
    C2 = m2/(m1+m2)
    '''
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

def get_results(traj, *args):
    '''
    Gather results from the trajectory. 
    args are strings representing attributes of the trajectory,
    if you need more than the count vector.
    '''
    results = {'e': f'{traj.E0/K2Har:.2f}',
    'b': f'{traj.b0:.2f}',
    'n12': int(traj.count[0]),
    'n23': int(traj.count[1]),
    'n31': int(traj.count[2]),
    'nd':  int(traj.count[3]),
    'nc':  int(traj.count[4]),
    'rej': int(traj.rejected)}
    for arg in args:
        results[arg] = getattr(traj,arg)
    return results

def hamiltonian(traj):
    '''
    Calculate energies and momentum
    '''
    w = traj.wn

    rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, \
        p1x, p1y, p1z, p2x, p2y, p2z = w
    
    # Internuclear distances 
    r12,r23,r31 = jac2cart(w[:6],traj.C1,traj.C2)
    
    # Kinetic energy
    ekin = 0.5*(p1x**2 + p1y**2 + p1z**2)/traj.mu12 + \
            0.5*(p2x**2 + p2y**2 + p2z**2)/traj.mu312
    
    # Potential energy
    epot = np.asarray(traj.v12(r12)) + np.asarray(traj.v23(r23)) + np.asarray(traj.v31(r31))    
    # Total energy
    etot = ekin + epot    
    # Keep track of angular momenta
    lx1 = rho1y*p1z - rho1z*p1y # Internal angular momentum
    lx2 = rho2y*p2z - rho2z*p2y # Relative angular momentum
    lx = lx1 + lx2     
    ly1 = rho1z*p1x - rho1x*p1z
    ly2 = rho2z*p2x - rho2x*p2z
    ly = ly1 + ly2     
    lz1 = rho1x*p1y - rho1y*p1x
    lz2 = rho2x*p2y - rho2y*p2x
    lz = lz1 + lz2     
    ll = np.sqrt(lx**2 + ly**2 + lz**2)    
    return (etot, epot, ekin, ll)

