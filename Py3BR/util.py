import numpy as np

def jac2cart(x,C1,C2):
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

def get_results(traj):
    '''
    Gather results from the trajectory. 
    If other attributes are desired, 
    '''
    results = {'e': traj.E0/K2Har,
               'b': traj.b0,
               'r12': traj.count[0],
               'r23': traj.count[1],
               'r31': traj.count[2],
               'nd': traj.count[3],
               'nc': traj.count[4],
               'rej': traj.rejected,
               'tf': traj.t[-1]*ttos}
    return results