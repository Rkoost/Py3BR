import numpy as np
from scipy.optimize import fsolve,brentq, minimize_scalar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocess as mp
import pandas as pd
import sys
import os
import time
from Py3BR.constants import *
import Py3BR.util as util

class TBR(object):
    def __init__(self, bare = True,**kwargs):
        self.m1 = kwargs.get('m1')
        self.m2 = kwargs.get('m2')
        self.m3 = kwargs.get('m3')
        self.E0 = kwargs.get('E0')*K2Har
        self.b0 = kwargs.get('b0')
        self.R0 = kwargs.get('R0')
        self.dR0  = kwargs.get('dR0')
        self.v12 = kwargs.get('v12')
        self.v23 = kwargs.get('v23')
        self.v31 = kwargs.get('v31')
        self.dv12 = kwargs.get('dv12')
        self.dv23 = kwargs.get('dv23')
        self.dv31 = kwargs.get('dv31')
        self.seed = kwargs.get('seed')
        self.bare = bare
        self.t_stop = kwargs.get('integ')['t_stop']
        self.r_stop = kwargs.get('integ')['r_stop']
        self.a_tol = kwargs.get('integ')['a_tol']
        self.r_tol = kwargs.get('integ')['r_tol']
        self.mtot = self.m1 + self.m2 + self.m3
        self.mu12 = self.m1*self.m2/(self.m1+self.m2) 
        self.mu31 = self.m1*self.m3/(self.m1+self.m3) 
        self.mu23 = self.m2*self.m3/(self.m2+self.m3) 
        self.mu312 = self.m3*(self.m1+self.m2)/self.mtot
        self.mu0 = np.sqrt(self.m1*self.m2*self.m3/self.mtot)
        self.C1 = self.m1/(self.m1+self.m2)
        self.C2 = self.m2/(self.m1+self.m2)
        

    def set_attrs(self):
        self.delta_e = [np.nan]
        self.delta_l = [np.nan]
        self.wn = [np.nan]
        self.t = [np.nan]*2

    def iCond(self):
        '''
        Generate initial conditions:
            - Randomize angles
            - Generate P0 (6-d momentum vector)
            - Generate b (impact parameter)
        '''
        rng = np.random.default_rng(self.seed)
        self.P0 = np.sqrt(2*self.mu0*self.E0)

        # Momentum angles
        aP1 = 2*np.pi*rng.random() # 2d
        uP2 = 1-2*rng.random() # 3d
        aP2 = np.arccos(uP2)

        # Including all directions
        # uP3 = rng.random()
        # fP3 = lambda x: 2*x - np.sin(2*x) - 2*np.pi*uP3
        # aP3 = brentq(fP3, a=0, b=np.pi)
        # uP4 = rng.random()
        # fP4 = lambda x: np.cos(x)**3 - 3*np.cos(x) - 4*uP4 + 2
        # aP4 = brentq(fP4,a = 0, b = np.pi)
        # # uP5 = rng.random()
        # # fP5 = lambda x: 12*x-8*np.sin(2*x)+np.sin(4*x)-12*np.pi*uP5
        # # aP5 = brentq(fP5, a=0, b=np.pi)
        # # self.P6D_0 = np.array([self.P0*np.sin(aP1)*np.sin(aP2)*np.sin(aP3)*np.sin(aP4)*np.sin(aP5),
        # #                       self.P0*np.cos(aP1)*np.sin(aP2)*np.sin(aP3)*np.sin(aP4)*np.sin(aP5),
        # #                       self.P0*np.cos(aP2)*np.sin(aP3)*np.sin(aP4)*np.sin(aP5),
        # #                       self.P0*np.cos(aP3)*np.sin(aP4)*np.sin(aP5),
        # #                       self.P0*np.cos(aP4)*np.sin(aP5),
        # #                       self.P0*np.cos(aP5)])

        uP5 = rng.random()
        fP5 = lambda x: 12*x-8*np.sin(2*x)+np.sin(4*x)-12*np.pi*uP5
        aP5 = fsolve(fP5, [0, np.pi])[0]
        self.P6D_0 = np.array([self.P0*np.sin(aP1)*np.sin(aP2)*np.sin(aP5),
                  self.P0*np.cos(aP1)*np.sin(aP2)*np.sin(aP5),
                  self.P0*np.cos(aP2)*np.sin(aP5),
                  0,
                  0,
                  self.P0*np.cos(aP5)])
        
        # Impact parameter angles
        ab1 = 2*np.pi*rng.random() # 2d
        ub2 = 1-2*rng.random()
        ab2 = np.arccos(ub2)
        ub3 = rng.random()
        fb3 = lambda x: 2*x - np.sin(2*x) - 2*np.pi*ub3
        ab3 = brentq(fb3, a= 0, b=np.pi)
        ub4r = rng.random() # generate random number for ub4
        ub4 = (2*ub4r + 2*np.sqrt(ub4r**2 - ub4r + 0j) - 1)**(1/3) # Add 0j for negative sqrt
        ab4 = (np.arccos((-(1j*np.sqrt(3)+1)*ub4**2+1j*np.sqrt(3)-1)/(2*ub4))).real

        # 6d impact parameter vector
        bnn = np.array([np.sin(ab1)*np.sin(ab2)*np.sin(ab3)*np.sin(ab4),
                        np.cos(ab1)*np.sin(ab2)*np.sin(ab3)*np.sin(ab4),
                        np.cos(ab2)*np.sin(ab3)*np.sin(ab4),
                        np.cos(ab3)*np.sin(ab4),
                        np.cos(ab4),
                        0])
        
        # orthogonalize P0 and b
        bn = bnn - np.dot(bnn,self.P6D_0)*self.P6D_0/np.dot(self.P6D_0,self.P6D_0)
        b = self.b0*bn/np.linalg.norm(bn)

        # Initial position in 6D space
        self.R = self.R0 + self.dR0*(2*rng.random()-1) # R0 +/ dR0
        self.rho6D_0 =  b - self.P6D_0*np.sqrt(self.R**2 - self.b0**2)/self.P0
        rho1x,rho1y,rho1z,rho2x,rho2y,rho2z = self.rho6D_0

        # Jacobi to Cartesian
        r12_0 = np.sqrt(rho1x**2 + rho1y**2 + rho1z**2)
        r23_0 = np.sqrt((rho2x - self.C1*rho1x)**2 
                       +(rho2y - self.C1*rho1y)**2 
                       +(rho2z - self.C1*rho1z)**2)
        r31_0 = np.sqrt((rho2x + self.C2*rho1x)**2 
                       +(rho2y + self.C2*rho1y)**2 
                       +(rho2z + self.C2*rho1z)**2)

        
        # Test for rejection criteria
        self.rejected = 0
        if self.R < self.b0:
            self.rejected = self.rejected+1
        elif abs(self.v12(r12_0)) > 1e-2*self.E0:
            self.rejected = self.rejected+1
        elif abs(self.v23(r23_0)) > 1e-2*self.E0:
            self.rejected = self.rejected+1
        elif abs(self.v31(r31_0)) > 1e-2*self.E0:
            self.rejected = self.rejected+1
        # Set attributes to NaN if rejected
        if self.rejected ==1:
            self.set_attrs()

        return self.rho6D_0, self.P6D_0
        
    def hamEq(self,t,w):
        ''' 
        Writes Hamilton's equations as a vector field. 
            Usage:
                Input function for scipy.integrate.solve_ivp
        t, None
            time
        w, list
            state variables; w = [rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, 
                                p1x, p1y, p1z, p2x, p2y, p2z]
        Returns:
        f, list
            Set of 12 first order differential equations.    
        '''
        # Unpack jacobi vectors
        rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = w
        r12,r23,r31 = util.jac2cart(w[:6], self.C1, self.C2)
        # Partial derivatives
        drho1x = p1x/self.mu12
        drho1y = p1y/self.mu12        
        drho1z = p1z/self.mu12 
        drho2x = p2x/self.mu312
        drho2y = p2y/self.mu312
        drho2z = p2z/self.mu312
        dP1x = - (self.dv12(r12)*rho1x/r12 + self.dv23(r23)*(-self.C1*rho2x + self.C1**2*rho1x)/r23 + 
                self.dv31(r31)*(self.C2*rho2x + self.C2**2*rho1x)/r31)
        dP1y = - (self.dv12(r12)*rho1y/r12 + self.dv23(r23)*(-self.C1*rho2y + self.C1**2*rho1y)/r23 + 
                self.dv31(r31)*(self.C2*rho2y + self.C2**2*rho1y)/r31)
        dP1z = - (self.dv12(r12)*rho1z/r12 + self.dv23(r23)*(-self.C1*rho2z + self.C1**2*rho1z)/r23 + 
                self.dv31(r31)*(self.C2*rho2z + self.C2**2*rho1z)/r31)
        dP2x = - (self.dv23(r23)*(rho2x-self.C1*rho1x)/r23 + 
                self.dv31(r31)*(rho2x +self.C2*rho1x)/r31)
        dP2y = - (self.dv23(r23)*(rho2y-self.C1*rho1y)/r23 + 
                self.dv31(r31)*(rho2y +self.C2*rho1y)/r31)
        dP2z = - (self.dv23(r23)*(rho2z-self.C1*rho1z)/r23 + 
                self.dv31(r31)*(rho2z +self.C2*rho1z)/r31)
        f = [drho1x, drho1y, drho1z, drho2x, drho2y, drho2z,
            dP1x, dP1y, dP1z, dP2x, dP2y, dP2z] # Hamilton's equations
        return f
    
    def runT(self, plot=False):
        np.set_printoptions(suppress=True) # remove sci notation
        self.count = [0,0,0,0,0] # Initiate count vector
        self.iCond() # Set intial position/momentum vectors in Jacobi coords.

        # Mass ratios 
        u1 = np.sqrt(self.mu12/self.mu0)
        u2 = np.sqrt(self.mu312/self.mu0)

        if self.bare == True:
            w0 = np.concatenate((self.rho6D_0,
                                self.P6D_0[:3]*u1, self.P6D_0[3:]*u2)) # Bare representation of rho6D, P6D
        else:
            w0 = np.concatenate((self.rho6D_0[:3]/u1, self.rho6D_0[3:]/u2,
                                 self.P6D_0[:3]*u1,self.P6D_0[3:]*u2)) # MW representation of Rho6D, P6D
        
        if self.rejected == 1: # Stop calculation if rejected
            return
        
        # Stop conditions
        def stop1(t,w):
            rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = w
            r12 = np.sqrt(rho1x**2 + rho1y**2 + rho1z**2)
            return r12 - self.R0*self.r_stop
        def stop2(t,w):
            rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = w
            r23 = np.sqrt((rho2x - self.C1*rho1x)**2
                        + (rho2y - self.C1*rho1y)**2 
                        + (rho2z - self.C1*rho1z)**2)
            return r23 - self.R0*self.r_stop
        def stop3(t,w):
            rho1x, rho1y, rho1z, rho2x, rho2y, rho2z, p1x, p1y, p1z, p2x, p2y, p2z = w
            r31 = np.sqrt((rho2x + self.C2*rho1x)**2
                        + (rho2y + self.C2*rho1y)**2 
                        + (rho2z + self.C2*rho1z)**2)
            return r31 - self.R0*self.r_stop
        
        stop1.terminal = True
        stop2.terminal = True
        stop3.terminal = True

        if self.bare == True:
            tscale = self.R/np.sqrt(2*self.E0/self.mu312) # Time until collision
        else:
            tscale = self.R*self.mu0/self.P0
        wsol = solve_ivp(y0 = w0, fun = lambda t, y: self.hamEq(t,y),    
                t_span = [0,tscale*self.t_stop], method = 'RK45', 
                rtol = self.r_tol, atol = self.a_tol, events = (stop1,stop2,stop3))    
        
        # t_eval = np.linspace(0, tscale*self.t_stop, 1000)
        # wsol = solve_ivp(y0 = w0, fun = lambda t, y: self.hamEq(t,y),    
        #         t_span = [0,tscale*self.t_stop],  t_eval = t_eval,
        #         rtol = self.r_tol, atol = self.a_tol, events = (stop1,stop2,stop3))   
        
        self.wn = wsol.y
        self.t = wsol.t
        x = self.wn[:6] #rho1, rho2
        En, Vn, Kn, Ln = util.hamiltonian(self)

        self.delta_e = np.abs(En[-1] - En[0]) # Energy difference
        self.delta_l = np.abs(Ln[-1] - Ln[0]) # Momentum difference
        if self.delta_e > 1e-5:
            print(f'Energy not conserved less than 1e-5: {self.delta_e}.')
            self.rejected=1
            return
        
        # print(f'Time elapsed: {time.time()-t0}')
        # print(f'Energy difference: {self.delta_e}')
        r12,r23,r31 = util.jac2cart(x,self.C1,self.C2) # Jacobi positions

        rho1x, rho1y, rho1z, rho2x, \
        rho2y, rho2z, p1x, p1y, p1z, \
        p2x, p2y, p2z = wsol.y

        # Distance components
        r23_x = rho2x - self.C1*rho1x
        r23_y = rho2y - self.C1*rho1y
        r23_z = rho2z - self.C1*rho1z
        r31_x = rho2x + self.C2*rho1x
        r31_y = rho2y + self.C2*rho1y
        r31_z = rho2z + self.C2*rho1z

        # Momenta components
        p23_x = self.mu23*p2x/self.mu312-self.mu23*p1x/self.m2
        p23_y = self.mu23*p2y/self.mu312-self.mu23*p1y/self.m2
        p23_z = self.mu23*p2z/self.mu312-self.mu23*p1z/self.m2
        p31_x = self.mu31*p2x/self.mu312+self.mu31*p1x/self.m1
        p31_y = self.mu31*p2y/self.mu312+self.mu31*p1y/self.m1
        p31_z = self.mu31*p2z/self.mu312+self.mu31*p1z/self.m1

        p12 = np.sqrt(p1x**2+p1y**2+p1z**2)
        p23 = np.sqrt(p23_x**2 + p23_y**2 + p23_z**2)
        p31 = np.sqrt(p31_x**2 + p31_y**2 + p31_z**2)

        # Angular momentum components
        j12_x = rho1y*p1z - rho1z*p1y
        j12_y = rho1z*p1x - rho1x*p1z
        j12_z = rho1x*p1y - rho1y*p1x

        j23_x = r23_y*p23_z - r23_z*p23_y
        j23_y = r23_z*p23_x - r23_x*p23_z
        j23_z = r23_x*p23_y - r23_y*p23_x

        j31_x = r31_y*p31_z - r31_z*p31_y
        j31_y = r31_z*p31_x - r31_x*p31_z
        j31_z = r31_x*p31_y - r31_y*p31_x

        # j_eff arrays
        self.j12 = np.round(-0.5 + 0.5*np.sqrt(1 + 4*(j12_x**2 + j12_y**2 + j12_z**2)))[-1]
        self.j23 = np.round(-0.5 + 0.5*np.sqrt(1 + 4*(j23_x**2 + j23_y**2 + j23_z**2)))[-1]
        self.j31 = np.round(-0.5 + 0.5*np.sqrt(1 + 4*(j31_x**2 + j31_y**2 + j31_z**2)))[-1]

        # Veff (rotational boundary added)
        v12eff = lambda x: self.v12(x) + self.j12*(self.j12+1)/(2*self.mu12*x**2)
        v23eff = lambda x: self.v23(x) + self.j23*(self.j23+1)/(2*self.mu23*x**2)
        v31eff = lambda x: self.v31(x) + self.j31*(self.j31+1)/(2*self.mu31*x**2)

        dv12eff = lambda x: self.dv12(x) - self.j12*(self.j12+1)/(self.mu12*x**3)
        dv23eff = lambda x: self.dv23(x) - self.j23*(self.j23+1)/(self.mu23*x**3)
        dv31eff = lambda x: self.dv31(x) - self.j31*(self.j31+1)/(self.mu31*x**3)


        K12 = p12**2/2/self.mu12
        K23 = p23**2/2/self.mu23
        K31 = p31**2/2/self.mu31

        # Relative energies to check if bound states exist
        self.E12 = self.v12(r12) + K12
        self.E23 = self.v23(r23) + K23
        self.E31 = self.v31(r31) + K31

        ### Plot V(r(t)) ###
        # xdata, ydata = np.asarray(r31), np.asarray(v31eff(r31))
        # fig = plt.figure()
        # line, = plt.plot(xdata, ydata, lw='0.5')
        # scatter = plt.scatter(xdata[0], ydata[0])
        
        # def animate(n):
        #     x, y = xdata[:n+1], ydata[:n+1]
        #     scatter.set_offsets([xdata[n], ydata[n]])
        
        # ani = animation.FuncAnimation(fig, animate, frames= len(r31), interval = 0.05)
        # plt.show()

        def checkBound(self, f_state):
            '''
             Check bound states:
               1. Is relative energy below binding energy? 
                    Either 0 or set by rotational barrier
               2. Are the final 50 iterations of the energy constant?
               3. Is relative distance below binding radius?
               4. Are all other molecules unbound?
            '''
            global bdry12, bdry23, bdry31

            E12, E23, E31, r12, r23, r31 = f_state

            # Find roots
            # First, find min of functions
            min12 = fsolve(dv12eff, 0.1)
            min23 = fsolve(dv23eff, 0.1)
            min31 = fsolve(dv31eff, 0.1)

            # Boundary is right of minimum
            if self.j12 > 0:
                bdx12 = fsolve(dv12eff, min12*1.2) # close right of min
                bdry12 = v12eff(bdx12)
            else:
                bdx12 = np.inf
                bdry12 = 0
            if self.j23 > 0:
                bdx23 = fsolve(dv23eff, min23*1.2) # close right of min
                bdry23 = v23eff(bdx23)
            else: 
                bdx23 = np.inf
                bdry23 = 0
            if self.j31 > 0:
                bdx31 = fsolve(dv31eff, min31*1.2) # close right of min
                bdry31 = v31eff(bdx31)
            else:
                bdx31 = np.inf
                bdry31 = 0
            
            bound12 = (E12 < bdry12) and (r12 < bdx12)
            bound23 = (E23 < bdry23) and (r23 < bdx23)
            bound31 = (E31 < bdry31) and (r31 < bdx31)
            
            if bound12:
                if not bound23 and not bound31:
                    self.count[0] += 1 # n12
                else:
                    self.count[4] += 1 # nc
            elif bound23:
                if not bound31 and not bound12:
                    self.count[1] += 1 # n23
                else:
                    self.count[4] += 1 #nc
            elif bound31:
                if not bound12 and not bound23:
                    self.count[2]+=1 # n23
                else:
                    self.count[4]+=1 # nc  
            else:
                self.count[3] +=1 # nd
           
        f_state = [self.E12[-1], self.E23[-1], self.E31[-1], r12[-1], r23[-1], r31[-1]]
        checkBound(self, f_state)

        if plot == True: # Set if running one trajectory 
            plt.figure(1)
            plt.plot(self.t,r12, label ='r12')
            plt.plot(self.t,r23, label = 'r23')
            plt.plot(self.t,r31, label = 'r31')
            plt.legend()
            # plt.show()  

            plt.figure(2)
            plt.plot(self.t,self.E12, label ='E12')
            plt.hlines(bdry12,self.t[0], self.t[-1],color = 'blue',linestyle = 'dashed', label = 'v12')
            plt.plot(self.t,self.E23, label = 'E23')
            plt.hlines(bdry23,self.t[0], self.t[-1],color = 'orange',linestyle = 'dashed', label = 'v23')
            plt.plot(self.t,self.E31, label = 'E31')
            plt.hlines(bdry31,self.t[0], self.t[-1],color = 'green',linestyle = 'dashed', label = 'v31')
            # if self.count[0] == 1:
            # if self.count[1] == 1:
            # if self.count[2] == 1:
            plt.legend()
            plt.show()  

            # plt.figure(3)
            # plt.plot(self.t, self.v12(r12), label = 'v12') # potential energy 
            # plt.plot(self.t, self.v23(r23), label = 'v32')
            # plt.plot(self.t, self.v31(r31), label = 'v31')
            # plt.plot(self.t, K12, label = 'K12') # kinetic energy
            # plt.plot(self.t, K23, label = 'K23')
            # plt.plot(self.t, K31, label = 'K31')
            # plt.legend()

            plt.show()
        return 
    

def runOneT(bare = True,*args,**kwargs):
    '''
    Runs one trajectory. Use this method as input into for loop or 
    multiprocess pool for multiple trajectories.
    '''
    try:
        traj = TBR(bare,**kwargs)
        traj.runT()
    except Exception as e:
        raise e
    
    return util.get_results(traj,*args)

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
    full, pandas dataframe
        long output with one line per trajectory
    counts, pandas dataframe
        short output for analysis
    '''
    t0 = time.time()
    result = []
    if mode == 'parallel':
        with mp.Pool(processes=cpus) as p:
            if attrs:
                event = [p.apply_async(runOneT, args = (*attrs,),kwds=(input_dict)) for i in range(n_traj)]
            else:
                event = [p.apply_async(runOneT,kwds=(input_dict)) for i in range(n_traj)]
            for res in event:
                result.append(res.get())
    if mode == 'serial':
        for i in range(n_traj):
            if attrs:
                output = runOneT(*attrs,**input_dict)
            else:
                output = runOneT(**input_dict)
            result.append(output)
    full = pd.DataFrame(result)
    cols = ['e','b','n12','n23','n31','nd','nc','rej']
    counts = full.loc[:,cols].groupby(['e','b']).sum() # sum counts
    counts['time'] = time.time() - t0
    # Long output to track attributes of each trajectory
    if long_out:
        full.to_csv(long_out, mode = 'a', index=False,
                header = os.path.isfile(long_out) == False or os.path.getsize(long_out) == 0)
    # Short output for Py3BR.analysis files
    if short_out:
        counts.to_csv(short_out, mode = 'a',
                    header = os.path.isfile(short_out) == False or os.path.getsize(short_out) == 0)
    return full, counts


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt 
    from Py3BR import plotters
    sys.path.insert(0, '../example/YbLi+/') # Add example input to path
    from inputs import *


    # R = [250,500,1000,1500,3000]
    # t = {}
    # for r in R:
    #     input_dict['R0'] = r
    #     input_dict['dR0'] = r/10
    #     traj = TBR(**input_dict)
    #     tdiff = []
    #     for i in range(10):
    #         traj.runT()
    #         rho1 = (np.sqrt(traj.wn[:3][0]**2 + traj.wn[:3][1]**2 + traj.wn[:3][2]**2))
    #         rho2 = (np.sqrt(traj.wn[3:6][0]**2 + traj.wn[3:6][1]**2 + traj.wn[3:6][2]**2))
    #         ti1 = np.where(rho1==min(rho1))
    #         ti2 = np.where(rho2==min(rho2))
    #         tdiff.append(traj.t[ti2] - traj.t[ti1])
    #     t[r] = np.mean(tdiff)
    # for k,v in t.items():
    #     plt.plot(k,v, 'o',label = f'R: {k} +/- {k/10}')
    # plt.xlabel('R0')
    # plt.ylabel('Average Time Difference')
    # plt.legend()
    # plt.show()

    #     plt.plot(traj.t, rho1, label = 'rho1')
    #     plt.plot(traj.t, rho2, label = 'rho2')
    # plt.title(f'{traj.R0} +/- {traj.dR0}')
    # plt.legend()
    # plt.show()

    input_dict['E0'] = 1e-3
    input_dict['R0'] = 20000
    input_dict['seed'] = 107
    traj = TBR(bare = True, **input_dict)
    traj.runT()
    # Search for recombination event
    # n = 101
    # while (traj.count[:3] == [0,0,0]) and n < 200:
    #     n+=1
    #     input_dict['seed'] = n
    #     traj = TBR(bare = True, **input_dict)
    #     traj.runT(plot = False)
    # print(n)

    print(util.get_results(traj))
    plt.figure(1)
    plotters.traj_plt(traj)
    plt.legend(['Li-Li','Li-Yb$^+$', 'Li-Yb$^+$'])
    plt.figure(2)
    ax = plt.gca()
    plotters.traj_3d(traj, ax = ax)
    # plt.legend(['I','I','Ar'])
    plt.show()
    

    # r12,r23,r31 = util.jac2cart(traj.wn[:6], traj.C1, traj.C2)
    # t_12 = traj.t[np.where(r12 == min(r12))]*ttos
    # t_23 = traj.t[np.where(r23 == min(r23))]*ttos
    # t_31 = traj.t[np.where(r31 == min(r31))]*ttos


    # plt.figure(1)
    # plotters.traj_plt(traj)
    # plt.vlines(t_12,0,min(r23))
    # plt.vlines(t_23,0,min(r23))
    # plt.vlines(t_31,0,min(r31))
    # # plt.show()
    # # plt.figure(2)
    # # plotters.traj_3d(traj)
    # plt.show()

    
    # i=0
    # while(traj.count[1] == 0 and traj.count[2] ==0) and i<100:
    #     traj = TBR(**input_dict)
    #     traj.runT()
    #     i+=1
    # if (traj.count[1]==1 or traj.count[2]==1):
    #     plotters.traj_plt(traj)
    #     plt.show()
