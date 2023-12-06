import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
import os
import time
from Py3BR.constants import *
import Py3BR.util as util

class TBR(object):
    def __init__(self,**kwargs):
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
        self.C1 = self.m1/(self.m1 + self.m2)
        self.C2 = self.m2/(self.m1 + self.m2)

    def set_t0(self):
        self.t = [0.]*2

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
        uP5 = rng.random()
        fP5 = lambda x: 12*x-8*np.sin(2*x)+np.sin(4*x)-12*np.pi*uP5
        aP5 = fsolve(fP5, [0, np.pi])[0]
        pe6D_0 = np.array([self.P0*np.sin(aP1)*np.sin(aP2)*np.sin(aP5),
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
        ab3 = fsolve(fb3, [0, np.pi])[0]
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
        self.P6D_0 = pe6D_0.conjugate()
        bn = bnn.conjugate() - np.dot(bnn,self.P6D_0)*self.P6D_0/np.dot(self.P6D_0,self.P6D_0)
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
            print(f'Rejected: Hyperradius R < impact parameter b0')
            self.set_t0()
        elif abs(self.v12(r12_0)) > 1e-2*self.E0:
            self.rejected = self.rejected+1
            print(f'Rejected: Collision energy ({self.E0}) < 100*V12 ({self.v12(r12_0)})')
            self.set_t0()
        elif abs(self.v23(r23_0)) > 1e-2*self.E0:
            self.rejected = self.rejected+1
            print(f'Rejected: Collision energy ({self.E0}) < 100*V23 ({self.v23(r23_0)})')
            self.set_t0()
        elif abs(self.v31(r31_0)) > 1e-2*self.E0:
            self.rejected = self.rejected+1
            print(f'Rejected: Collision energy ({self.E0}) < 100*V31 ({self.v31(r31_0)})')
            self.set_t0()

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
        w0 = np.concatenate((self.rho6D_0,self.P6D_0[:3]*np.sqrt(self.mu12/self.mu0),
                             self.P6D_0[3:]*np.sqrt(self.mu312/self.mu0))) # Bare representation of P0
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

        tscale = self.R/np.sqrt(2*self.E0/self.mu312) # Time until collision
        wsol = solve_ivp(y0 = w0, fun = lambda t, y: self.hamEq(t,y),    
                t_span = [0,tscale*self.t_stop], method = 'RK45', 
                rtol = self.r_tol, atol = self.a_tol, events = (stop1,stop2,stop3))
        
        self.wn = wsol.y
        self.t = wsol.t
        x = self.wn[:6] #rho1, rho2
        En, Vn, Kn, Ln = util.hamiltonian(self,self.wn)

        self.delta_e = En[-1] - En[0] # Energy difference
        self.delta_l = Ln[-1] - Ln[0] # Momentum difference
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

        K12 = p12**2/2/self.mu12
        K23 = p23**2/2/self.mu23
        K31 = p31**2/2/self.mu31

        # Relative energies to check if bound states exist
        E12 = self.v12(r12) + K12
        E23 = self.v23(r23) + K23
        E31 = self.v31(r31) + K31

        # Check bound states
        if E12[-1] < 0:
            if E23[-1] > 0 and E31[-1] > 0:
                self.count[0]+=1 # r12
            else:
                self.count[4]+=1  # complex
        elif E23[-1]<0:
            if E31[-1] > 0 and E12[-1] > 0:
                self.count[1]+=1 # r23
            else:
                self.count[4]+=1 # complex
        elif E31[-1]<0:
            if E12[-1] > 0 and E23[-1] > 0:
                self.count[2]+=1 # r31
            else:
                self.count[4]+=1 # complex
        else:
            self.count[3]+=1 # dissociation

        if plot == True: # Set if running one trajectory 
            plt.figure(1)
            plt.plot(self.t,r12, label ='r12')
            plt.plot(self.t,r23, label = 'r23')
            plt.plot(self.t,r31, label = 'r31')
            plt.legend()
            # plt.show()  

            # plt.figure(2)
            # plt.plot(self.t,E12, label ='E12')
            # plt.plot(self.t,E23, label = 'E23')
            # plt.plot(self.t,E31, label = 'E31')
            # plt.hlines(0,self.t[0], self.t[-1], color = 'black',linestyle = 'dashed')
            # plt.legend()
            # # plt.show()  

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
    

def runOneT(*args,**kwargs):
    '''
    Runs one trajectory. Use this method as input into for loop or 
    multiprocess pool for multiple trajectories.
    '''
    try:
        traj = TBR(**kwargs)
        traj.runT()
    except:
        return
    return util.get_results(traj,*args)
