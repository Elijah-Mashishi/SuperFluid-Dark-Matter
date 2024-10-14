import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.misc import derivative
from sympy import *

G = 6.67e-11
Mplanck = 2.18e-8

class Galaxy:
    def __init__(self, data):
        self.data = data

        self.rad = data['Rad'].values #Cylindrical radius kpc
        self.sdbul = data['SDbul'].values #Surface density L_sol / pc^2
        self.sdgas = data["SDgas"].values #Surface density L_sol / pc^2
        self.sddisk = data["SDdisk"].values #Surface density L_sol / pc^
        self.vobs = data["Vobs"].values #Observed velocity km/s

        # Interpolate between the data points to get constants needed
        self.popt, self.pcov = curve_fit(Galaxy.surface_density_buldge, self.rad, self.sdbul)
        self.qopt, self.qcov = curve_fit(Galaxy.surface_density_gas, self.rad, self.sdgas, maxfev=5000)
        self.ropt, self.rcov = curve_fit(Galaxy.surface_density_disk, self.rad, self.sddisk) 
        self.sopt, self.scov = curve_fit(Galaxy.v, self.rad, self.vobs, bounds=(1e-2, 15))
        #problem with above line is curvefit takes "self" as the x-axis and replaces self object with rad
    
        self.v_inter = [Galaxy.v_interpolated(self, R) for R in self.rad]
    

# ---------------- Rho_bulge ---------------------# 
    @staticmethod
    def rho_bulge(R, z, rho_0, r_b):
        r = (R**2 + z**2)**0.5 
        return rho_0 * np.exp(-r/r_b)    

    @staticmethod
    def surface_density_buldge(Rs, rho_0, r_b):
        if hasattr(Rs, '__len__'):
            return [2 * quad(Galaxy.rho_bulge, 0, np.inf, args=(R, rho_0, r_b))[0] for R in Rs]
        else:
            return 2 * quad(Galaxy.rho_bulge, 0, np.inf, args=(Rs, rho_0, r_b))[0]

    def sdbul_interpolated(self, R):
        return Galaxy.surface_density_buldge(R, self.popt[0], self.popt[1])


    # Differentiate surface density wrt rbar
    @staticmethod
    def sdbul_prime(R):
        return derivative(Galaxy.sdbul_interpolated, R)

    # Define final surface density function
    @staticmethod
    def rho_buldge_final(R):
        
        def func(R):
            return Galaxy.sdbul_prime(R)/R
        return -1/np.pi * quad(func, 0, np.inf)[0]

# ---------------------- Rho_gas ----------------------- #

    @staticmethod
    def surface_density_gas(R, sigma0, c1, c2, c3, c4, R_sig):
        return sigma0*(1 + c1*R + c2*R**2 + c3*R**3 + c4*R**4) * np.exp(-R/R_sig)


    def sdgas_interpolated(self, R):
        return Galaxy.surface_density_gas(R, self.qopt[0], self.qopt[1], self.qopt[2], self.qopt[3], self.qopt[4], self.qopt[5])


    def rho_gas(self, R):
        return 1.4 * Galaxy.sdgas_interpolated(self, R)/ (np.sqrt(2*np.pi)* 0.130e3) # 1.4 to account for lack of H2 data. 0.130e3 is 0.130 kpc from Hossenfelder et al


#--------------------Rho_disk (*)---------------------------#
    @staticmethod
    def rho_disk(z, R, rho_0, Rstar):
        zstar = 0.196*Rstar**0.633 #kpc following Hossenfelder
        return rho_0*np.exp(-R/Rstar)*(1-(np.tanh(z/zstar))**2)

    @staticmethod
    def surface_density_disk(Rs, rho_0, Rstar):
        if hasattr(Rs, '__len__'):
            return [2 * quad(Galaxy.rho_disk, 0, np.inf, args=(R, rho_0, Rstar))[0] for R in Rs]
        else:
            return 2 * quad(Galaxy.rho_disk, 0, np.inf, args=(Rs, rho_0, Rstar))[0]

    def sddisk_interpolated(self, R):
        return Galaxy.surface_density_disk(R, self.ropt[0], self.ropt[1])

    def rho_disk_final(self, R):
        return Galaxy.sddisk_interpolated(self, R)/(np.sqrt(2*np.pi)*0.196*self.ropt[1]**0.633)

#-----------------------Final expressions------------------------------#

    def rho(self, R, Q):
        return Galaxy.rho_gas(self,R) + 0.5*Q*Galaxy.rho_disk_final(self,R) + 0.7*Q*Galaxy.rho_buldge_final(R)

    def mass(self, R, Q):
        return 4/3 * np.pi * R**3 * Galaxy.rho(self, R, Q)

    def v(self, Rs, Q):
        alpha = 5.7
        lamb = 0.05
        if hasattr(Rs, '__len__'):
            return [np.sqrt(R*(G*Galaxy.mass(self, R, Q) + np.sqrt((alpha**3 * lamb**2/ Mplanck)* G * Galaxy.mass(self, R, Q)))) for R in Rs]
        else:
            return np.sqrt(Rs*(G*Galaxy.mass(self, Rs, Q) + np.sqrt((alpha**3 * lamb**2/ Mplanck)* G * Galaxy.mass(self, Rs, Q))))

    def v_interpolated(self, R):
        return Galaxy.v(self, R, self.sopt[0])