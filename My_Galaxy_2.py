import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.optimize import curve_fit
from scipy.misc import derivative
from astropy import units as u
from astropy import constants as c
from sympy import *

G = c.G.value
hbar = c.hbar.value
c = c.c.value
α = 5.7
Λ = 0.05
#Q = 1
M_pl = ((hbar * c)/(8*np.pi*G))**0.5

Mplanck = 2.18e-8

class Galaxy:
    def __init__(self, data):
        self.data = data                        # Stores the Observational data in the data attribute of the class.

        self.rad = data['Rad'].values           # Cylindrical radius kpc
        self.sdbul = data['SDbul'].values#*1e6       # Surface density L_sol / pc^2
        self.sdgas = data["SDgas"].values#*1e6       # Surface density L_sol / pc^2
        self.sddisk = data["SDdisk"].values#*1e6     # Surface density L_sol / pc^2
        self.vobs = data["Vobs"].values         # Observed velocity km/s
        self.vobs_err = data["errV"].values     # Error of the observed velocity

        # Curve fitting is done below, interpolate between the data points to get the fit parameters
        
        self.popt, self.pcov = curve_fit(Galaxy.surface_density_bulge, self.rad, self.sdbul)
        print("Bulge Fit Parameters:", self.popt)
        
        self.qopt, self.qcov = curve_fit(Galaxy.surface_density_gas, self.rad, self.sdgas, maxfev=5000)
        print("Gas Fit Parameters:", self.qopt)
        
        self.ropt, self.rcov = curve_fit(Galaxy.surface_density_disk, self.rad, self.sddisk)
        print("Disk Fit Parameters:", self.ropt)
        
        self.sopt, self.scov = curve_fit(self._v_fit, self.rad, self.vobs, bounds=(1e-2, 15))
        print("Velocity Fit Parameters:", self.sopt)
         
        self.v_inter = [self.v_interpolated(r_bar) for r_bar in self.rad]
        self.mass_dissk = [self.mass_disk(r_bar, self.sopt[0]) for r_bar in self.rad]

    
# ---------------- Rho_bulge ---------------------# 

    @staticmethod
    
    def rho_bulge(r, rho_0, r_b): 
        return rho_0 * np.exp(-r/r_b)    

# The code below is for the fitting parameters
    @staticmethod     
    def surface_density_bulge(r_bar_s, rho_0, r_b):
        def func(r, r_bar_s, rho_0, r_b):
            return (Galaxy.rho_bulge(r,rho_0,r_b) * r)/ (r**2 - r_bar_s**2)**0.5
        if hasattr(r_bar_s, '__len__'):
            return [2 * quad(func, r_bar, np.inf, args=(r_bar,rho_0, r_b))[0] for r_bar in r_bar_s]
        else:
            return 2 * quad(func, r_bar_s, np.inf, args=(r_bar_s,rho_0, r_b))[0]


# Finally I write the density of the bulge with the curve fit parameters interpolated
    def rho_bulge_final(self, r): 
        return Galaxy.rho_bulge(r, self.popt[0], self.popt[1])             #interpolation with SPARC data


# ---------------------- Rho_gas ----------------------- #

    @staticmethod
    def surface_density_gas(r_bar, sigma0, c1, c2, c3, c4, r_g):
        return sigma0 * (1 + c1 * r_bar + c2 * r_bar**2 + c3 * r_bar**3 + c4 * r_bar**4) * np.exp(-r_bar / r_g)
    
    def sdgas_interpolated(self, r_bar):
        return Galaxy.surface_density_gas(r_bar, self.qopt[0], self.qopt[1], self.qopt[2], self.qopt[3], self.qopt[4], self.qopt[5])

    def sdgas_prime(self, r_bar) :
        return derivative(self.sdgas_interpolated, r_bar)

    def rho_gas(self, r):
        def func(r_bar):
            return self.sdgas_prime(r_bar) / ((r_bar**2 - r**2)**0.5)
        return -1 / np.pi * quad(func, r, np.inf)[0]                      #Abel transform



#--------------------Rho_disk (*)---------------------------#

    @staticmethod
    def rho_disk(r_bar, z, rho_0, r_d):
        z_star = 0.196 * r_d**0.633  # kpc following Hossenfelder
        return rho_0 * np.exp(-r_bar / r_d) * (1 - (np.tanh(z / z_star))**2)

# The code below is for the fitting parameters
    @staticmethod
    def surface_density_disk(r_bar_s, rho_0, r_d):
        z_star = 0.196 * r_d**0.633
        if hasattr(r_bar_s, '__len__'):
            return [2 * rho_0 * z_star * np.exp(-r_bar / r_d) for r_bar in r_bar_s]
        else:
            return 2 * rho_0 * z_star * np.exp(-r_bar_s / r_d)

# Finally I write the density of the disk with the curve fit parameters interpolated
    def rho_disk_final(self, r_bar, z):
        return Galaxy.rho_disk(r_bar, z, self.ropt[0], self.ropt[1])



#-----------------------Final expressions------------------------------#

    def mass_bulge(self, r, Q):
        return (2.8/3) * np.pi * Q * (r**3) * self.rho_bulge_final(r)

    def mass_gas(self, r):
        return (4/3) * np.pi * (r**3) * self.rho_gas(r)

#    def mass_disk(self, r, Q):
#        def integrand(r_bar, z):
#            return r_bar * self.rho_disk_final(r_bar, z)
#        result, error = dblquad(integrand, 0, r, lambda z: 0, lambda z: (r**2 - z**2)**0.5)
#        return 2 * np.pi * Q * result


    def mass_disk(self, r, Q):
        def integrand(r_bar, z):
            return r_bar * self.rho_disk_final(r_bar, z)
        result, _ = dblquad(integrand, 0, r, lambda z:0, lambda z: (r**2 - z**2)**0.5) #[0]
        return 2 * np.pi * Q * result



    def mass_b(self, r, Q):
        return self.mass_gas(r) + self.mass_disk(r, Q) + self.mass_bulge(r, Q)

#    def rho_b(self, r_bar, z, Q):
#        r = (r_bar**2 + z**2)**0.5
#        return self.rho_gas(r) + 0.5 * Q * self.rho_disk_final(r_bar) + 0.7 * Q * self.rho_bulge_final(r_bar)

#    def mass_b(self, r, Q):
#        def integrand(r_bar, z, Q):
#            return 2 * np.pi * r_bar * self.rho_b(r_bar, z, Q)
#        result, _ = dblquad(integrand, 0, r, lambda z:0, lambda z:(r**2 - z**2)**0.5 [0])
#        return 2 * result

    def a_b(self, r, Q):
        return (G * self.mass_b(r, Q)) / (r**2)

    def a_0(self, α, Λ, M_pl):
#        M_pl = ((hbar * c)/(8*np.pi*G))**0.5
        frac = (α**3 * Λ**2) / (M_pl)
        return frac

    def a_tot(self, r, Q):
        return (self.a_0(α, Λ, M_pl) * self.a_b(r, Q))**0.5
    
    def v(self, r_s, Q):
        if hasattr(r_s, '__len__'):
            return [(r * self.a_tot(r, Q))**0.5 for r in r_s]
        else:
            return (r_s * self.a_tot(r_s, Q))**0.5

    def _v_fit(self, r_s, Q):
        return self.v(r_s, Q)

    def v_interpolated(self, r):
        return self.v(r, self.sopt[0])

#    def v(self, r_bar_s, Q):
#        α= 5.7
#        Λ = 0.05
#        if hasattr(r_bar_s, '__len__'):
#            return [np.sqrt(r_bar * (G * self.mass(r_bar, Q)/(r_bar**2) + np.sqrt((α**3 * Λ**2 / Mplanck) * G * self.mass(r_bar, Q)))) for r_bar in r_bar_s]
#        else:
#            return np.sqrt(r_bar_s * (G * self.mass(r_bar_s, Q) + np.sqrt((α**3 * Λ**2 / Mplanck) * G * self.mass(r_bar_s, Q))))

#    def _v_fit(self, r_bar_s, Q):
#        return self.v(r_bar_s, Q)

#    def v_interpolated(self, r_bar):
#        return self.v(r_bar, self.sopt[0])
