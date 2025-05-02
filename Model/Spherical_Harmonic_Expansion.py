import numpy as np
from scipy.special import sph_harm
from scipy.integrate import dblquad
from scipy.interpolate import RectBivariateSpline
import time

# load data from txt file
angles = np.loadtxt('angle.txt')
data = np.load('rho.npy')

a00 = []
a10 = []
a20 = []

time_start = time.time()

for i in range(300):
    theta = angles[:, 0]
    phi = angles[:, 1]
    rho = data[i + 150,:]      

    period = np.pi

    theta = np.mod(theta, period)
    phi = np.mod(phi, period)

    theta_unique = np.unique(theta)
    phi_unique = np.unique(phi)

    rho_grid = rho.reshape(len(theta_unique), len(phi_unique))

    l_max = 2

    def rho_interp(theta_val, phi_val):
        interp_func_1 = RectBivariateSpline(theta_unique, phi_unique, rho_grid, kx=3, ky=3)
        theta_true = theta_val % np.pi
        phi_true = phi_val % np.pi
        return interp_func_1(theta_true, phi_true)

    def compute_spherical_harmonic_coefficients(f_interp, l_max):
        coeffs = []
        for l in range(l_max + 1):
            for m in [0]:
                def integrand_real(theta, phi):
                    value = f_interp(theta, phi) * np.conj(sph_harm(m, l, phi, theta)) * np.sin(theta)
                    return value.real
                
                coeff_lm_real, _ = dblquad(integrand_real, 0, 2 * np.pi, lambda phi: 0, lambda phi: np.pi, epsabs=1e-4, epsrel=1e-6)    
                
                if np.abs(coeff_lm_real) < 1e-4:
                    coeff_lm_real = 0
                    
                coeffs.append((l, m, coeff_lm_real))
        
        return coeffs

    coefficients_rho = []
    coefficients_rho = compute_spherical_harmonic_coefficients(rho_interp, l_max)

    for (l, m, coeff_real) in coefficients_rho:
        print(f"a_{l}^{m} = {coeff_real:4f}")
        if l == 0 and m == 0:
            a00.append(coeff_real)
            
        elif l == 1 and m == 0:
            a10.append(coeff_real)
        
        elif l == 2 and m == 0:
            a20.append(coeff_real)
            
    print(f'Expansion {i + 1} finished!, time cost: {time.time() - time_start}')
            
np.savetxt('a00.txt', a00)
np.savetxt('a10.txt', a10)
np.savetxt('a20.txt', a20)