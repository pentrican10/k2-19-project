import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from lightkurve import search_targetpixelfile
import pandas as pd
import os
import batman
from scipy.stats import norm
from scipy.optimize import minimize

# Download the light curve data
lc = lk.search_lightcurve("K2-19",author = 'SPOC').download_all()

# Flatten the light curve
lc = lc.stitch().flatten(window_length=901).remove_outliers()
#lc.plot()
#plt.show()
time = np.array(lc.time.value)
flux=np.array(lc.flux)
flux_err = np.array(lc.flux_err)

#'''
#plt.scatter(time, flux)
#plt.plot(time, model_flux, color='red')
#plt.show()

def log_likelihood(theta, x, y, yerr):
    # Initialize batman transit model
    params = batman.TransitParams()
    params.t0, params.per, params.rp,params.b, params.T14, q1, q2 = theta

    params.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]  # Limb darkening coefficients
    params.limb_dark = 'quadratic'
    
    transit_model = batman.TransitModel(params, x)
    
    # Generate model light curve
    model_flux = transit_model.light_curve(params)
    
    # Calculate chi-squared value
    sigma2 = yerr**2 #f value from tutorial??  model**2 * np.exp(2*log_f)
    chi_squared = np.sum(((y - model_flux) / sigma2)**2)
    print(chi_squared)
    
    # Compute log-likelihood
    log_likelihood = -0.5 * chi_squared
    
    return log_likelihood

### params from exoplanet archive
per_b = 7.9222
rp_b = 0.0777
T14_b = 3.237 * 0.0416667  # convert to days
b_b = 0.17
q1_b = 0.4
q2_b = 0.3

### number of transits for planet b
num_b = 8
num_c = 2

#tc1 = np.linspace(time.min(),2600, 100)
ttv_min= 0.00694444
ttv_hour = 6* 0.0416667 # 1 hour to days
tc_guess = (2530.2, 2546, 2554, 2562, 2570, 2577.8, 3266.8, 3282.7)
#t1 guess
### get tc ranges 
tc = []
for i in range(len(tc_guess)):
    start = tc_guess[i] - ttv_hour
    end = tc_guess[i] + ttv_hour
    t = np.linspace(start,end, 100)
    tc.append(t)


tc_chi = np.zeros(len(tc))
### plot X^2 vs tc for each guess
for j in range(len(tc)):
    tc1 = tc[j]
    chi_sq = np.zeros(len(tc1))
    for i in range(len(tc1)):
        t0_b = 	tc1[i]
        theta_initial = [t0_b, per_b, rp_b, b_b, T14_b, q1_b, q2_b]
        ### initialize params
        params = batman.TransitParams()
        params.t0, params.per, params.rp,params.b, params.T14, q1, q2 = theta_initial
        params.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]  # Limb darkening coefficients
        params.limb_dark = 'quadratic'
        
        ### mask data - extract relevant photometry
        start = tc_guess[j] - ttv_hour
        end = tc_guess[j] + ttv_hour
        mask = (time > (start)) & (time < (end))
        
        transit_model = batman.TransitModel(params, time[mask])
            
        # Generate model light curve
        model_flux = transit_model.light_curve(params)
        
        # Calculate chi-squared value
        sigma2 = flux_err[mask] 
        chi_squared = np.sum(((flux[mask] - model_flux) / sigma2)**2)
        chi_sq[i] = (chi_squared)
        

    #print(chi_sq)
    min_chi_idx = tc1[np.argmin(chi_sq)]
    tc_chi[j] = min_chi_idx
    plt.plot(tc1, chi_sq,label='chisq')
    plt.axvline(x=tc_guess[j], color='r', linestyle='--', label='Bls Guess')
    plt.axvline(x=min_chi_idx, color='green', linestyle='--', label='Chisq min')
    plt.title(f'Transit {j+1}: Planet b')
    plt.xlabel('tc')
    plt.ylabel('X^2')
    plt.legend()
    plt.show()












'''
# Example parameters
per_b = 7.9222
rp_b = 0.0777
T14_b = 3.237 * 0.0416667  # convert to days
b_b = 0.17
t0_b = 	2530
q1_b = 0.4
q2_b = 0.3
theta_initial = [t0_b, per_b, rp_b, b_b, T14_b, q1_b, q2_b]
# Calculate log-likelihood for initial parameters
#lg_like = log_likelihood(theta_initial, time, flux, flux_err)


params = batman.TransitParams()
params.t0, params.per, params.rp,params.b, params.T14, q1, q2 = theta_initial

params.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]  # Limb darkening coefficients
params.limb_dark = 'quadratic'
    
transit_model = batman.TransitModel(params, time)
    
    # Generate model light curve
model_flux = transit_model.light_curve(params)
mask = (time > (t0_b - 5*0.0416667)) & (time < (t0_b + 5*0.0416667))
plt.scatter(time, flux)
plt.plot(time, model_flux, color='red')
plt.show()

# Calculate chi-squared value
sigma2 = flux_err #f value from tutorial??  model**2 * np.exp(2*log_f)
chi_squared = np.sum(((flux - model_flux) / sigma2)**2)
#print(chi_squared)

'''






# nll = lambda *args: -log_likelihood(*args)
# soln = minimize(nll, theta_initial, args=(time, flux, flux_err))
# t0, per, rp, b, T14, q1, q2 = soln.x

# print("Maximum likelihood estimates:")
# print("t0 = {0:.3f}".format(t0))
# print("per = {0:.3f}".format(per))
# print("rp = {0:.3f}".format(rp))
# print("b = {0:.3f}".format(b))
# print("T14 = {0:.3f}".format(T14))

#print(lg_like)