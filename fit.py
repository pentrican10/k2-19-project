import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from lightkurve import search_targetpixelfile
import pandas as pd
import os
import batman
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import root_scalar

### Download the light curve data
lc = lk.search_lightcurve("K2-19",author = 'SPOC').download_all()

### Flatten the light curve
lc = lc.stitch().flatten(window_length=901).remove_outliers()

### Create array of periods to search
period = np.linspace(1, 20, 10000)
### Create a BLSPeriodogram
bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500)
### exctract period and t0
planet_b_period = bls.period_at_max_power
planet_b_t0 = bls.transit_time_at_max_power

### initialize guess times
transit_num = [0,2,3,4,5,6,93,95]
tc_guess=[]
for num in transit_num:
    t = planet_b_t0.value + (num * planet_b_period.value)
    tc_guess.append(t)
#lc.plot()
#plt.show()
time_tess = np.array(lc.time.value)
flux=np.array(lc.flux)
flux_err = np.array(lc.flux_err)

def convert_time(times):
    ### TESS offset 
    BTJD = times + 2457000
    new_time = BTJD - 2454833
    return new_time


time = convert_time(time_tess)
tc_guess = convert_time(np.array(tc_guess))

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
### set range for search: [#hours] * [days per hour]
ttv_hour = 2* 0.0416667 # 1 hour to days
#tc_guess = (2530.28, 2546.12, 2554.04, 2561.96, 2569.88, 2577.8, 3266.84, 3282.68)
#t1 guess
### get tc ranges 
tc = []
for i in range(len(tc_guess)):
    start = tc_guess[i] - ttv_hour
    end = tc_guess[i] + ttv_hour
    t = np.linspace(start,end, 100)
    tc.append(t)


tc_chi = np.zeros(len(tc))
errors = []
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
    min_chi = chi_sq.min()
    tc_chi[j] = min_chi_idx

    ### delta chisq = 1 gives errors
    err_threshold = min_chi + 1
    # Find the intersection points
    def intersection_func(t):
        return np.interp(t, tc1, chi_sq) - err_threshold
    
    # Find the intersection using root_scalar
    intersections = []
    for k in range(len(tc1) - 1):
        if (chi_sq[k] - err_threshold) * (chi_sq[k + 1] - err_threshold) < 0:
            sol = root_scalar(intersection_func, bracket=[tc1[k], tc1[k + 1]])
            if sol.converged:
                intersections.append((sol.root - min_chi_idx))
    errors.append(intersections)
    
    plt.plot(tc1, chi_sq,label='chisq')
    plt.axvline(x=tc_guess[j], color='r', linestyle='--', label='Bls Guess')
    plt.axvline(x=min_chi_idx, color='green', linestyle='--', label='Chisq min')
    # for inter in intersections:
    #     plt.axvline(x=inter, color='blue', linestyle='--')
    plt.axhline(y=err_threshold, color='purple', linestyle='--', label='Error Threshold')
    plt.title(f'Transit {j+1}: Planet b')
    plt.xlabel('tc')
    plt.ylabel('X^2')
    plt.legend()
    plt.show()

print(tc_chi)
print(errors)












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