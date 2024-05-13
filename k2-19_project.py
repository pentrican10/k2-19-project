import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from lightkurve import search_targetpixelfile
import pandas as pd
import os
import batman
from scipy.stats import norm

data_dir = "C:\\Users\\Paige\\Projects\\data\\k2-19_data"

### path to table - Petigura et al 2020
file_path = os.path.join(data_dir, "ajab5220t1_mrt.txt")

### Define the column names 
columns = ["Planet", "Transit", "Inst", "Tc", "e_Tc", "Source"]

### Read the text file, specifying space as the delimiter, skipping initial rows
df = pd.read_csv(file_path, delim_whitespace=True, skiprows=22, names=columns)

### Remove NaN values
df = df.dropna()

# ### Display the DataFrame
# #print(df)

# Download the light curve data
lc = lk.search_lightcurve("K2-19",author = 'SPOC').download_all()

# Flatten the light curve
lc = lc.stitch().flatten(window_length=901).remove_outliers()
#lc.plot()
time = lc.time
flux=lc.flux
flux_err = lc.flux_err



# def log_likelihood(theta, x, y, yerr):
#     # Initialize batman transit model
#     params = batman.TransitParams()
#     params.t0 = theta[0]  # Transit center time
#     params.per = theta[1]  # Orbital period
#     params.rp = theta[2]   # Planet-to-star radius ratio
#     params.b = theta[3]    
#     params.T14 = theta[4]  
#     q1=theta[5]
#     q2=theta[6]
#     params.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]  # Limb darkening coefficients
#     params.limb_dark = 'quadratic'

#     transit_model = batman.TransitModel(params, x)
    
#     # Generate model light curve
#     model_flux = transit_model.light_curve(params)
    
#     # Calculate chi-squared value
#     sigma2 = yerr**2
#     chi_squared = np.sum(((y - model_flux) / sigma2)**2)
    
#     # Compute log-likelihood
#     log_likelihood = -0.5 * chi_squared
    
#     return log_likelihood



# '''
# def log_likelihood(theta, x, y, yerr):
#     # Create a TransitParams object
#     params = batman.TransitParams()
#     params.t0 = theta[0]  # Transit center time
#     params.per = theta[1]  # Orbital period
#     params.rp = theta[2]   # Planet-to-star radius ratio
#     params.b = theta[3]    
#     params.T14 = theta[4]  
#     q1=theta[5]
#     q2=theta[6]
#     params.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]  # Limb darkening coefficients
#     params.limb_dark = 'quadratic'

#     # Initialize the TransitModel with the TransitParams
#     model = batman.TransitModel(params, np.array(x))

#     # Generate model predictions
#     model_flux = model.light_curve(params)

#     # Compute residuals
#     residuals = np.array(y) - model_flux

#     # Compute likelihood assuming Gaussian distribution
#     likelihoods = norm.logpdf(residuals, loc=0, scale=np.array(yerr))

#     # Sum the log-likelihoods
#     log_likelihood_value = np.sum(likelihoods)

#     return log_likelihood_value
# '''
# # Example parameters
# per_b = 7.9222
# rp_b = 0.0777
# T14_b = 3.237
# b_b = 0.17
# t0_b = 	2456860.902
# q1_b = 0.4
# q2_b = 0.3
# theta_initial = [t0_b, per_b, rp_b, b_b, T14_b, q1_b, q2_b]
# # Calculate log-likelihood for initial parameters
# lg_like = log_likelihood(theta_initial, time, flux, flux_err)

# print(lg_like)


# Create array of periods to search
period = np.linspace(1, 20, 10000)
# Create a BLSPeriodogram
bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500)
bls.plot()

planet_b_period = bls.period_at_max_power
planet_b_t0 = bls.transit_time_at_max_power
planet_b_dur = bls.duration_at_max_power
# Check the value for period
print(planet_b_period)
print(planet_b_t0)

ax = lc.fold(period=planet_b_period, epoch_time=planet_b_t0).scatter()
ax.set_xlim(-5, 5)
#plt.show()
# Create a cadence mask using the BLS parameters
planet_b_mask = bls.get_transit_mask(period=planet_b_period,
                                     transit_time=planet_b_t0,
                                     duration=planet_b_dur)
masked_lc = lc[~planet_b_mask]
ax = masked_lc.scatter()
lc[planet_b_mask].scatter(ax=ax, c='r', label='Masked')

# Create a BLS model using the BLS parameters
planet_b_model = bls.get_transit_model(period=planet_b_period,
                                       transit_time=planet_b_t0,
                                       duration=planet_b_dur)

# Find all times where the transit model of planet b is minimum
# min_times_b = lc.time[planet_b_model == np.min(planet_b_model)]
# print(min_times_b)
# assert 1==0
ax = lc.fold(planet_b_period, planet_b_t0).scatter()
planet_b_model.fold(planet_b_period, planet_b_t0).plot(ax=ax, c='r', lw=2)
ax.set_xlim(-5, 5)

period = np.linspace(1, 300, 10000)
bls = masked_lc.to_periodogram('bls', period=period, frequency_factor=500)
bls.plot()

planet_c_period = bls.period_at_max_power
planet_c_t0 = bls.transit_time_at_max_power
planet_c_dur = bls.duration_at_max_power

# Check the value for period
print(planet_c_period)

ax = masked_lc.fold(planet_c_period, planet_c_t0).scatter()
masked_lc.fold(planet_c_period, planet_c_t0).bin(.1).plot(ax=ax, c='r', lw=2,
                                                          label='Binned Flux')
ax.set_xlim(-5, 5)

planet_c_model = bls.get_transit_model(period=planet_c_period,
                                       transit_time=planet_c_t0,
                                       duration=planet_c_dur)

ax = lc.scatter()
planet_b_model.plot(ax=ax, c='dodgerblue', label='Planet b Transit Model')
planet_c_model.plot(ax=ax, c='r', label='Planet c Transit Model')

plt.show()

