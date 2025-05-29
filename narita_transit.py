import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import matplotlib
matplotlib.use("TkAgg")  # Use an interactive backend
from tess_transits import tess_transit_data
from bls_fit import planet_b_period, planet_b_t0
period_b_bls = planet_b_period.value
tc_b_bls = planet_b_t0.value


# data_dir = "C:\\Users\\Paige\\Projects\\data\\k2-19_data"
data_dir = "/mnt/c/Users/Paige/Projects/data/k2-19_data"


file = "ajab5220t1_mrt.txt"
def read_table(file_name):
    ### path to table - Petigura et al 2020
    file_path = os.path.join(data_dir, "ajab5220t1_mrt.txt")

    ### Define the column names 
    columns = ["Planet", "Transit", "Inst", "Tc", "e_Tc", "Source"]

    ### Read the text file, specifying space as the delimiter, skipping model_guess_omc rows
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=22, names=columns)

    ### Remove NaN values
    df = df.dropna()
    return df

df = read_table(file)


# time offset from BJD
petigura_offset = 2454833 # BJD - 2454833
tess_offset = 2457000 # BTJD - Barycentric TESS Julian Date (Julian Date - 2457000)

planet_b_data = df[df["Planet"] == "K2-19b"]
planet_c_data = df[df["Planet"] == "K2-19c"]
# Print the DataFrame
print(df["Tc"])
print(planet_b_data["Tc"])
print(np.array(planet_b_data["Transit"]))

petigura_ind = np.array(planet_b_data["Transit"]) 

period_b_petigura = 7.9222 # [days]
tc_b_petigura = 2027.9023 # [days] using petigura offset
period_c_petigura = 11.8993 # [days]
tc_c_petigura = 2020.0007 # [days] using petigura offset
### get Narita et al times (K2)
df_narita = pd.read_csv('narita_times.txt', delim_whitespace=True)
df_narita_b = df_narita[df_narita["planet_num"] == 1]
tc_narita_b = np.array(df_narita_b["Tc"]) - petigura_offset
df_narita_c = df_narita[df_narita["planet_num"] == 2]
tc_narita_c = np.array(df_narita_c["Tc"]) - petigura_offset


### put times into petigura ephem (shift epoch by -6)
narita_ind_b = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3]
narita_ind_b = np.array(narita_ind_b)
narita_ind_c = [-3, -2, -1, 0, 1, 2, 3]
narita_ind_c = np.array(narita_ind_c)

tc_petigura_b = np.array(planet_b_data["Tc"])
print(f"Measured transit times b (Petigura): {tc_petigura_b}")

tc_petigura_c = np.array(planet_c_data["Tc"])
print(f"Measured transit times c (Petigura): {tc_petigura_c}")

'''
tc_guess=[]
for i in range(len(petigura_ind)):
    ind = tc_b_petigura + (petigura_ind[i]*period_b_petigura)
    tc_guess.append(ind)
print(f"Guess times (Petigura ephem): {tc_guess}")

print(f"Difference in measured and guess times based on Petigura ephem: {tc_guess - tc_petigura_b}")

tc_initial = []
for i in range(len(tc_petigura_b)):
    tc = tc_petigura_b[i] - (petigura_ind[i]*period_b_petigura)
    tc_initial.append(tc)

print(f"Inferred tc from Petigura et al: {tc_initial}")
'''

petigura_ind_b_updated = np.array(planet_b_data["Transit"]) - 6
print(f"Updated transit indices: {petigura_ind_b_updated}")
petigura_ind_c_updated = np.array(planet_c_data["Transit"]) - 3


### tess times in the paper ephem
tnum_tess = [337,339,340,341,342,343,430,432]

### updated indices for petigura ephem
tess_ind_b = np.array(tess_transit_data["Transit"]) + 337
tc_tess_b = np.array(tess_transit_data["Tc(TESS)"])
# tess_ind_b = tess_ind_b[:-2]
# tc_tess_b = tc_tess_b[:-2]

### collect all transit indices
all_transit_num_b = []
for i in range(len(narita_ind_b)):
    all_transit_num_b.append(narita_ind_b[i])
for i in range(len(petigura_ind_b_updated)):
    all_transit_num_b.append(petigura_ind_b_updated[i])
for i in range(len(tess_ind_b)):
    all_transit_num_b.append(tess_ind_b[i])
print(f"All transit indices: {all_transit_num_b}")

### collect all observed transit times
all_obs_tc_b = []
for i in range(len(tc_narita_b)):
    all_obs_tc_b.append(tc_narita_b[i])
for i in range(len(tc_petigura_b)):
    all_obs_tc_b.append(tc_petigura_b[i])
for i in range(len(tc_tess_b)):
    all_obs_tc_b.append(tc_tess_b[i])
print(f"All observed transit times: {all_obs_tc_b}")

### collect all observed transit time errors
all_obs_tc_b_err = []
tc_b_err_narita = np.array(df_narita_b["Tc_err"])
tc_b_err_tess = np.array(tess_transit_data["Tc_err"])
tc_b_err_petigura = np.array(planet_b_data["e_Tc"])
for i in range(len(tc_b_err_narita)):
    all_obs_tc_b_err.append(tc_b_err_narita[i])
for i in range(len(tc_b_err_petigura)):
    all_obs_tc_b_err.append(tc_b_err_petigura[i])
for i in range(len(tc_b_err_tess)):
    all_obs_tc_b_err.append(tc_b_err_tess[i])
print(f"All observed transit times error: {all_obs_tc_b_err}")

### collect all calculated transit times with petigura ephem
all_calc_tc_b = []
for i in range(len(all_transit_num_b)):
    ind = tc_b_petigura + (all_transit_num_b[i] * period_b_petigura)
    all_calc_tc_b.append(ind)
print(f"All calculated transit times: {all_calc_tc_b}")

### collect all omc
all_omc_b = []
for i in range(len(all_calc_tc_b)):
    omc_ = all_obs_tc_b[i] - all_calc_tc_b[i]
    all_omc_b.append(omc_)
print(f"All OMC: {all_omc_b}")

all_omc_b_err = all_obs_tc_b_err
print(f"All OMC err: {all_omc_b_err}")

###########################################################################################################
### collect all values for planet c
### collect all transit indices
all_transit_num_c = []
for i in range(len(narita_ind_c)):
    all_transit_num_c.append(narita_ind_c[i])
for i in range(len(petigura_ind_c_updated)):
    all_transit_num_c.append(petigura_ind_c_updated[i])

print(f"All transit indices c: {all_transit_num_c}")

### collect all observed transit times
all_obs_tc_c = []
for i in range(len(tc_narita_c)):
    all_obs_tc_c.append(tc_narita_c[i])
for i in range(len(tc_petigura_c)):
    all_obs_tc_c.append(tc_petigura_c[i])

print(f"All observed transit times c: {all_obs_tc_c}")

### collect all observed transit time errors
all_obs_tc_c_err = []
tc_c_err_narita = np.array(df_narita_c["Tc_err"])
tc_c_err_petigura = np.array(planet_c_data["e_Tc"])
for i in range(len(tc_c_err_narita)):
    all_obs_tc_c_err.append(tc_c_err_narita[i])
for i in range(len(tc_c_err_petigura)):
    all_obs_tc_c_err.append(tc_c_err_petigura[i])
print(f"All observed transit times error c: {all_obs_tc_c_err}")

### collect all calculated transit times with petigura ephem
all_calc_tc_c = []
for i in range(len(all_transit_num_c)):
    ind = tc_c_petigura + (all_transit_num_c[i] * period_c_petigura)
    all_calc_tc_c.append(ind)
print(f"All calculated transit times c: {all_calc_tc_c}")

### collect all omc
all_omc_c = []
for i in range(len(all_calc_tc_c)):
    omc_ = all_obs_tc_c[i] - all_calc_tc_c[i]
    all_omc_c.append(omc_)
print(f"All OMC c: {all_omc_c}")

all_omc_c_err = all_obs_tc_c_err
print(f"All OMC err c: {all_omc_c_err}")



### fitting the omc
all_transit_num_b = np.array(all_transit_num_b)
### Define a sine function
# def sine_function(ind, A, omega, phi, p, offset):
#     return A * np.sin(omega * ind + phi) + (p * ind) + offset

# def sine_function(ind, A, omega, phi, b, p, offset):
#     return A * np.sin(omega * ind + phi) + (b * ind**2) + (p * ind) + offset

def sine_function(ind, A1, omega1, phi1, A2, omega2, phi2, b, p, offset):
    sine_term1 = A1 * np.sin(omega1 * ind + phi1)
    sine_term2 = A2 * np.sin(omega2 * ind + phi2)
    quadratic_term = b * ind**2
    linear_term = p * ind
    constant_offset = offset
    
    return sine_term1 + sine_term2 + quadratic_term + linear_term + constant_offset

### Initial guess for the parameters: amplitude, frequency, phase, period, offset
# initial_guess = [0.07, 2*np.pi/100, -np.pi/2, 7.92089, 2027.84470]
# initial_guess = [0.08, 2*np.pi/100, -0.5293, 0, 7.920985, 2027.838470871639]
# initial_guess = [0.07, 2*np.pi/100, -np.pi/2, 0, 7.92089, 2027.84470]
### Initial guess for the parameters: amplitude1, frequency1, phase1, amplitude2, frequency2, phase2, quadratic, period, offset
# initial_guess = [0.02, 2*np.pi/100, -np.pi/2, 0.05, 2*np.pi/100, -np.pi/2, 0, 7.92089, 2027.84470]
### initial guess with best fit
initial_guess = [0.03, 0.10526351520898146, -64.6530084190281, 0.015670698974558715, 0.09234863218711666, 27.291244929981197, -1.485667343253648e-08, 7.92092123921758, 2027.9073172414364]

# Fit the sine function to the data
# popt, pcov = curve_fit(sine_function, all_transit_num_b, all_obs_tc_b, p0=initial_guess, sigma=all_omc_b_err, absolute_sigma=True)
# A_fit, omega_fit, phi_fit,b_fit, period_fit, offset_fit = popt
# print(f'fit parameters: {A_fit}, {omega_fit}, {phi_fit}, {b_fit}, {period_fit}, {offset_fit}')
# print(f'offset_fit: {offset_fit}')
# print(f'period_fit: {period_fit}')

popt, pcov = curve_fit(sine_function, all_transit_num_b, all_obs_tc_b, p0=initial_guess, sigma=all_omc_b_err, absolute_sigma=True)
A1_fit, omega1_fit, phi1_fit,A2_fit, omega2_fit, phi2_fit, b_fit, period_fit, offset_fit = popt
print(f'fit parameters: {A1_fit}, {omega1_fit}, {phi1_fit}, {A2_fit}, {omega2_fit}, {phi2_fit}, {b_fit}, {period_fit}, {offset_fit}')
print(f'offset_fit: {offset_fit}')
print(f'period_fit: {period_fit}')


# def ephem_fit(transit_num):
#     return offset_fit + (period_fit * transit_num)
# def ephem_guess(transit_num):
#     return initial_guess[4] + (transit_num * initial_guess[3])
def ephem_fit(transit_num):
    return offset_fit + (period_fit * transit_num) + (b_fit * transit_num**2)
def ephem_guess(transit_num):
    return initial_guess[5] + (transit_num * initial_guess[4]) + (transit_num**2 * initial_guess[3])

### for plotting
transit_num_plot = np.linspace(min(all_transit_num_b),max(all_transit_num_b),1000)
model_fit_sine = sine_function(transit_num_plot, *popt) - ephem_fit(transit_num_plot)

omc = all_obs_tc_b - ephem_fit(all_transit_num_b)

model_guess_sine = sine_function(transit_num_plot, *initial_guess) - ephem_fit(transit_num_plot)

residuals = (omc - sine_function(all_transit_num_b, *popt) + ephem_fit(all_transit_num_b)) / all_obs_tc_b_err # [days]/[days] -> unitless

omc_narita = np.array(omc[:10])
omc_err_narita = np.array(all_omc_b_err[:10])
residual_narita = np.array(residuals[:10])
omc_petigura = np.array(omc[10:16])
omc_err_petigura = np.array(all_omc_b_err[10:16])
residual_petigura = np.array(residuals[10:16])
omc_tess = np.array(omc[16:])
omc_err_tess = np.array(all_omc_b_err[16:])
residual_tess = np.array(residuals[16:])

### plot readability, plot in ttv hours
day2hr = 24

fig2, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},sharex=True)
ax1.plot(ephem_fit(transit_num_plot), model_guess_sine*day2hr, label='Guess sine curve')
ax1.plot(ephem_fit(transit_num_plot), model_fit_sine*day2hr, 'r-', label='Fitted sine curve')
ax1.errorbar(ephem_fit(narita_ind_b), omc_narita*day2hr, yerr=omc_err_narita*day2hr, fmt='ys',label='Narita', capsize=5)
ax1.errorbar(ephem_fit(petigura_ind_b_updated), omc_petigura*day2hr, yerr=omc_err_petigura*day2hr, fmt='bs',label='Petigura', capsize=5)
ax1.errorbar(ephem_fit(tess_ind_b), omc_tess*day2hr, yerr=omc_err_tess*day2hr, fmt='go',label='TESS', capsize=5)

ax1.set_title('TTV Fit: Planet b')
ax1.set_ylabel('TTV (hour)')
ax1.legend(loc='lower left')

ax2.plot(ephem_fit(narita_ind_b), residual_narita,'ys')
ax2.plot(ephem_fit(petigura_ind_b_updated), residual_petigura,'bs')
ax2.plot(ephem_fit(tess_ind_b), residual_tess,'go')
ax2.axhline(y=0, color='r', linestyle='-')
ax2.set_ylabel('Norm Residuals') #hour?? day?? unitless
ax2.set_xlabel('TC (BJD - 2454833 days)')
plt.show()

### ttv_fast plot
# call ttv_fast 
# def ttv_fast(ind, theta)
    # ttv_fast(_,_,theta)
    # tts = dict[tts]
    # call curvefit with ttv_fast func instead of sine 
# 
'''
from ttvfast_run import ttvfast_sim, planet1_params, planet2_params
mass1,per1,ecc1,i1,Omega1,argument1,M1 = planet1_params
print(mass1, per1, ecc1, i1,Omega1,argument1,M1)
mass2,per2,ecc2,i2,Omega2,argument2,M2 = planet2_params
initial_guess_ttvf = [2218, 5500, planet1_params, planet2_params]
popt_ttvf, pcov_ttvf = curve_fit(ttvfast_sim, all_transit_num_b, all_obs_tc_b, p0=initial_guess_ttvf, sigma=all_omc_b_err, absolute_sigma=True)
start_time_fit, end_time_fit, planet1_fit, planet2_fit = popt_ttvf
print(f'Optimized params: {start_time_fit}, {end_time_fit}, {planet1_fit}, {planet2_fit}')

### for plotting
transit_num_plot = np.linspace(min(all_transit_num_b),max(all_transit_num_b),1000)
model_fit_ttvf = ttvfast_sim(transit_num_plot, *popt_ttvf) - ephem_fit(transit_num_plot)

omc = all_obs_tc_b - ephem_fit(all_transit_num_b)

model_guess_ttvf = ttvfast_sim(transit_num_plot, *initial_guess_ttvf) - ephem_fit(transit_num_plot)

residuals_ttvf = (omc - ttvfast_sim(all_transit_num_b, *popt_ttvf) + ephem_fit(all_transit_num_b)) / all_obs_tc_b_err # [days]/[days] -> unitless

fig2, (ax3, ax4) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},sharex=True)
ax3.plot(ephem_fit(transit_num_plot), model_guess_ttvf*day2hr, label='Guess sine curve')
ax3.plot(ephem_fit(transit_num_plot), model_fit_ttvf*day2hr, 'r-', label='Fitted sine curve')
ax3.errorbar(ephem_fit(narita_ind_b), omc_narita*day2hr, yerr=omc_err_narita*day2hr, fmt='ys',label='Narita', capsize=5)
ax3.errorbar(ephem_fit(petigura_ind_b_updated), omc_petigura*day2hr, yerr=omc_err_petigura*day2hr, fmt='bs',label='Petigura', capsize=5)
ax3.errorbar(ephem_fit(tess_ind_b), omc_tess*day2hr, yerr=omc_err_tess*day2hr, fmt='go',label='TESS', capsize=5)

ax3.set_title('TTV Fit: Planet b')
ax3.set_ylabel('TTV (hour)')
ax3.legend(loc='lower left')

ax4.plot(ephem_fit(all_transit_num_b), residuals_ttvf)
# ax4.plot(ephem_fit(narita_ind_b), residual_narita,'ys')
# ax4.plot(ephem_fit(petigura_ind_b_updated), residual_petigura,'bs')
# ax4.plot(ephem_fit(tess_ind_b), residual_tess,'go')
ax4.axhline(y=0, color='r', linestyle='-')
ax4.set_ylabel('Norm Residuals') #hour?? day?? unitless
ax4.set_xlabel('TC (BJD - 2454833 days)')
plt.show()

# from ttvfast_run import times_1, omc_1, times_2, omc_2, start_time
# # plt.scatter(times_1,omc_1,color='orange',s=4,label='planet b ttvfast')
# # plt.scatter(times_2,omc_2,color='blue',s=4, label='planet c ttvfast')
# plt.errorbar(tc_narita_b, omc_narita, xerr=tc_b_err_narita, yerr=omc_err_narita, fmt='ys', capsize=5, label='Narita')
# plt.errorbar(tc_petigura_b, omc_petigura, xerr=tc_b_err_petigura, yerr=omc_err_petigura, fmt='rs', capsize=5, label='Petigura')
# plt.errorbar(tc_tess_b, omc_tess, xerr=tc_b_err_tess, yerr=omc_err_tess, fmt='go', capsize=5, label='TESS')
# plt.plot(ephem_fit(transit_num_plot), model_fit_sine,color='black', label='Fitted Sine')
# plt.title(f"TTVfast and Measured Time: K2-19 (start time {start_time})")
# plt.ylabel('TTV (days)')
# plt.xlabel('TC (days)')
# plt.legend()
# plt.show()
'''

### make txt file with results 
output_file = "ttv_results.txt"

### Combine all data into a structured format
source = ['Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', # b
          'Petigura2020', 'Petigura2020', 'Petigura2020', 'Petigura2020', 'Petigura2020', 'Petigura2020', # b
          'TESS', 'TESS', 'TESS', 'TESS', 'TESS', 'TESS', 'TESS', 'TESS', # b
          'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', # c
          'Petigura2020', 'Petigura2020'] # c

instrument = ['K2', 'K2', 'K2', 'K2', 'K2', 'K2', 'K2', 'K2', 'K2', 'K2', # b
          'FLWO', 'TRAPPIST', 'MuSCAT', 'Spitzer', 'LCO', 'Spitzer',  # b
          'TESS', 'TESS', 'TESS', 'TESS', 'TESS', 'TESS', 'TESS', 'TESS', # b
          'K2', 'K2', 'K2', 'K2', 'K2', 'K2', 'K2', # c
          'Spitzer', 'Spitzer'] # c

planet_number = []
for i in range(len(all_obs_tc_b)):
    planet_number.append(1)
for i in range(len(all_obs_tc_c)):
    planet_number.append(2)

all_transit_num = [] 
for i in range(len(all_transit_num_b)):
    all_transit_num.append(all_transit_num_b[i])
for i in range(len(all_transit_num_c)):
    all_transit_num.append(all_transit_num_c[i])

all_obs_tc = []
for i in range(len(all_obs_tc_b)):
    all_obs_tc.append(all_obs_tc_b[i])
for i in range(len(all_obs_tc_c)):
    all_obs_tc.append(all_obs_tc_c[i])

all_obs_tc_err = []
for i in range(len(all_obs_tc_b_err)):
    all_obs_tc_err.append(all_obs_tc_b_err[i])
for i in range(len(all_obs_tc_c_err)):
    all_obs_tc_err.append(all_obs_tc_c_err[i])

all_omc = []
for i in range(len(all_omc_b)):
    all_omc.append(all_omc_b[i])
for i in range(len(all_omc_c)):
    all_omc.append(all_omc_c[i])

### Ensure source and instrument are numpy arrays with dtype=object
source = np.array(source, dtype=object)
instrument = np.array(instrument, dtype=object)


data_to_save = np.column_stack((planet_number, all_transit_num, all_obs_tc, all_obs_tc_err, all_omc, source, instrument))

### Define header
header = "Planet_num Index Tc Tc_err OMC Source Instrument"

### Save to file
np.savetxt(output_file, data_to_save, fmt='%d %d %.8f %.8f %.8f %s %s', header=header, comments='')

print(f"Results saved to {output_file}")