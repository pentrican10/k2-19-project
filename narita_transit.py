import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

from tess_transits import tess_transit_data
from bls_fit import planet_b_period, planet_b_t0
period_b_bls = planet_b_period.value
tc_b_bls = planet_b_t0.value


data_dir = "C:\\Users\\Paige\\Projects\\data\\k2-19_data"


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

# Print the DataFrame
print(df["Tc"])
print(planet_b_data["Tc"])
print(np.array(planet_b_data["Transit"]))

petigura_ind = np.array(planet_b_data["Transit"]) 

period_b_petigura = 7.9222 # [days]
tc_b_petigura = 2027.9023 # [days] using petigura offset

### get Narita et al times (K2)
df_narita = pd.read_csv('narita_times.txt', delim_whitespace=True)
tc_narita = np.array(df_narita["Tc"]) - petigura_offset
### put times into petigura ephem (shift epoch by -6)
narita_ind = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3]
narita_ind = np.array(narita_ind)

tc_petigura = np.array(planet_b_data["Tc"])
print(f"Measured transit times (Petigura): {tc_petigura}")

tc_guess=[]
for i in range(len(petigura_ind)):
    t = tc_b_petigura + (petigura_ind[i]*period_b_petigura)
    tc_guess.append(t)
print(f"Guess times (Petigura ephem): {tc_guess}")

print(f"Difference in measured and guess times based on Petigura ephem: {tc_guess - tc_petigura}")

tc_initial = []
for i in range(len(tc_petigura)):
    tc = tc_petigura[i] - (petigura_ind[i]*period_b_petigura)
    tc_initial.append(tc)

print(f"Inferred tc from Petigura et al: {tc_initial}")

petigura_ind_updated = np.array(planet_b_data["Transit"]) - 6
print(f"Updated transit indices: {petigura_ind_updated}")



### tess times in the paper ephem
tnum_tess = [337,339,340,341,342,343,430,432]
### updated indices for petigura ephem
tess_ind = np.array(tess_transit_data["Transit"]) + 337
tc_tess = np.array(tess_transit_data["Tc(TESS)"])

### collect all transit indices
all_transit_num = []
for i in range(len(narita_ind)):
    all_transit_num.append(narita_ind[i])
for i in range(len(petigura_ind_updated)):
    all_transit_num.append(petigura_ind_updated[i])
for i in range(len(tess_ind)):
    all_transit_num.append(tess_ind[i])
print(f"All transit indices: {all_transit_num}")

### collect all observed transit times
all_obs_tc = []
for i in range(len(tc_narita)):
    all_obs_tc.append(tc_narita[i])
for i in range(len(tc_petigura)):
    all_obs_tc.append(tc_petigura[i])
for i in range(len(tc_tess)):
    all_obs_tc.append(tc_tess[i])
print(f"All observed transit times: {all_obs_tc}")

### collect all observed transit time errors
all_obs_tc_err = []
tc_err_narita = np.array(df_narita["Tc_err"])
tc_err_tess = np.array(tess_transit_data["Tc_err"])
tc_err_petigura = np.array(planet_b_data["e_Tc"])
for i in range(len(tc_err_narita)):
    all_obs_tc_err.append(tc_err_narita[i])
for i in range(len(tc_err_petigura)):
    all_obs_tc_err.append(tc_err_petigura[i])
for i in range(len(tc_err_tess)):
    all_obs_tc_err.append(tc_err_tess[i])
print(f"All observed transit times error: {all_obs_tc_err}")

### collect all calculated transit times with petigura ephem
all_calc_tc = []
for i in range(len(all_transit_num)):
    t = tc_b_petigura + (all_transit_num[i] * period_b_petigura)
    all_calc_tc.append(t)
print(f"All calculated transit times: {all_calc_tc}")

### collect all omc
all_omc = []
for i in range(len(all_calc_tc)):
    omc_ = all_obs_tc[i] - all_calc_tc[i]
    all_omc.append(omc_)
print(f"All OMC: {all_omc}")

all_omc_err = all_obs_tc_err
print(f"All OMC err: {all_omc_err}")


### fitting the omc
all_transit_num = np.array(all_transit_num)
### Define a sine function
def sine_function(t, A, omega, phi, p, offset):
    return A * np.sin(omega * t + phi) + (p * t) + offset

### Initial guess for the parameters: amplitude, frequency, phase, period, offset
# initial_guess = [0.07, 2*np.pi/100, -np.pi/2, 7.92089, 2027.84470]
initial_guess = [0.08, 2*np.pi/100, -0.5293, 7.920985, 2027.838470871639]


# Fit the sine function to the data
popt, pcov = curve_fit(sine_function, all_transit_num, all_obs_tc, p0=initial_guess, sigma=all_omc_err, absolute_sigma=True)
A_fit, omega_fit, phi_fit, period_fit, offset_fit = popt
print(f'fit parameters: {A_fit}, {omega_fit}, {phi_fit}, {period_fit}, {offset_fit}')
print(f'offset_fit: {offset_fit}')
print(f'period_fit: {period_fit}')

def ephem_fit(transit_num):
    return offset_fit + (period_fit * transit_num)
def ephem_guess(transit_num):
    return initial_guess[4] + (transit_num * initial_guess[3])

### for plotting
transit_num_plot = np.linspace(min(all_transit_num),max(all_transit_num),1000)
model_fit_sine = sine_function(transit_num_plot, *popt) - ephem_fit(transit_num_plot)

omc = all_obs_tc - ephem_fit(all_transit_num)

model_guess_sine = sine_function(transit_num_plot, *initial_guess) - ephem_fit(transit_num_plot)

residuals = (omc - sine_function(all_transit_num, *popt) + ephem_fit(all_transit_num)) / all_obs_tc_err # [days]/[days] -> unitless

omc_narita = np.array(omc[:10])
omc_err_narita = np.array(all_omc_err[:10])
residual_narita = np.array(residuals[:10])
omc_petigura = np.array(omc[10:16])
omc_err_petigura = np.array(all_omc_err[10:16])
residual_petigura = np.array(residuals[10:16])
omc_tess = np.array(omc[16:])
omc_err_tess = np.array(all_omc_err[16:])
residual_tess = np.array(residuals[16:])

### plot readability, plot in ttv hours
day2hr = 24

fig2, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},sharex=True)
ax1.plot(ephem_fit(transit_num_plot), model_guess_sine*day2hr, label='Guess sine curve')
ax1.plot(ephem_fit(transit_num_plot), model_fit_sine*day2hr, 'r-', label='Fitted sine curve')
ax1.errorbar(ephem_fit(narita_ind), omc_narita*day2hr, yerr=omc_err_narita*day2hr, fmt='ys',label='Narita', capsize=5)
ax1.errorbar(ephem_fit(petigura_ind_updated), omc_petigura*day2hr, yerr=omc_err_petigura*day2hr, fmt='bs',label='Petigura', capsize=5)
ax1.errorbar(ephem_fit(tess_ind), omc_tess*day2hr, yerr=omc_err_tess*day2hr, fmt='go',label='TESS', capsize=5)

ax1.set_title('TTV Fit: Planet b')
ax1.set_ylabel('TTV (hour)')
ax1.legend(loc='lower left')

ax2.plot(ephem_fit(narita_ind), residual_narita,'ys')
ax2.plot(ephem_fit(petigura_ind_updated), residual_petigura,'bs')
ax2.plot(ephem_fit(tess_ind), residual_tess,'go')
ax2.axhline(y=0, color='r', linestyle='-')
ax2.set_ylabel('Norm Residuals') #hour?? day?? unitless
ax2.set_xlabel('TC (BJD - 2454833 days)')
plt.show()

### ttv_fast plot
from ttvfast_run import times_1, omc_1, times_2, omc_2, start_time
plt.scatter(times_1,omc_1,color='orange',s=4,label='planet b ttvfast')
plt.scatter(times_2,omc_2,color='blue',s=4, label='planet c ttvfast')
plt.errorbar(tc_narita, omc_narita, xerr=tc_err_narita, yerr=omc_err_narita, fmt='ys', capsize=5, label='Narita')
plt.errorbar(tc_petigura, omc_petigura, xerr=tc_err_petigura, yerr=omc_err_petigura, fmt='rs', capsize=5, label='Petigura')
plt.errorbar(tc_tess, omc_tess, xerr=tc_err_tess, yerr=omc_err_tess, fmt='go', capsize=5, label='TESS')
plt.plot(ephem_fit(transit_num_plot), model_fit_sine,color='black', label='Fitted Sine')
plt.title(f"TTVfast and Measured Time: K2-19 (start time {start_time})")
plt.ylabel('TTV (days)')
plt.xlabel('TC (days)')
plt.legend()
plt.show()

### make txt file with results 
output_file = "ttv_results.txt"

# Combine all data into a structured format
source = ['Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 'Narita2015', 
          'Petigura2020', 'Petigura2020', 'Petigura2020', 'Petigura2020', 'Petigura2020', 'Petigura2020', 
          'TESS', 'TESS', 'TESS', 'TESS', 'TESS', 'TESS', 'TESS', 'TESS']

instrument = ['K2', 'K2', 'K2', 'K2', 'K2', 'K2', 'K2', 'K2', 'K2', 'K2', 
          'FLWO', 'TRAPPIST', 'MuSCAT', 'Spitzer', 'LCO', 'Spitzer', 
          'TESS', 'TESS', 'TESS', 'TESS', 'TESS', 'TESS', 'TESS', 'TESS']

# Ensure source and instrument are numpy arrays with dtype=object
source = np.array(source, dtype=object)
instrument = np.array(instrument, dtype=object)

data_to_save = np.column_stack((all_transit_num, all_obs_tc, all_obs_tc_err, all_omc, source,instrument))

# Define header
header = "Index Tc Tc_err OMC Source Instrument"

# Save to file
np.savetxt(output_file, data_to_save, fmt='%d %.8f %.8f %.8f %s %s', header=header, comments='')

print(f"Results saved to {output_file}")