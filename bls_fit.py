import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from lightkurve import search_targetpixelfile
import pandas as pd
import os
import batman
from scipy.stats import norm
import matplotlib
# matplotlib.use("TkAgg")  # Use an interactive backend

# data_dir = "C:\\Users\\Paige\\Projects\\data\\k2-19_data"
# data_dir = "/mnt/c/Users/Paige/Projects/data/k2-19_data"
data_dir = "/Users/pentrican10/Projects/Data/k2-19-data"


### path to table - Petigura et al 2020
file_path = os.path.join(data_dir, "ajab5220t1_mrt.txt")

### Define the column names 
columns = ["Planet", "Transit", "Inst", "Tc", "e_Tc", "Source"]

### Read the text file, specifying space as the delimiter, skipping initial rows
df = pd.read_csv(file_path, delim_whitespace=True, skiprows=22, names=columns)

### Remove NaN values
df = df.dropna()

# Download the light curve data
lc = lk.search_lightcurve("K2-19",author = 'SPOC').download_all()

# Flatten the light curve
lc = lc.stitch().flatten(window_length=901).remove_outliers()
#lc.plot()
time = lc.time
flux=lc.flux
flux_err = lc.flux_err


# Create array of periods to search
period = np.linspace(1, 20, 10000)
# Create a BLSPeriodogram
bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500)
#bls.plot()

planet_b_period = bls.period_at_max_power
planet_b_t0 = bls.transit_time_at_max_power
planet_b_dur = bls.duration_at_max_power


'''
# Check the value for period
print(f'Period b: {planet_b_period}')
print(f'tc b: {planet_b_t0}')
print(f'duration b: {planet_b_dur}')

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
print(planet_c_t0)
print(planet_c_dur)

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
'''
