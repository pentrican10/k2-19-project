import pandas as pd
import numpy as np
from tess_transits import tess_transit_data
from k2_19_project import planet_b_period, planet_b_t0
period_b_bls = planet_b_period.value
tc_b_bls = planet_b_t0.value

# Read data into a Pandas DataFrame
def read_data_as_dataframe(filename):
    df = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=["Planet", "Transit", "Instrument", "Tc(days)", "σ(Tc)(days)", "Notes"])
    return df

filename = "planet_transits.txt"
df = read_data_as_dataframe(filename)

# time offset from BJD
petigura_offset = 2454833 # BJD - 2454833
tess_offset = 2457000 # BTJD - Barycentric TESS Julian Date (Julian Date - 2457000)

planet_b_data = df[df["Planet"] == "K2-19b"]

# Print the DataFrame
print(df["Tc(days)"])
print(planet_b_data["Tc(days)"])
print(np.array(planet_b_data["Transit"]))

t_num = np.array(planet_b_data["Transit"]) 

period_b_petigura = 7.9222 # [days]
tc_b_petigura = 2027.9023 # [days] using petigura offset

tc_petigura = np.array(planet_b_data["Tc(days)"])
print(f"Measured transit times (Petigura): {tc_petigura}")

tc_guess=[]
for i in range(len(t_num)):
    t = tc_b_petigura + (t_num[i]*period_b_petigura)
    tc_guess.append(t)
print(f"Guess times (Petigura ephem): {tc_guess}")

print(f"Difference in measured and guess times based on Petigura ephem: {tc_guess - tc_petigura}")

tc_initial = []
for i in range(len(tc_petigura)):
    tc = tc_petigura[i] - (t_num[i]*period_b_petigura)
    tc_initial.append(tc)

print(f"Inferred tc from Petigura et al: {tc_initial}")

updated_indices = np.array(planet_b_data["Transit"]) - 6
print(f"Updated transit indices: {updated_indices}")



### tess times in the paper ephem
tnum_tess = [337,339,340,341,342,343,430,432]
### updated indices for petigura ephem
tess_ind = np.array(tess_transit_data["Transit"]) + 337

all_transit_num = []
for i in range(len(updated_indices)):
    all_transit_num.append(updated_indices[i])
for i in range(len(tess_ind)):
    all_transit_num.append(tess_ind[i])

print(f"All transit indices: {all_transit_num}")

all_obs_tc = []

