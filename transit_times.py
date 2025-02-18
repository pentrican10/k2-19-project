import pandas as pd
import numpy as np

# Read data into a Pandas DataFrame
def read_data_as_dataframe(filename):
    df = pd.read_csv(filename, delim_whitespace=True, skiprows=1, names=["Planet", "Transit", "Instrument", "Tc(days)", "σ(Tc)(days)", "Notes"])
    return df

# Example usage
filename = "planet_transits.txt"
df = read_data_as_dataframe(filename)

# time offset from BJD
petigura_offset = 2454833 # BJD - 2454833
tess_offset = 2457000 # BTJD - Barycentric TESS Julian Date (Julian Date - 2457000)

planet_b_data = df[df["Planet"] == "K2-19b"]
planet_c_data = df[df["Planet"] == "K2-19c"]


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