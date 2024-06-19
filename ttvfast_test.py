import ttvfast
from ttvfast import models
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the planets using models.Planet
planet1 = models.Planet(
    mass=(32.4 * 3.0027e-6),  # M_sun
    period=7.9222,            # days
    eccentricity=0.2,
    inclination=91.5,         # degrees
    longnode=-1.1729336712101943e-18,  # degrees
    argument=180.94838714599581,       # degrees
    mean_anomaly=-87.093652691581923,  # degrees
)

planet2 = models.Planet(
    mass=(10.8 * 3.0027e-6),  # M_sun
    period=11.8993,           # days
    eccentricity=0.21,
    inclination=91.1,         # degrees
    longnode=46.220554014026838,  # degrees
    argument=164.37004273382669,  # degrees
    mean_anomaly=-19.584857031843157,  # degrees
)

# The central star's mass in solar masses
star_mass = 1.0

# Define the additional required arguments
time = 2000.0  # Start time in days
dt = 0.1       # Time step in days
total = 3500.0  # Total integration time in days

# Pack the planetary parameters into a list
planets = [planet1, planet2]

# Run the TTVFast simulation
results = ttvfast.ttvfast(planets, star_mass, time, dt, total)

# Extract necessary data from the results
planet_ind = (results['positions'][0])
epoch_int = (results['positions'][1])
times_ = (results['positions'][2])
rsky_values = np.array(results['positions'][3])
vsky_values = np.array(results['positions'][4])

print(times_)
assert 1==0
# Separate transit times for each planet
actual_transit_times_1 = times_[planet_ind == 0]
epoch_1 = epoch_int[planet_ind == 0]
actual_transit_times_2 = times_[planet_ind == 1]
epoch_2 = epoch_int[planet_ind == 1]
# Debug: Check the lengths of actual transit times
print(f"actual_transit_times_1 length: {len(actual_transit_times_1)}")
print(f"actual_transit_times_2 length: {len(actual_transit_times_2)}")

# Calculate expected transit times assuming a constant orbital period
num_transits_1 = len(actual_transit_times_1)
num_transits_2 = len(actual_transit_times_2)
expected_transit_times_1 = np.zeros(len(epoch_1))
expected_transit_times_2 = np.zeros(len(epoch_2))
for i in range(len(epoch_1)):
    expected_transit_times = epoch_1[i] + planet1.period * i
    expected_transit_times_1[i] = expected_transit_times
for i in range(len(epoch_2)):
    expected_transit_times = epoch_2[i] + planet2.period * i
    expected_transit_times_2[i] = expected_transit_times

# Calculate TTVs (actual - expected)
ttvs_1 = actual_transit_times_1 - expected_transit_times_1
ttvs_2 = actual_transit_times_2 - expected_transit_times_2

# Convert TTVs from days to minutes for better readability
ttvs_1_minutes = ttvs_1 * 24 * 60
ttvs_2_minutes = ttvs_2 * 24 * 60

# Plot the TTVs
plt.figure(figsize=(10, 6))
plt.plot(actual_transit_times_1, ttvs_1_minutes, 'o-', label='Planet 1')
plt.plot(actual_transit_times_2, ttvs_2_minutes, 'o-', label='Planet 2')
plt.xlabel('Time (days)')
plt.ylabel('TTV (minutes)')
plt.title('Transit Timing Variations')
plt.legend()
plt.show()

'''
from ttvfast import models
import ttvfast
import matplotlib.pyplot as plt


gravity = 0.000295994511                        # AU^3/day^2/M_sun
stellar_mass = 0.88                    # M_sun


planet1 = models.Planet(
    mass=(32.4 * 3.0027e-6),                         # M_sun
    period=7.9222,              # days
    eccentricity=5.6159310042858110e-02,
    inclination=91.5,         # degrees
    longnode=-1.1729336712101943e-18,           # degrees
    argument=1.8094838714599581e+02,            # degrees
    mean_anomaly=-8.7093652691581923e+01,       # degrees
)

planet2 = models.Planet(
    mass=(10.8 * 3.0027e-6),
    period=11.8993,
    eccentricity=5.6691301931178648e-02,
    inclination=91.1,
    longnode=4.6220554014026838e-01,
    argument=1.6437004273382669e+00,
    mean_anomaly=-1.9584857031843157e+01,
)

planets = [planet1, planet2]
Time = -1045                                    # days
dt = 0.54                                       # days
Total = 1700                                    # days

results = ttvfast.ttvfast(planets, stellar_mass, Time, dt, Total)
print(results)
plt.plot(results['positions'])
plt.show()
'''