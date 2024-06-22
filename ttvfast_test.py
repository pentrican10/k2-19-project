import ttvfast
from ttvfast import models
import numpy as np
import matplotlib.pyplot as plt
'''
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
N_step = (total-time) / dt
print(N_step)
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

'''
from ttvfast import models
import ttvfast
import matplotlib.pyplot as plt



gravity = 0.000295994511                        # AU^3/day^2/M_sun

'''
stellar_mass = 0.95573417954                    # M_sun


planet1 = models.Planet(
    mass=0.00002878248,                         # M_sun
    period=1.0917340278625494e+01,              # days
    eccentricity=5.6159310042858110e-02,
    inclination=9.0921164935951211e+01,         # degrees
    longnode=-1.1729336712101943e-18,           # degrees
    argument=1.8094838714599581e+02,            # degrees
    mean_anomaly=-8.7093652691581923e+01,       # degrees
)

planet2 = models.Planet(
    mass=0.00061895914,
    period=2.2266898036209028e+01,
    eccentricity=5.6691301931178648e-02,
    inclination=8.7598285693573246e+01,
    longnode=4.6220554014026838e-01,
    argument=1.6437004273382669e+00,
    mean_anomaly=-1.9584857031843157e+01,
)
'''

stellar_mass = 0.88                    # M_sun


planet1 = models.Planet(
    mass=(0.0777 * stellar_mass),               # M_sun
    period=7.9222,                              # days
    eccentricity=0.20,
    inclination=91.5,                           # degrees
    longnode= 0.,                   #fixed      # degrees
    argument=1.8094838714599581e+02,            # degrees
    mean_anomaly=-8.7093652691581923e+01,       # degrees
)

planet2 = models.Planet(
    mass=(0.0458 * stellar_mass),
    period=11.8993,
    eccentricity=0.21,
    inclination=91.1,
    longnode=-7.4, 
    argument=1.6437004273382669e+00,
    mean_anomaly=-1.9584857031843157e+01,
)

planets = [planet1, planet2]
Time = 2000     #-1045                        # days
dt = 0.1                                      # days
Total = 5500     #1700                        # days
N_step = int((Total-Time) / dt)
print(N_step)

results = ttvfast.ttvfast(planets, stellar_mass, Time, dt, Total)

# Extract necessary data from the results
planet_ind = np.array(results['positions'][0])
epoch_int = np.array(results['positions'][1])   #transit number
times_ = (np.array(results['positions'][2]))
rsky_values = np.array(results['positions'][3])
vsky_values = np.array(results['positions'][4])
print((times_[planet_ind==0]))
### finding index where the time values end and -2. fills rest of array
time_end = np.where(times_ == -2.)[0][0]
print(time_end)
print(results['positions'][2])

# ### Reverse the array
# reversed_array = planet_ind[::-1]

# ### Find the first occurrence of 1 in the reversed array
# first_occurrence_reversed =  np.where(reversed_array == 1)[0][0]

# ### Calculate the index in the original array
# original_index = len(planet_ind) - 1 - first_occurrence_reversed
# print(original_index)

planet_ind = planet_ind[:time_end]
times_= times_[:time_end]
print((results['positions'][0])[:time_end])

t_1 = times_[planet_ind==0]
t_1_aligned = t_1[1:]  # align with the np.diff calculation because it shortened the array (disregard first time)
ttv1 = np.diff(t_1) - np.mean(np.diff(t_1))
plt.scatter(t_1_aligned,ttv1, label='planet 1',color='orange')

t_2 = times_[planet_ind==1]
t_2_aligned = t_2[1:]
ttv2 = np.diff(t_2) - np.mean(np.diff(t_2))
plt.scatter(t_2_aligned, ttv2, label='planet 2', color='blue')
plt.ylabel('TTV (days)')
plt.xlabel('Time (days)')
plt.title('TTV plots')
plt.legend()
plt.show()