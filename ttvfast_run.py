import ttvfast
from ttvfast import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares



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

'''


gravity = 0.000295994511                        # AU^3/day^2/M_sun



'''
stellar_mass = 0.95573417954                    # M_sun


planet1 = models.Planet(
    mass=0.00002878248,                         # M_sun
    period=1.0917340278625494e+01,              # days
    eccentricity=5.6159310042858110e-02,
    inclination=9.0921164935951211e+01,         # degrees
    longnode=-1.1729336712101943e-18,           # degrees
    argument=180.94e+00,            # degrees
    mean_anomaly=-8.7093652691581923e+01,       # degrees
)

planet2 = models.Planet(
    mass=0.00061895914,
    period=2.2266898036209028e+01,
    eccentricity=5.6691301931178648e-02,
    inclination=8.7598285693573246e+01,
    longnode=4.6220554014026838e-01,
    argument=1.6437e+00,
    mean_anomaly=-1.9584857031843157e+01,
)

'''

#''' k2-19 param
stellar_mass = 0.88                    # M_sun
earth_mass = 5.972e+24 / 1.989e+30     # M_sun

### from paper
mass1 = (32.4 * earth_mass)     #M_sun
per1 = 7.9222                   #days
ecc1 = 0.20
i1 = 91.5                       #deg
Omega1 = 0.                     #longnode  deg
#argument1 = np.arccos((0.02)/(np.sqrt(ecc1))) #deg
argument1 = np.rad2deg(np.arctan2(-0.44,0.02)) #deg   sqrt(ecc)cos(argument)=0.02...
### from paper 
mass2 = (10.8 * earth_mass)     #M_sun
per2 = 11.8993                  #days
ecc2 = 0.21
i2 = 91.1                       #deg
Omega2 = -7.4                   #longnode deg
#argument2 = np.arccos((0.04)/(np.sqrt(ecc2))) #deg
argument2 = np.rad2deg(np.arctan2(-0.46,0.04)) #deg

### mean anomaly
def mean_anomaly(ecc, Omega, argument):
    ### omega is longitude of periastron (radians)  omega = Omega + argument = longnode + argument
    omega = Omega + argument
    f = 90 - omega
    ee = 2 * np.rad2deg(np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc))))  # eccentric anomaly
    m=ee - ecc*np.sin(ee)
    return m

M1 = mean_anomaly(ecc1, Omega1, argument1)
M2 = mean_anomaly(ecc2, Omega2, argument2)


planet1 = models.Planet(
    mass=mass1,                         # M_sun
    period=per1,                        # days
    eccentricity=ecc1,
    inclination=i1,                     # degrees
    longnode=Omega1,                    # degrees
    argument=argument1,                 # degrees
    mean_anomaly=M1,                    # degrees
)

planet2 = models.Planet(
    mass=mass2,
    period=per2,
    eccentricity=ecc2,
    inclination=i2,
    longnode=Omega2,
    argument=argument2,
    mean_anomaly=M2,
)


planets = [planet1, planet2]
### first transit time from paper (used TC given in paper)
# Time = 2020 #value of start time from paper                # days
start_time = 1980
Time = start_time
dt = 0.1                                      # days
Total = 5500                       # days
N_step = int((Total-Time) / dt)
#N_step = 1000
dt = (Total-Time) / N_step
### Results dictionary
results = ttvfast.ttvfast(planets, stellar_mass, Time, dt, Total)

### Extract necessary data from the results
planet_ind = np.array(results['positions'][0])
epoch_int = np.array(results['positions'][1])   #transit number
times_ = np.array(results['positions'][2])
rsky_values = np.array(results['positions'][3])
vsky_values = np.array(results['positions'][4])

### finding index where the time values end and -2. fills rest of array
time_end = np.where(times_ == -2.)[0][0]

### trim the arrays
planet_ind = planet_ind[:time_end]
epoch_int = epoch_int[:time_end]
times_= times_[:time_end]
# rsky_values = rsky_values[:time_end]
# vsky_values = vsky_values[:time_end]

### separate by planet 
times_1 = times_[planet_ind==0]
times_2 = times_[planet_ind==1]
epoch_1 = epoch_int[planet_ind==0]
epoch_2 = epoch_int[planet_ind==1]
# rsky_values_1 = rsky_values[planet_ind==0]
# rsky_values_2 = rsky_values[planet_ind==1]
# vsky_values_1 = vsky_values[planet_ind==0]
# vsky_values_2 = vsky_values[planet_ind==1]

### re-indexing about center data point
half_t1 = int(((len(times_1))/2))
half_t2 = int(((len(times_2))/2))
epoch_1_ind = epoch_1 - half_t1
epoch_2_ind = epoch_2 - half_t2

### linear regression planet1
X1 = epoch_1_ind
y1 = times_1

# Fit the model using numpy.polyfit
slope_1, intercept_1 = np.polyfit(X1, y1, 1)
# print(f'slope 1: {slope_1}')
# print(f'intercept 1: {intercept_1}')

# Predict the y values
y_pred1 = slope_1 * X1 + intercept_1

# Calculate residuals
omc_1 = y1 - y_pred1
# calc_times_1 = period_b_petigura*epoch_1_ind + tc_b_petigura
# omc_1 = times_1 - calc_times_1

### linear regression planet2
X2 = epoch_2_ind
y2 = times_2

# Fit the model using numpy.polyfit
slope_2, intercept_2 = np.polyfit(X2, y2, 1)
# print(f'slope 2: {slope_2}')
# print(f'intercept 2: {intercept_2}')

# Predict y values (expected times)
y_pred2 = slope_2 * X2 + intercept_2

# Calculate residuals (omc)
omc_2 = y2 - y_pred2
# calc_times_2 = period_b_petigura*epoch_2_ind + tc_b_petigura
# omc_2 = times_2 - calc_times_2

### plot 
# plt.scatter(times_1,omc_1,color='orange',label='planet b lin')
# plt.scatter(times_2,omc_2,color='blue', label='planet c lin')
# plt.title("Linear Regression TTVs - K2-19")
# plt.ylabel('TTV (days)')
# plt.xlabel('Time (days)')
# plt.legend()
# plt.show()




# ### using ephem to calculate
# period_sim1 = planet1.period
# period_sim2 = planet2.period
# expected_time_1 = np.zeros(len(times_1))
# expected_time_2 = np.zeros(len(times_2))
# # re-index and find intercept 
# for i in range(len(epoch_1_ind)):
#     expected_time1 =  period_sim1 * epoch_1_ind[i]
#     expected_time_1[i] = expected_time1
# for i in range(len(epoch_2_ind)):
#     expected_time2 =   period_sim2 * epoch_2_ind[i]
#     expected_time_2[i] = expected_time2

# omc_1 = times_1 - expected_time_1
# omc_2 = times_2 - expected_time_2


'''
def ttvfast_omc(planets):
    ### first transit time from paper (used TC given in paper)
    Time = 2218                 # days
    dt = 0.1                                      # days
    Total = 5500                       # days
    N_step = int((Total-Time) / dt)
    #N_step = 1000
    dt = (Total-Time) / N_step
    ### Results dictionary
    results = ttvfast.ttvfast(planets, stellar_mass, Time, dt, Total)

    ### Extract necessary data from the results
    planet_ind = np.array(results['positions'][0])
    epoch_int = np.array(results['positions'][1])   #transit number
    times_ = np.array(results['positions'][2])
    rsky_values = np.array(results['positions'][3])
    vsky_values = np.array(results['positions'][4])

    ### finding index where the time values end and -2. fills rest of array
    time_end = np.where(times_ == -2.)[0][0]


    ### trim the arrays
    planet_ind = planet_ind[:time_end]
    epoch_int = epoch_int[:time_end]
    times_= times_[:time_end]
    rsky_values = rsky_values[:time_end]
    vsky_values = vsky_values[:time_end]

    ### separate by planet 
    times_1 = times_[planet_ind==0]
    times_2 = times_[planet_ind==1]
    epoch_1 = epoch_int[planet_ind==0]
    epoch_2 = epoch_int[planet_ind==1]
    rsky_values_1 = rsky_values[planet_ind==0]
    rsky_values_2 = rsky_values[planet_ind==1]
    vsky_values_1 = vsky_values[planet_ind==0]
    vsky_values_2 = vsky_values[planet_ind==1]

    ### re-indexing about center data point
    half_t1 = int(((len(times_1))/2))
    half_t2 = int(((len(times_2))/2))
    epoch_1_ind = epoch_1 - half_t1
    epoch_2_ind = epoch_2 - half_t2



    ### linear regression planet1
    X1 = epoch_1_ind
    y1 = times_1


    # Fit the model using numpy.polyfit
    slope_1, intercept_1 = np.polyfit(X1, y1, 1)

    # Predict the y values
    y_pred1 = slope_1 * X1 + intercept_1

    # Calculate residuals
    omc_1 = y1 - y_pred1

    ### linear regression planet2
    X2 = epoch_2_ind
    y2 = times_2

    # Fit the model using numpy.polyfit
    slope_2, intercept_2 = np.polyfit(X2, y2, 1)

    # Predict y values (expected times)
    y_pred2 = slope_2 * X2 + intercept_2

    # Calculate residuals (omc)
    omc_2 = y2 - y_pred2

    return times_1, omc_1, times_2, omc_2
'''