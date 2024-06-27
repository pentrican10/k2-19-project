import ttvfast
from ttvfast import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

#'''
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
'''

planets = [planet1, planet2]
Time = -1045                        # days
dt = 0.1                                      # days
Total = 1700                        # days
N_step = int((Total-Time) / dt)

### Results dictionary
results = ttvfast.ttvfast(planets, stellar_mass, Time, dt, Total)

### Extract necessary data from the results
planet_ind = np.array(results['positions'][0])
epoch_int = np.array(results['positions'][1])   #transit number
times_ = (np.array(results['positions'][2]))
rsky_values = np.array(results['positions'][3])
vsky_values = np.array(results['positions'][4])

### finding index where the time values end and -2. fills rest of array
time_end = np.where(times_ == -2.)[0][0]


'''
# ### Reverse the array
# reversed_array = planet_ind[::-1]

# ### Find the first occurrence of 1 in the reversed array
# first_occurrence_reversed =  np.where(reversed_array == 1)[0][0]

# ### Calculate the index in the original array
# original_index = len(planet_ind) - 1 - first_occurrence_reversed
# print(original_index)
# time_end = original_index
'''
### trim the arrays
planet_ind = planet_ind[:time_end]
epoch_int = epoch_int[:time_end]
times_= times_[:time_end]
rsky_values = rsky_values[:time_end]
vsky_values = vsky_values[:time_end]

### separate by planet 
t_1 = times_[planet_ind==0]
t_2 = times_[planet_ind==1]
epoch_1 = epoch_int[planet_ind==0]
epoch_2 = epoch_int[planet_ind==1]
rsky_values_1 = rsky_values[planet_ind==0]
rsky_values_2 = rsky_values[planet_ind==1]
vsky_values_1 = vsky_values[planet_ind==0]
vsky_values_2 = vsky_values[planet_ind==1]

### re-indexing about center data point
half_t1 = int(((len(t_1))/2))
half_t2 = int(((len(t_2))/2))
epoch_1_ind = epoch_1 - half_t1
epoch_2_ind = epoch_2 - half_t2



### linear regression planet1
X1 = epoch_1_ind
y1 = t_1

# Fit the model using numpy.polyfit
slope_1, intercept_1 = np.polyfit(X1, y1, 1)

# Predict the y values
y_pred1 = slope_1 * X1 + intercept_1

# Calculate residuals
residuals1 = y1 - y_pred1


### linear regression planet2
X2 = epoch_2_ind
y2 = t_2

# Fit the model using numpy.polyfit
slope_2, intercept_2 = np.polyfit(X2, y2, 1)

# Predict y values (expected times)
y_pred2 = slope_2 * X2 + intercept_2

# Calculate residuals (omc)
residuals2 = y2 - y_pred2

### plot 
plt.scatter(t_1,residuals1,color='orange',label='planet 1 lin')
plt.scatter(t_2,residuals2,color='blue', label='planet 2 lin')
plt.title("Linear Regression and hacky TTVs")
plt.ylabel('TTV (days)')
plt.xlabel('Time (days)')
plt.legend()
plt.show()




### using ephem to calculate
period_sim1 = planet1.period
period_sim2 = planet2.period
expected_time_1 = np.zeros(len(t_1))
expected_time_2 = np.zeros(len(t_2))
# re-index and find intercept 
for i in range(len(epoch_1_ind)):
    expected_time1 =  period_sim1 * epoch_1_ind[i]
    expected_time_1[i] = expected_time1
for i in range(len(epoch_2_ind)):
    expected_time2 =   period_sim2 * epoch_2_ind[i]
    expected_time_2[i] = expected_time2
# for i in range(len(epoch_1_ind)):
#     expected_time1 =  intercept_1 + slope_1 * epoch_1_ind[i]
#     expected_time_1[i] = expected_time1
# for i in range(len(epoch_2_ind)):
#     expected_time2 =  intercept_2 + slope_2 * epoch_2_ind[i]
#     expected_time_2[i] = expected_time2

omc1 = t_1 - expected_time_1
omc2 = t_2 - expected_time_2
plt.plot(t_1,omc1, color='orange',label='planet 1 omc')
plt.plot(t_2,omc2, color='blue', label='planet 2 omc')
plt.title("Manual TTV Calculation")
plt.ylabel('TTV (days)')
plt.xlabel('Time (days)')
plt.legend()
plt.show()


### slopes and intercepts 
### simulation periods 
print(f'Simulation period 1: {period_sim1}')
print(f'Simulation period 2: {period_sim2}')

### linear regression periods
print(f'Linear regression Slope 1: {slope_1}')
print(f'Linear regression Intercept 1: {intercept_1}')
print(f'Linear regression Slope 2: {slope_2}')
print(f'Linear regression Intercept 2: {intercept_2}')


### pairwise periods (difference in consecutive times)
p_1 = np.zeros(len(t_1)-1)
for i in range(len(p_1)):
    p = t_1[i+1] - t_1[i]
    p_1[i] = p

p_2 = np.zeros(len(t_2)-1)
for i in range(len(p_2)):
    p = t_2[i+1] - t_2[i]
    p_2[i] = p

# Plot histogram 1
plt.hist(p_1, bins='auto', edgecolor='black')
plt.title('Histogram of p_1')
plt.xlabel('Period 1')
plt.ylabel('Frequency')
plt.show()

# Plot histogram 2
plt.hist(p_2, bins='auto', edgecolor='black')
plt.title('Histogram of p_2')
plt.xlabel('Period 2')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()