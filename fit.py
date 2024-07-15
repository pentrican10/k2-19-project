import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from lightkurve import search_targetpixelfile
import pandas as pd
import os
import batman
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
#import ttvfast_test


### switch to mask out transits
mask_transits = True

data_dir = "C:\\Users\\Paige\\Projects\\data\\k2-19_data"


file = "ajab5220t1_mrt.txt"
def read_table(file_name):
    ### path to table - Petigura et al 2020
    file_path = os.path.join(data_dir, "ajab5220t1_mrt.txt")

    ### Define the column names 
    columns = ["Planet", "Transit", "Inst", "Tc", "e_Tc", "Source"]

    ### Read the text file, specifying space as the delimiter, skipping initial rows
    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=22, names=columns)

    ### Remove NaN values
    df = df.dropna()
    return df

def convert_time(times):
    ### TESS offset 
    BTJD = times + 2457000
    new_time = BTJD - 2454833
    return new_time

def omc(obs_time, t_num, p, tc):
    calc_time = tc + (t_num* p)
    omc = obs_time - calc_time
    return omc#*24 #days to hours

### Find the intersection points
def intersection_func(t): #masked
    return np.interp(t, tc1, chi_sq) - err_threshold
    

df = read_table(file)

### params from exoplanet archive
per_b = 7.9222
rp_b = 0.0777
T14_b = 3.237 * 0.0416667  # convert to days
b_b = 0.17
q1_b = 0.4
q2_b = 0.3

per_c = 11.8993
rp_c = 0.0458
T14_c = 3.823 * 0.0416667
b_c = 0.630
q1_c = 0.4
q2_c = 0.3  

#########################################################################################################################################

### generate ttv (lin ephem from params in table 3)
p_b = 7.9222 #7.920925490169578   ### used linear regression, changed the slope to the one extracted original paper value 7.9222
tc_b = 2027.9023 #2027.9158659031389 ###2027.9023
# Perform linear regression using numpy.polyfit
# transit_indices = np.array([24, 28, 35, 127, 135, 144, 337, 339, 340, 341, 342, 343, 430, 432])
# transit_t = np.array([2218.0041, 2249.6955, 2305.1505, 3033.8604, 3097.2502, 3168.5368, 4697.28890755,
#                        4713.12745394, 4721.04173821, 4728.95982641, 4736.88095462, 4744.79876123, 5433.87888953, 5449.72028567])
# coefficients, cov_matrix = np.polyfit(transit_indices, transit_t, 1, cov=True)

# # Extract slope and intercept
# slope = coefficients[0]
# intercept = coefficients[1]

# # Calculate standard errors
# slope_error = np.sqrt(cov_matrix[0, 0])
# intercept_error = np.sqrt(cov_matrix[1, 1])
# print(f"Slope: {slope} ± {slope_error}")
# print(f"Intercept: {intercept} ± {intercept_error}")
# p_b = slope
# err_p_b = slope_error
# tc_b = intercept
# err_tc_b = intercept_error

print(f'Period(paper) b: {p_b}')
print(f'TC(paper) b: {tc_b}')
transit_b = [24,28,35,127,135,144]
predicted_time = []
for i in transit_b:
    ephem = tc_b + i* p_b
    predicted_time.append(ephem)
print(f'Predicted times(ephem) b: {predicted_time}')
#print(f'Predicted times error(ephem) b: {err_predicted_time}')
pl_b = df[df["Planet"] == "K2-19b"]
tc_paper = pl_b.Tc.values
err_tc_paper = pl_b.e_Tc.values
paper_ttv_b = tc_paper - predicted_time
err_paper_ttv_b = err_tc_paper

print(f'TC(paper) b: {tc_paper}')
print(f'TC err(paper) b: {err_tc_paper}')
print(f'TTV from ephem b: {paper_ttv_b}')
print(f'TTV error from ephem b: {err_paper_ttv_b}')

### from BLS in k2-19_project.py
### since this is from bls, not sure how to get errors
planet_b_period = 7.9204920492049204
planet_b_t0 = 2530.2807708159753
print(f'Period(BLS): {planet_b_period}')
print(f'Tc(BLS): {planet_b_t0}')

### planet c
p_c = 11.8993
tc_c = 2020.0007
transit_c = [84, 99]
predicted_time_c=[]
for i in transit_c:
    ephem = tc_c + i* p_c
    predicted_time_c.append(ephem)
print(f'Predicted times(ephem) c: {predicted_time_c}')

pl_c = df[df["Planet"] == "K2-19c"]
paper_ttv_c = pl_c.Tc.values - predicted_time_c
print(f'TC(paper) c: {pl_c.Tc.values}')
print(f'TTV from ephem c: {paper_ttv_c}')

##########################################################################################################################################

### Download the light curve data
lc = lk.search_lightcurve("K2-19",author = 'SPOC').download_all()
lc = lc.stitch()
if mask_transits == True:
    ### mask transit times before flattening
    #transit_times = [4697.28834658, 4713.12428017, 4721.03972171, 4728.96021376, 4736.88070581, 4744.79614735, 5433.87895563, 5449.71993973]
    transit_times = [4697.28834658, 4713.12933068, 4721.03972171, 4728.96021376, 4736.88070581, 4744.80119786, 5433.87895563, 5449.71993973]
    masked_lc = lc

    times = convert_time(masked_lc.time.value)

    ### Initialize a mask with all False values (i.e., include all data points initially)
    mask = np.zeros_like(times, dtype=bool)

    ### Iterate through each transit time and update the mask
    for transit_time in transit_times:
        mask |= (times > (transit_time - T14_b/2)) & (times < (transit_time + T14_b/2))

    ### Flatten the masked light curve
    masked_lc = masked_lc.flatten(window_length=901, mask=mask).remove_outliers()
    lc = masked_lc
else:
    ### flatten unmasked lightcurve 
    lc = lc.flatten(window_length=901).remove_outliers()


### initialize guess times
transit_num = [0,2,3,4,5,6,93,95]
### transit num with paper ephem
t_num_paper = [337,339,340,341,342,343,430,432]
t_num_paper_c = []


tc_guess=[]
for num in transit_num:
    t = planet_b_t0 + (num * planet_b_period)
    tc_guess.append(t)



### transit num with paper ephem
t_num_paper = [337,339,340,341,342,343,430,432]
t_num_paper_c = []

### data from lightcurve 
time_tess = np.array(lc.time.value)
flux=np.array(lc.flux)
flux_err = np.array(lc.flux_err)

time = convert_time(time_tess)
tc_guess = convert_time(np.array(tc_guess))
tc_guess = np.array(tc_guess)
print(f'TC guess(TESS): {tc_guess}')



### set range for search: [#hours] * [days per hour]
ttv_hour = 2* 0.0416667 # 1 hour to days
#tc_guess = (2530.28, 2546.12, 2554.04, 2561.96, 2569.88, 2577.8, 3266.84, 3282.68)

### get tc ranges 
tc = []
for i in range(len(tc_guess)):
    start = tc_guess[i] - ttv_hour
    end = tc_guess[i] + ttv_hour
    t = np.linspace(start,end, 1000)
    tc.append(t)


tc_chi = np.zeros(len(tc))
tc_chi_parabola = np.zeros(len(tc))
ttv = np.zeros(len(tc))
ttv_p = np.zeros(len(tc))
errors = []
errors_p = []
### plot X^2 vs tc for each guess
for j in range(len(tc)):
    tc1 = tc[j]
    chi_sq = np.zeros(len(tc1))
    chi_sq_lc = np.zeros(len(tc1))
    for i in range(len(tc1)):
        t0_b = 	tc1[i]
        theta_initial = [t0_b, per_b, rp_b, b_b, T14_b, q1_b, q2_b]
        ### initialize params
        params = batman.TransitParams()
        params.t0, params.per, params.rp,params.b, params.T14, q1, q2 = theta_initial
        params.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]  # Limb darkening coefficients
        params.limb_dark = 'quadratic'
        
        ### mask data - extract relevant photometry
        start = tc_guess[j] - ttv_hour
        end = tc_guess[j] + ttv_hour
        mask = (time > (start)) & (time < (end))
        
        transit_model = batman.TransitModel(params, time[mask])
            
        # Generate model light curve
        model_flux = transit_model.light_curve(params)
        
        # Calculate chi-squared value
        sigma2 = flux_err[mask] 
        chi_squared = np.sum(((flux[mask] - model_flux) / sigma2)**2)
        chi_sq[i] = (chi_squared)



    ### masked
    min_chi_time = tc1[np.argmin(chi_sq)]
    min_chi = chi_sq.min()

    tc_chi[j] = min_chi_time
    idx = t_num_paper[j]
    ttv[j] = omc(min_chi_time, idx, p_b, tc_b)#*24 #days to hours

    chi_mask = (chi_sq <= min_chi + 3)
    fit_mask = (chi_sq <= min_chi + 1)

    ### fit parabola to the chisq
    p_chi_sq = np.polyfit(tc1[fit_mask], chi_sq[fit_mask], 2)  

    ### Extract the coefficients   y = ax^2 + bx + c
    a_chi_sq, b_chi_sq, c_chi_sq = p_chi_sq
    
    ### Find the minimum of the parabola xmin = -b/2a from taking derivative=0
    tc_best_fit = -b_chi_sq / (2 * a_chi_sq)
    
    ### Calculate the minimum chi-squared value
    chi_sq_min = a_chi_sq * tc_best_fit**2 + b_chi_sq * tc_best_fit + c_chi_sq
    tc_chi_parabola[j] = tc_best_fit

    ### Calculate the parabola best fit 
    p_1 = a_chi_sq*tc1**2 + b_chi_sq*tc1 + c_chi_sq

    ### calculate ttv from parabola fit 
    ttv_p[j] = omc(tc_best_fit, idx, p_b, tc_b)#*24 #days to hours
    

    
  

    ### delta chisq = 1 gives errors
    err_threshold = min_chi + 1 # using chisq discrete minimum
    err_threshold_p = chi_sq_min + 1 # using minimum of parabola
  
    # Find the intersection using root_scalar
    intersections = []
    for k in range(len(tc1) - 1):
        if (chi_sq[k] - err_threshold) * (chi_sq[k + 1] - err_threshold) < 0:
            sol = root_scalar(intersection_func, bracket=[tc1[k], tc1[k + 1]])
            if sol.converged:
                intersections.append((sol.root - min_chi_time))
    errors.append(intersections)

    intersections_p = []
    for k in range(len(tc1) - 1):
        if (p_1[k] - err_threshold_p) * (p_1[k + 1] - err_threshold_p) < 0:
            sol = root_scalar(intersection_func, bracket=[tc1[k], tc1[k + 1]])
            if sol.converged:
                intersections_p.append((sol.root - tc_best_fit))
    errors_p.append(intersections_p)

  
    # plt.plot(tc1[chi_mask], chi_sq[chi_mask],label='chisq')
    # plt.plot(tc1[chi_mask], p_1[chi_mask],label='chisq parabola', color='orange')
    # plt.axvline(x=tc_guess[j], color='r', linestyle='--', label='Bls Guess')
    # plt.axvline(x=min_chi_time, linestyle='--', label='Chisq min')
    # plt.axvline(x=tc1[np.argmin(p_1)], color='orange', linestyle='--', label='Chisq min parabola')

    # # for inter in intersections:
    # #     plt.axvline(x=inter, color='blue', linestyle='--')
    # plt.axhline(y=err_threshold, color='green', linestyle='--', label='Error Threshold')
    # plt.title(f'Transit {j+1}: Planet b')
    # plt.xlabel('tc')
    # plt.ylabel('X^2')
    # plt.legend()
    # plt.show()


#avg the errors   sig^2 = 0.5(sig1^2 + sig2^2)
err_tc_chi = []
for i in range(len(errors)):
    sig = np.sqrt(errors[i][0]**2 + errors[i][1]**2)
    err_tc_chi.append(sig)

err_tc_chi_p = []
for i in range(len(errors_p)):
    sig = np.sqrt(errors_p[i][0]**2 + errors_p[i][1]**2)
    err_tc_chi_p.append(sig)

### print values 
if mask_transits == True:
    print(f'Transit Times(TESS) Masked: {tc_chi}')
    print(f'Avg Errors Masked: {err_tc_chi}')
    print(f'TTV(TESS) Masked: {ttv}')

    print(f'Transit Times(TESS parabola) Masked: {tc_chi_parabola}')
    print(f'Avg Errors (parabola) Masked: {err_tc_chi_p}')
    print(f'TTV(TESS parabola) Masked: {ttv_p}')
else:
    print(f'Transit Times(TESS) Un-Masked: {tc_chi}')
    print(f'Avg Errors Un-Masked: {err_tc_chi}')
    print(f'TTV(TESS) Un-Masked: {ttv}')

    print(f'Transit Times(TESS parabola) Un-Masked: {tc_chi_parabola}')
    print(f'Avg Errors (parabola) Un-Masked: {err_tc_chi_p}')
    print(f'TTV(TESS parabola) Un-Masked: {ttv_p}')

# Define a sine function
def sine_function(t, A, omega, phi, p, offset):
    return A * np.sin(omega * t + phi) + (p * t) + offset

# Concatenate all data points for fitting
all_times = np.concatenate([tc_paper, tc_chi_parabola])
all_ttv = np.concatenate([ paper_ttv_b, ttv_p])
all_err_ttv = np.concatenate([err_paper_ttv_b, err_tc_chi_p])
all_transit_num = np.concatenate([transit_b, t_num_paper])

# Initial guess for the parameters: amplitude, frequency, phase, offset
initial_guess = [0.01, 2*np.pi/365, 0, 0, 0]

# Fit the sine function to the data
#popt, pcov = curve_fit(sine_function, all_times, all_ttv, p0=initial_guess, sigma=all_err_ttv, absolute_sigma=True)
popt, pcov = curve_fit(sine_function, all_times, all_ttv, p0=initial_guess, sigma=all_err_ttv, absolute_sigma=True)

# Extract the fitted parameters
# add linear slope param +ct to get slope
A_fit, omega_fit, phi_fit, p, offset_fit = popt
print(f'offset: {offset_fit}')
print(f'p: {p}')

### Calculate the fitted TTV values
fitted_ttv = sine_function(all_times, *popt) 
t_fit = np.linspace(min(all_times), max(all_times), 1000)
### Create a figure with two subplots (one above the other)
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},sharex=True)
normalized_sine = sine_function(t_fit, *popt) - (p*t_fit) - offset_fit
norm_chi_ttv = ttv_p - (p*tc_chi_parabola) - offset_fit
norm_paper_ttv = paper_ttv_b - (p*tc_paper) - offset_fit

ax1.plot(t_fit, normalized_sine, 'r-', label='Fitted sine curve')
ax1.errorbar(tc_chi_parabola, norm_chi_ttv, xerr=err_tc_chi_p, yerr=err_tc_chi_p, fmt='bo', capsize=5, label='TESS times')
ax1.errorbar(tc_paper, norm_paper_ttv, xerr=err_tc_paper, yerr=err_paper_ttv_b, fmt='bs', capsize=5, label='Paper times')
ax1.set_title('TTV (chisq parabola): Planet b')
ax1.set_ylabel('TTV value (days)')
ax1.legend()

# Plot the residuals in the second subplot
residuals = all_ttv - fitted_ttv
residuals_paper = (paper_ttv_b - sine_function(tc_paper, *popt)) / err_paper_ttv_b
residuals_TESS = (ttv_p - sine_function(tc_chi_parabola, *popt)) / err_tc_chi_p
ax2.plot(tc_paper, residuals_paper, 'bs')
ax2.plot(tc_chi_parabola, residuals_TESS, 'bo')
ax2.axhline(y=0, color='r', linestyle='-')
ax2.set_xlabel('TC (days)')
ax2.set_ylabel('Norm Residuals (days)')
residual_range = max(abs(residuals_TESS))
ax2.set_ylim(-1.1 * residual_range, 1.1 * residual_range)

plt.tight_layout()
plt.show()

from ttvfast_test import t_1, residuals1, t_2, residuals2
#fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},sharex=True)
plt.scatter(t_1,residuals1,color='orange',s=4,label='planet b ttvfast')
plt.scatter(t_2,residuals2,color='blue',s=4, label='planet c ttvfast')
plt.errorbar(tc_chi_parabola, norm_chi_ttv, xerr=err_tc_chi_p, yerr=err_tc_chi_p, fmt='ro', capsize=5, label='TESS times b')
plt.errorbar(tc_paper, norm_paper_ttv, xerr=err_tc_paper, yerr=err_paper_ttv_b, fmt='rs', capsize=5, label='Paper times b')
plt.plot(t_fit, normalized_sine,color='black', label='Fitted sine curve b')
plt.title("TTVfast and Measured Times - K2-19")
plt.ylabel('TTV (days)')
plt.legend()
plt.show()


### method 2
# Initial guess for the parameters: amplitude, frequency, phase, offset
initial_guess2 = [0.01, 2*np.pi/365, 0, 7.9222, 2027.9023]

# Fit the sine function to the data
popt2, pcov2 = curve_fit(sine_function, all_transit_num, all_times, p0=initial_guess2, sigma=all_err_ttv, absolute_sigma=True)
A_fit2, omega_fit2, phi_fit2, p2, offset_fit2 = popt2
print(f'offset2: {offset_fit2}')
print(f'p2: {p2}')

transit_b = np.array(transit_b,dtype='float64')
t_num_paper = np.array(t_num_paper,dtype='float64')

### Calculate the fitted TTV values
# fitted_ttv = sine_function(all_times, *popt2) 
t_fit2 = np.linspace(min(all_transit_num), max(all_transit_num), 1000)
### Create a figure with two subplots (one above the other)
fig2, (ax3, ax4) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},sharex=True)
normalized_sine2 = sine_function(t_fit2, *popt2) - (p2*t_fit2) - offset_fit2
norm_tc_chi_parabola = tc_chi_parabola - (p2*t_num_paper) - offset_fit2
norm_tc_paper = tc_paper - (p2*transit_b) - offset_fit2


ax3.plot(t_fit2,normalized_sine2, 'r-', label='Fitted sine curve')
ax3.errorbar(t_num_paper, norm_tc_chi_parabola, yerr=err_tc_chi_p, fmt='bo', capsize=5, label='TESS times')
ax3.errorbar(transit_b, norm_tc_paper, yerr=err_tc_paper, fmt='bs', capsize=5, label='Paper times')
ax3.set_title('Times: Planet b')
ax3.set_ylabel('times (days)')
ax3.set_xlabel('transit number')
ax3.legend()

transit_b = np.asarray(transit_b, dtype='float64')
residual_paper = tc_paper - sine_function(transit_b, *popt2)
residual_tess = tc_chi_parabola - sine_function(t_num_paper, *popt2)
ax4.plot(transit_b, residual_paper, 'bs')
ax4.plot(t_num_paper, residual_tess, 'bo')
ax4.axhline(y=0, color='r', linestyle='-')
ax4.set_ylabel('Residuals (days)')


plt.show()
#######################################################################################################################

### Function to generate the transit model
def transit_model(theta, time):
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.b, params.T14, q1, q2 = theta
    params.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]  # Limb darkening coefficients
    params.limb_dark = 'quadratic'
    
    model = batman.TransitModel(params, time)
    return model.light_curve(params)

### Residuals function for least_squares
def residuals(theta, time, flux, flux_err):
    model_flux = transit_model(theta, time)
    return (flux - model_flux) / flux_err

### Initialize arrays to store results
optimal_params_list = []
errors_list = []

# Loop over each tc_guess
for i in range(len(tc_guess)):
    ### start lstsq with the chisq guess from parabola
    t0_b = tc_chi_parabola[i]
    ### Initial guess for parameters
    theta_initial = [t0_b, per_b, rp_b, b_b, T14_b, q1_b, q2_b]
    
    ### Mask data - extract relevant photometry
    start = t0_b - ttv_hour
    end = t0_b + ttv_hour
    mask = (time > start) & (time < end)
    
    ### Use least_squares to find optimal parameters
    result = least_squares(residuals, theta_initial, args=(time[mask], flux[mask], flux_err[mask]))
    
    ### Extract optimal parameters
    optimal_params = result.x
    optimal_params_list.append(optimal_params)

    ### calculate errors
    # Calculate the covariance matrix from the Jacobian
    J = result.jac
    cov = np.linalg.pinv(J.T @ J)   
    
    # Calculate standard errors
    errors = np.sqrt(np.diag(cov))
    errors_list.append(errors)


### Output results
tc_lstsq = []
err_tc_lstsq = []
for params, errors in zip(optimal_params_list, errors_list):
    tc_lstsq.append(params[0])
    err_tc_lstsq.append(errors[0])

ttv_lstsq = np.zeros(len(tc))
for j in range(len(tc)):
    ttv_lstsq[j] = omc(tc_lstsq[j], t_num_paper[j], p_b, tc_b)

# Concatenate all data points for fitting
all_times = np.concatenate([tc_lstsq, tc_paper])
all_ttv = np.concatenate([ttv_lstsq, paper_ttv_b])
all_err_ttv = np.concatenate([err_tc_lstsq, err_paper_ttv_b])

# Initial guess for the parameters: amplitude, frequency, phase, offset
initial_guess = [0.01, 2*np.pi/365, 0,0, 0]

# Fit the sine function to the data
popt, pcov = curve_fit(sine_function, all_times, all_ttv, p0=initial_guess, sigma=all_err_ttv, absolute_sigma=True)

# Extract the fitted parameters
A_fit, omega_fit, phi_fit, p, offset_fit = popt

# Calculate the fitted TTV values
fitted_ttv = sine_function(all_times, *popt)
# Plot the fitted sine curve
t_fit = np.linspace(min(all_times), max(all_times), 1000)
plt.plot(t_fit, sine_function(t_fit, *popt), 'r-', label='Fitted sine curve')


### plot ttvs
plt.errorbar(tc_lstsq, ttv_lstsq,xerr=err_tc_lstsq,yerr = err_tc_lstsq, fmt='o', capsize=5, label='TESS times')
plt.errorbar(tc_paper, paper_ttv_b,xerr=err_tc_paper, yerr = err_paper_ttv_b, fmt='o', capsize=5, label='paper times')
#plt.scatter(pl_c.Tc.values, paper_ttv_c)
plt.title(f'TTV (lstsq): Planet b')
plt.xlabel('TC (days)')
plt.ylabel('TTV value (days)')
plt.legend()
#plt.show()

### Print the rounded values
if mask_transits == True:
    print(f'Transit Times(Least Sq) Masked: {[round(val, 4) for val in tc_lstsq]}')
    print(f'Errors (Least Sq) Masked: {[round(val, 4) for val in err_tc_lstsq]}')
    print(f'Transit Times(TESS Chi sq parabola) Masked: {[round(val, 4) for val in tc_chi_parabola]}')
    print(f'Avg Errors (chi sq parabola) Masked: {[round(val, 4) for val in err_tc_chi_p]}')
    print(f'Transit Times(TESS Chi sq) Masked: {[round(val, 4) for val in tc_chi]}')
    print(f'Avg Errors (chi sq) Masked: {[round(val, 4) for val in err_tc_chi]}')
    print(f'TC guess(TESS): {[round(val, 4) for val in tc_guess]}')
else:
    print(f'Transit Times(Least Sq) Un-Masked: {[round(val, 4) for val in tc_lstsq]}')
    print(f'Errors (Least Sq) Un-Masked: {[round(val, 4) for val in err_tc_lstsq]}')
    print(f'Transit Times(TESS Chi sq parabola) Un-Masked: {[round(val, 4) for val in tc_chi_parabola]}')
    print(f'Avg Errors (chi sq parabola) Un-Masked: {[round(val, 4) for val in err_tc_chi_p]}')
    print(f'Transit Times(TESS Chi sq) Un-Masked: {[round(val, 4) for val in tc_chi]}')
    print(f'Avg Errors (chi sq) Un-Masked: {[round(val, 4) for val in err_tc_chi]}')
    print(f'TC guess(TESS): {[round(val, 4) for val in tc_guess]}')

transit_index = range(len(tc_lstsq))

### Loop through each transit index to create individual plots
# for i in transit_index:
#     ### Plot Chi Square with error bars
#     plt.errorbar(i+1, tc_chi[i], yerr=err_tc_chi[i], fmt='s', label='Chi Square', capsize=5)

#     ### Plot chi sq parabola with error bars
#     plt.errorbar(i+1, tc_chi_parabola[i], yerr=err_tc_chi_p[i], fmt='+', label='Chi Square Parabola', capsize=5)

#     ### Plot Least Squares with error bars
#     plt.errorbar(i+1, tc_lstsq[i], yerr=err_tc_lstsq[i], fmt='o', label='Least Squares', capsize=5)

#     ### Plot TC Guess
#     plt.plot(i+1, tc_guess[i], 'x', label='TC Guess')

#     plt.xlabel('Transit num')
#     plt.ylabel('Transit Times')
#     plt.title(f'Transit Times with Error Bars - Transit {i+1}')
#     plt.legend()
#     plt.show()


####################################################################################################################
'''
### Photometry
tc_test = tc_chi
for i in range(len(tc_test)):
    start = tc_test[i] - (24* 0.0416667)
    end = tc_test[i] + (24* 0.0416667)
    # start = tc_test[1] - (24* 0.0416667)
    # end = tc_test[1] + (24* 0.0416667)
    mask = (time > (start)) & (time < (end))

    theta_initial = [tc_test[i], p_b, rp_b, b_b, T14_b, q1_b, q2_b]
    ### initialize params
    params = batman.TransitParams()
    params.t0, params.per, params.rp,params.b, params.T14, q1, q2 = theta_initial
    params.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]  # Limb darkening coefficients
    params.limb_dark = 'quadratic'
                
    transit_model = batman.TransitModel(params, time[mask])
                
    # Generate model light curve
    model_flux = transit_model.light_curve(params)

    # Binning function
    def bin_photometry(time, flux, bin_size):
        bins = np.arange(time.min(), time.max(), bin_size)
        digitized = np.digitize(time, bins)
        binned_time = []
        binned_flux = []
        for i in range(1, len(bins)):
            bin_mask = digitized == i
            if bin_mask.any():
                binned_time.append(time[bin_mask].mean())
                binned_flux.append(flux[bin_mask].mean())
        return np.array(binned_time), np.array(binned_flux)

    # Bin the data
    bin_size = 0.04  # Define bin size
    binned_time, binned_flux = bin_photometry(time[mask], flux[mask], bin_size)

    # Plot binned data and photometry and model
    plt.scatter(time[mask],flux[mask], s=10,label='data')
    plt.plot(time[mask], model_flux, color='red',label='model')
    plt.scatter(binned_time, binned_flux, color='orange', s=15, label='Binned Data')
    plt.title(f'transit {i+1}')
    plt.xlabel('transit time (days)')
    plt.ylabel('flux')
    plt.legend()
    plt.show()

'''


'''
# Example parameters
per_b = 7.9222
rp_b = 0.0777
T14_b = 3.237 * 0.0416667  # convert to days
b_b = 0.17
t0_b = 	2530
q1_b = 0.4
q2_b = 0.3
theta_initial = [t0_b, per_b, rp_b, b_b, T14_b, q1_b, q2_b]
# Calculate log-likelihood for initial parameters
#lg_like = log_likelihood(theta_initial, time, flux, flux_err)


params = batman.TransitParams()
params.t0, params.per, params.rp,params.b, params.T14, q1, q2 = theta_initial

params.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]  # Limb darkening coefficients
params.limb_dark = 'quadratic'
    
transit_model = batman.TransitModel(params, time)
    
    # Generate model light curve
model_flux = transit_model.light_curve(params)
mask = (time > (t0_b - 5*0.0416667)) & (time < (t0_b + 5*0.0416667))
plt.scatter(time, flux)
plt.plot(time, model_flux, color='red')
plt.show()

# Calculate chi-squared value
sigma2 = flux_err #f value from tutorial??  model**2 * np.exp(2*log_f)
chi_squared = np.sum(((flux - model_flux) / sigma2)**2)
#print(chi_squared)



def log_likelihood(theta, x, y, yerr):
    # Initialize batman transit model
    params = batman.TransitParams()
    params.t0, params.per, params.rp,params.b, params.T14, q1, q2 = theta

    params.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]  # Limb darkening coefficients
    params.limb_dark = 'quadratic'
    
    transit_model = batman.TransitModel(params, x)
    
    # Generate model light curve
    model_flux = transit_model.light_curve(params)
    
    # Calculate chi-squared value
    sigma2 = yerr**2 #f value from tutorial??  model**2 * np.exp(2*log_f)
    chi_squared = np.sum(((y - model_flux) / sigma2)**2)
    print(chi_squared)
    
    # Compute log-likelihood
    log_likelihood = -0.5 * chi_squared
    
    return log_likelihood


'''

