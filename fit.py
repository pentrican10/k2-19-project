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

### create switch to use mask or not
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
def intersection_func_lc(t): #unmasked
    return np.interp(t, tc1, chi_sq_lc) - err_threshold_lc
    

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


### generate ttv (lin ephem from params in table 3)
p_b = 7.920925490169578   ### used linear regression, changed the slope to the one extracted original paper value 7.9222
tc_b = 2027.9158659031389 ###2027.9023
print(f'Period(paper) b: {p_b}')
print(f'TC(paper) b: {tc_b}')
transit_b = [24,28,35,127,135,144]
predicted_time = []
for i in transit_b:
    ephem = tc_b + i* p_b
    predicted_time.append(ephem)
print(f'Predicted times(ephem) b: {predicted_time}')

pl_b = df[df["Planet"] == "K2-19b"]
paper_ttv_b = pl_b.Tc.values - predicted_time
print(f'TC(paper) b: {pl_b.Tc.values}')
print(f'TTV from ephem b: {paper_ttv_b}')

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


### Download the light curve data
lc = lk.search_lightcurve("K2-19",author = 'SPOC').download_all()
lc = lc.stitch()
### plot this without flatten, mask transit times before flattening
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
### flatten unmasked lightcurve 
lc = lc.flatten(window_length=901).remove_outliers()


### from BLS in k2-19_project.py
planet_b_period = 7.9204920492049204
planet_b_t0 = 2530.2807708159753
print(f'Period(BLS): {planet_b_period}')
print(f'Tc(BLS): {planet_b_t0}')

### initialize guess times
transit_num = [0,2,3,4,5,6,93,95]

tc_guess=[]
for num in transit_num:
    t = planet_b_t0 + (num * planet_b_period)
    tc_guess.append(t)

### transit num with paper ephem
t_num_paper = [337,339,340,341,342,343,430,432]
t_num_paper_c = []

### masked data
time_tess = np.array(masked_lc.time.value)
flux=np.array(masked_lc.flux)
flux_err = np.array(masked_lc.flux_err)

### un-masked data
time_tess_lc = np.array(lc.time.value)
flux_lc=np.array(lc.flux)
flux_err_lc = np.array(lc.flux_err)

time = convert_time(time_tess)
time_lc = convert_time(time_tess_lc)
tc_guess = convert_time(np.array(tc_guess))
tc_guess = np.array(tc_guess)
print(f'TC guess(TESS): {tc_guess}')



ttv_min= 0.00694444
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
tc_chi_lc = np.zeros(len(tc))
ttv = np.zeros(len(tc))
ttv_lc = np.zeros(len(tc))
errors = []
errors_lc = []
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

        ### repeat for the unmasked lc
        mask_lc = (time_lc > (start)) & (time_lc < (end))
        transit_model_lc = batman.TransitModel(params, time_lc[mask_lc])
        model_flux_lc = transit_model_lc.light_curve(params)
        sigma2_lc = flux_err_lc[mask_lc]
        chi_squared_lc = np.sum(((flux_lc[mask_lc] - model_flux_lc) / sigma2_lc)**2)
        chi_sq_lc[i] = (chi_squared_lc)

    ### fit parabola to the chisq
    p_chi_sq = np.polyfit(tc1, chi_sq, 2)  #masked
    p_chi_sq_lc = np.polyfit(tc1, chi_sq_lc, 2)  #unmasked 

    ### Extract the coefficients   y = ax^2 + bx + c
    a_chi_sq, b_chi_sq, c_chi_sq = p_chi_sq
    a_chi_sq_lc, b_chi_sq_lc, c_chi_sq_lc = p_chi_sq_lc
    
    ### Find the minimum of the parabola xmin = -b/2a from taking derivative=0
    tc_best_fit = -b_chi_sq / (2 * a_chi_sq)
    tc_best_fit_lc = -b_chi_sq_lc / (2 * a_chi_sq_lc)
    
    ### Calculate the minimum chi-squared value
    chi_sq_min = a_chi_sq * tc_best_fit**2 + b_chi_sq * tc_best_fit + c_chi_sq
    chi_sq_min_lc = a_chi_sq_lc * tc_best_fit_lc**2 + b_chi_sq_lc * tc_best_fit_lc + c_chi_sq_lc

    ### Calculate the parabola best fit 
    p_1 = a_chi_sq*tc1**2 + b_chi_sq*tc1 + c_chi_sq
    

    ### masked
    min_chi_time = tc1[np.argmin(chi_sq)]
    min_chi = chi_sq.min()

    tc_chi[j] = min_chi_time
    idx = t_num_paper[j]
    ttv[j] = omc(min_chi_time, idx, p_b, tc_b)#*24 #days to hours

    ### unmasked
    min_chi_time_lc = tc1[np.argmin(chi_sq_lc)]
    min_chi_lc = chi_sq_lc.min()

    tc_chi_lc[j] = min_chi_time_lc
    idx_lc = t_num_paper[j]
    ttv_lc[j] = omc(min_chi_time_lc, idx_lc, p_b, tc_b)#*24 #days to hours


    ### delta chisq = 1 gives errors
    err_threshold = min_chi + 1
    err_threshold_lc = min_chi_lc +1
  
    # Find the intersection using root_scalar
    intersections = []
    for k in range(len(tc1) - 1):
        if (chi_sq[k] - err_threshold) * (chi_sq[k + 1] - err_threshold) < 0:
            sol = root_scalar(intersection_func, bracket=[tc1[k], tc1[k + 1]])
            if sol.converged:
                intersections.append((sol.root - min_chi_time))
    errors.append(intersections)

    intersections_lc = []
    for k in range(len(tc1) - 1):
        if (chi_sq_lc[k] - err_threshold_lc) * (chi_sq_lc[k + 1] - err_threshold_lc) < 0:
            sol_lc = root_scalar(intersection_func_lc, bracket=[tc1[k], tc1[k + 1]])
            if sol_lc.converged:
                intersections_lc.append((sol_lc.root - min_chi_time_lc))
    errors_lc.append(intersections_lc)
    
    plt.plot(tc1, chi_sq,label='chisq')
    plt.plot(tc1, p_1,label='chisq parabola', color='orange')
    plt.axvline(x=tc_guess[j], color='r', linestyle='--', label='Bls Guess')
    plt.axvline(x=min_chi_time, color='green', linestyle='--', label='Chisq min')
    plt.axvline(x=tc1[np.argmin(p_1)], color='orange', linestyle='--', label='Chisq min parabola')

    # for inter in intersections:
    #     plt.axvline(x=inter, color='blue', linestyle='--')
    plt.axhline(y=err_threshold, color='purple', linestyle='--', label='Error Threshold')
    plt.title(f'Transit {j+1}: Planet b')
    plt.xlabel('tc')
    plt.ylabel('X^2')
    plt.legend()
    plt.show()

print(f'Transit Times(TESS) Masked: {tc_chi}')
print(f'Transit Times(TESS) Unmasked: {tc_chi_lc}')
print(f'Difference in times(masked-unmasked): {tc_chi - tc_chi_lc}')
 
#avg the errors   sig^2 = 0.5(sig1^2 + sig2^2)
err_tc_chi = []
for i in range(len(errors)):
    sig = np.sqrt(errors[i][0]**2 + errors[i][1]**2)
    err_tc_chi.append(sig)
print(f'Avg Errors Masked: {err_tc_chi}')

error_lc = []
for i in range(len(errors_lc)):
    sig = np.sqrt(errors_lc[i][0]**2 + errors_lc[i][1]**2)
    error_lc.append(sig)
print(f'Avg Errors Unmasked: {error_lc}')

print(f'TTV(TESS) Masked: {ttv}')
print(f'TTV(TESS) Unmasked: {ttv_lc}')




plt.scatter(tc_chi, ttv)
plt.scatter(pl_b.Tc.values, paper_ttv_b)
#plt.scatter(pl_c.Tc.values, paper_ttv_c)

plt.title(f'TTV Paper: Planet b')
plt.xlabel('tc')
plt.ylabel('TTV value')
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
    ### start lstsq with the chisq guess
    t0_b = tc_chi[i]
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
    cov = np.linalg.pinv(J.T @ J)   #is this ok? I got an err_tc_chi with .inv and it suggested using .pinv pseudo inverse
    
    # Calculate standard errors
    errors = np.sqrt(np.diag(cov))
    errors_list.append(errors)


### Output results
tc_lstsq = []
err_tc_lstsq = []
print("Optimal Parameters for each guess:")
for params, errors in zip(optimal_params_list, errors_list):
    tc_lstsq.append(params[0])
    err_tc_lstsq.append(errors[0])


# print(f'Transit Times(Least Sq) Masked: {tc_lstsq}')
# print(f'Errors (Least Sq) Masked: {err_tc_lstsq}')
# print(f'Transit Times(TESS Chi sq) Masked: {tc_chi}')
# print(f'Avg Errors (chi sq) Masked: {err_tc_chi}')
# print(f'TC guess(TESS): {tc_guess}')
### Print the rounded values
print(f'Transit Times(Least Sq) Masked: {[round(val, 4) for val in tc_lstsq]}')
print(f'Errors (Least Sq) Masked: {[round(val, 4) for val in err_tc_lstsq]}')
print(f'Transit Times(TESS Chi sq) Masked: {[round(val, 4) for val in tc_chi]}')
print(f'Avg Errors (chi sq) Masked: {[round(val, 4) for val in err_tc_chi]}')
print(f'TC guess(TESS): {[round(val, 4) for val in tc_guess]}')

transit_index = range(len(tc_lstsq))

### Loop through each transit index to create individual plots
for i in transit_index:
    ### Plot Chi Square with error bars
    plt.errorbar(i+1, tc_chi[i], yerr=err_tc_chi[i], fmt='s', label='Chi Square', capsize=5)

    ### Plot Least Squares with error bars
    plt.errorbar(i+1, tc_lstsq[i], yerr=err_tc_lstsq[i], fmt='o', label='Least Squares', capsize=5)

    ### Plot TC Guess
    plt.plot(i+1, tc_guess[i], 'x', label='TC Guess')

    plt.xlabel('Transit num')
    plt.ylabel('Transit Times')
    plt.title(f'Transit Times with Error Bars - Transit {i+1}')
    plt.legend()
    plt.show()


####################################################################################################################
'''
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

