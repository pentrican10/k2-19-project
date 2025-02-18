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

    ### Read the text file, specifying space as the delimiter, skipping model_guess_omc rows
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

### generate ttv (lin ephem from params in table 3)
### paper params
p_b = 7.9222 #7.920925490169578   ### used linear regression, changed the slope to the one extracted original paper value 7.9222
tc_b = 2027.9023 #2027.9158659031389 ###2027.9023
print(f'Period(paper) b: {p_b}')
print(f'TC(paper) b: {tc_b}')

tnum_k2 = [24,28,35,127,135,144]
### predict transit times
predicted_time = []
for i in tnum_k2:
    ephem = tc_b + i* p_b
    predicted_time.append(ephem)

### extract values from paper
planet_b = df[df["Planet"] == "K2-19b"]
tc_paper = planet_b.Tc.values
err_tc_paper = planet_b.e_Tc.values
### calculate ttv from predicted times
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
#transit_num = [0,2,3,4,5,6,93,95] # bls ephem
transit_num = [63,65,66,67,68,69,156,158] # paper ephem
### transit num with paper ephem
tnum_tess = [337,339,340,341,342,343,430,432]


# tc_guess=[]
# for num in transit_num:
#     t = planet_b_t0 + (num * planet_b_period)
#     tc_guess.append(t)

tc_guess=[]
for num in transit_num:
    t = tc_b + (num * p_b)
    tc_guess.append(t)

print(tc_guess)
### data from lightcurve 
time_tess = np.array(lc.time.value)
flux=np.array(lc.flux)
flux_err = np.array(lc.flux_err)

time = convert_time(time_tess)
tc_guess = convert_time(np.array(tc_guess))
tc_guess = np.array(tc_guess)
print(f'TC guess(TESS): {tc_guess}')

#assert 1==0

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
    idx = tnum_tess[j]
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
    parab_best_fit_1 = a_chi_sq*tc1**2 + b_chi_sq*tc1 + c_chi_sq

    ### calculate ttv from parabola fit 
    ttv_p[j] = omc(tc_best_fit, idx, p_b, tc_b)#*24 #days to hours
    

    
  

    ### delta chisq = 1 gives errors
    err_threshold = min_chi + 1 # using chisq discrete minimum
    err_threshold_p = chi_sq_min + 1 # using minimum of parabola

    plt.plot(tc1[chi_mask], chi_sq[chi_mask],label='chisq')
    plt.plot(tc1[chi_mask], parab_best_fit_1[chi_mask],label='chisq parabola', color='orange')
    plt.axvline(x=tc_guess[j], color='r', linestyle='--', label='Bls Guess')
    plt.axvline(x=min_chi_time, linestyle='--', label='Chisq min')
    plt.axvline(x=tc1[np.argmin(parab_best_fit_1)], color='orange', linestyle='--', label='Chisq min parabola')

    # for inter in intersections:
    #     plt.axvline(x=inter, color='blue', linestyle='--')
    plt.axhline(y=err_threshold, color='green', linestyle='--', label='Error Threshold')
    plt.title(f'Transit {j+1}: Planet b')
    plt.xlabel('tc')
    plt.ylabel('X^2')
    plt.legend()
    plt.show()


  
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
        if (parab_best_fit_1[k] - err_threshold_p) * (parab_best_fit_1[k + 1] - err_threshold_p) < 0:
            sol = root_scalar(intersection_func, bracket=[tc1[k], tc1[k + 1]])
            if sol.converged:
                intersections_p.append((sol.root - tc_best_fit))
    errors_p.append(intersections_p)

    

  
    # plt.plot(tc1[chi_mask], chi_sq[chi_mask],label='chisq')
    # plt.plot(tc1[chi_mask], parab_best_fit_1[chi_mask],label='chisq parabola', color='orange')
    # plt.axvline(x=tc_guess[j], color='r', linestyle='--', label='Bls Guess')
    # plt.axvline(x=min_chi_time, linestyle='--', label='Chisq min')
    # plt.axvline(x=tc1[np.argmin(parab_best_fit_1)], color='orange', linestyle='--', label='Chisq min parabola')

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

# Define a sine function
def sine_function(t, A, omega, phi, p, offset):
    return A * np.sin(omega * t + phi) + (p * t) + offset


###################################################################################################


### method 2: fit the transit times and indices directly
### Initial guess for the parameters: amplitude, frequency, phase, period, offset
initial_guess2 = [0.07, 2*np.pi/100, -np.pi/2, 7.92089, 2027.84470]

# Concatenate all data points for fitting
all_tc = np.concatenate([tc_paper, tc_chi_parabola])
all_ttv = np.concatenate([ paper_ttv_b, ttv_p])
all_err_ttv = np.concatenate([err_paper_ttv_b, err_tc_chi_p])
all_tnum = np.concatenate([tnum_k2, tnum_tess])

# Fit the sine function to the data
popt2, pcov2 = curve_fit(sine_function, all_tnum, all_tc, p0=initial_guess2, sigma=all_err_ttv, absolute_sigma=True)
A_fit2, omega_fit2, phi_fit2, p2_fit, offset_fit2 = popt2
print(f'offset2: {offset_fit2}')
print(f'p2_fit: {p2_fit}')

tnum_k2 = np.array(tnum_k2,dtype='float64')
tnum_tess = np.array(tnum_tess,dtype='float64')

### Calculate the fitted TTV values
tnum_plot = np.linspace(min(all_tnum), max(all_tnum), 1000)

def ephem_fit(tnum):
    return (p2_fit*tnum) + offset_fit2
def ephem_guess(tnum):
    return (initial_guess2[3]*tnum) + initial_guess2[4]

model_fit_sine_simul = sine_function(tnum_plot, *popt2) - ephem_fit(tnum_plot)

ttv_chi_parabola_simul = tc_chi_parabola - ephem_fit(tnum_tess)
ttv_paper_simul = tc_paper - ephem_fit(tnum_k2)

model_guess_simul = sine_function(tnum_plot,*initial_guess2) - ephem_fit(tnum_plot)

fig2, (ax3, ax4) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},sharex=True)
ax3.plot(ephem_fit(tnum_plot), model_guess_simul, label='model_guess_omc guess')
ax3.plot(ephem_fit(tnum_plot), model_fit_sine_simul, 'r-', label='Fitted sine curve')
ax3.errorbar(ephem_fit(tnum_tess), ttv_chi_parabola_simul, yerr=err_tc_chi_p, fmt='bo', capsize=5, label='TESS times')
ax3.errorbar(ephem_fit(tnum_k2), ttv_paper_simul, yerr=err_tc_paper, fmt='bs', capsize=5, label='Paper times')
ax3.set_title('TTV Fit: Planet b')
ax3.set_ylabel('TTV (days)')
ax3.legend()

residual_paper = (ttv_paper_simul - sine_function(tnum_k2, *popt2) + ephem_fit(tnum_k2)) / err_tc_paper
residual_tess = (ttv_chi_parabola_simul - sine_function(tnum_tess, *popt2) + ephem_fit(tnum_tess)) / err_tc_chi_p
ax4.plot(ephem_fit(tnum_k2), residual_paper, 'bs')
ax4.plot(ephem_fit(tnum_tess), residual_tess, 'bo')
ax4.axhline(y=0, color='r', linestyle='-')
ax4.set_ylabel('Norm Residuals (days)')
ax4.set_xlabel('TC (days)')
plt.show()

from ttvfast_test import t_1, residuals1, t_2, residuals2
tc_plot = np.linspace(min(all_tc), max(all_tc), 1000)


### plot
plt.scatter(t_1,residuals1,color='orange',s=4,label='planet b ttvfast')
plt.scatter(t_2,residuals2,color='blue',s=4, label='planet c ttvfast')
plt.errorbar(tc_chi_parabola, ttv_chi_parabola_simul, xerr=err_tc_chi_p, yerr=err_tc_chi_p, fmt='ro',color='purple', capsize=5, label='TESS times b')
plt.errorbar(tc_paper, ttv_paper_simul, xerr=err_tc_paper, yerr=err_paper_ttv_b, fmt='rs',color='green', capsize=5, label='Paper times b')
plt.plot(tc_plot, model_fit_sine_simul,color='black', label='Simult Fitted Sine')
plt.title("TTVfast and Measured Times - K2-19")
plt.ylabel('TTV (days)')
plt.xlabel('TC (days)')
plt.legend()
plt.show()


