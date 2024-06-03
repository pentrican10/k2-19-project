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

df = read_table(file)

### generate ttv (lin ephem from params in table 3)
### planet c
p_c = 11.881982213238517 #11.8993   from linear regression
tc_c = 2021.6596526681883 #2020.0007
transit_c = [84, 99]
predicted_time_c=[]
for i in transit_c:
    ephem = tc_c + i* p_c
    predicted_time_c.append(ephem)
print(f'Predicted times(ephem): {predicted_time_c}')

pl_c = df[df["Planet"] == "K2-19c"]
paper_ttv_c = pl_c.Tc.values - predicted_time_c
print(f'TC(paper): {pl_c.Tc.values}')
print(f'TTV from ephem: {paper_ttv_c}')


### Download the light curve data
lc = lk.search_lightcurve("K2-19",author = 'SPOC').download_all()

### Flatten the light curve
lc = lc.stitch().flatten(window_length=901).remove_outliers()


### from BLS in k2-19_project.py
planet_c_period = 101.53385338533853
planet_c_t0 = 2553.160770815975
planet_c_dur =0.2
print(f'Period(BLS) c: {planet_c_period}')
print(f'Tc(BLS) c: {planet_c_t0}')


### initialize guess times
transit_num = [0,7]

tc_guess=[]
for num in transit_num:
    t = planet_c_t0 + (num * planet_c_period)
    tc_guess.append(t)
# ## num with paper ephem
t_num_paper_c = [227,287]


time_tess = np.array(lc.time.value)
flux=np.array(lc.flux)
flux_err = np.array(lc.flux_err)

def convert_time(times):
    ### TESS offset 
    BTJD = times + 2457000
    new_time = BTJD - 2454833
    return new_time

print(tc_guess)
time = convert_time(time_tess)
tc_guess = convert_time(np.array(tc_guess))
tc_guess = np.array(tc_guess)
print(f'TC guess(TESS): {tc_guess}')

per_c = 11.8993
rp_c = 0.0458
T14_c = 3.823 * 0.0416667
b_c = 0.630
q1_c = 0.4
q2_c = 0.3
### number of transits for planet b
num_b = 8
num_c = 2

def omc(obs_time, t_num, p, tc):
    calc_time = tc + (t_num* p)
    omc = obs_time - calc_time
    return omc#*24 #days to hours

#tc1 = np.linspace(time.min(),2600, 100)
ttv_min= 0.00694444
### set range for search: [#hours] * [days per hour]
ttv_hour = 6* 0.0416667 # 1 hour to days
#tc_guess = (2530.28, 2546.12, 2554.04, 2561.96, 2569.88, 2577.8, 3266.84, 3282.68)

### get tc ranges 
tc = []
for i in range(len(tc_guess)):
    start = tc_guess[i] - ttv_hour
    end = tc_guess[i] + ttv_hour
    t = np.linspace(start,end, 100)
    tc.append(t)


tc_chi = np.zeros(len(tc))
ttv = np.zeros(len(tc))
errors = []
### plot X^2 vs tc for each guess
for j in range(len(tc)):
    tc1 = tc[j]
    chi_sq = np.zeros(len(tc1))
    for i in range(len(tc1)):
        t0_c = 	tc1[i]
        theta_initial = [t0_c, per_c, rp_c, b_c, T14_c, q1_c, q2_c]
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

        

    #print(chi_sq)
    min_chi_time = tc1[np.argmin(chi_sq)]
    min_chi = chi_sq.min()
    tc_chi[j] = min_chi_time
    idx = t_num_paper_c[j]
    ttv[j] = omc(min_chi_time, idx, p_c, tc_c)#*24 #days to hours
    #ttv[j] = omc 

    ### delta chisq = 1 gives errors
    err_threshold = min_chi + 1
    # Find the intersection points
    def intersection_func(t):
        return np.interp(t, tc1, chi_sq) - err_threshold
    
    # Find the intersection using root_scalar
    intersections = []
    for k in range(len(tc1) - 1):
        if (chi_sq[k] - err_threshold) * (chi_sq[k + 1] - err_threshold) < 0:
            sol = root_scalar(intersection_func, bracket=[tc1[k], tc1[k + 1]])
            if sol.converged:
                intersections.append((sol.root - min_chi_time))
    errors.append(intersections)
    
    plt.plot(tc1, chi_sq,label='chisq')
    plt.axvline(x=tc_guess[j], color='r', linestyle='--', label='Bls Guess')
    plt.axvline(x=min_chi_time, color='green', linestyle='--', label='Chisq min')
    # for inter in intersections:
    #     plt.axvline(x=inter, color='blue', linestyle='--')
    plt.axhline(y=err_threshold, color='purple', linestyle='--', label='Error Threshold')
    plt.title(f'Transit {j+1}: Planet c')
    plt.xlabel('tc')
    plt.ylabel('X^2')
    plt.legend()
    plt.show()

print(f'Transit Times(TESS): {tc_chi}')
 
#avg the errors   sig^2 = 0.5(sig1^2 + sig2^2)
error = []
for i in range(len(errors)):
    sig = np.sqrt(errors[i][0]**2 + errors[i][1]**2)
    error.append(sig)
print(f'Avg Errors: {error}')

print(f'TTV(TESS): {ttv}')




plt.scatter(tc_chi, ttv)
#plt.scatter(pl_b.Tc.values, paper_ttv_b)
plt.scatter(pl_c.Tc.values, paper_ttv_c)

plt.title(f'TTV Paper: Planet c')
plt.xlabel('tc')
plt.ylabel('TTV value')
plt.show()

