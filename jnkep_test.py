import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("TkAgg")  # Use an interactive backend
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from jax import config, random
import numpyro, jax
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
config.update('jax_enable_x64', True)
numpyro.set_platform('cpu') 
num_chains = 4
numpyro.set_host_device_count(num_chains)
print ('# jax device count:', jax.local_device_count())

from jnkepler.jaxttv import JaxTTV
from jnkepler.jaxttv import ttv_default_parameter_bounds, ttv_optim_curve_fit, scale_pdic
import corner
# adding comment for testing 
### test comment 


# test data
d = pd.read_csv("ttv_results.txt", sep="\s+", header=0, names=['Planet_num', 'Index', 'Tc', 'Tc_err', 'OMC', 'Source', 'Instrument'])
### use only narita and first 3 petigura points 
# d = pd.read_csv("intermediate.txt", sep="\s+", header=None, names=['Planet_num', 'Index', 'Tc', 'Tc_err', 'OMC', 'Source', 'Instrument'])


# these are what we need
list_of_obs_transit_times = []
list_of_transit_time_errs = []
period_guess = [7.9222, 11.8993] # rough initial guess to track transit epochs (from Petigura 2020)
# period_guess = [7.91975198, 11.90285385]
for j in range(2):
    list_of_obs_transit_times.append(np.array(d.Tc[d.Planet_num==j+1]))
    list_of_transit_time_errs.append(np.array(d.Tc_err[d.Planet_num==j+1]))

tcobs = []
errorobs = []
for j in range(2):
    tcobs.append(np.array(d.Tc[d.Planet_num == j + 1]))
    errorobs.append(np.array(d.Tc_err[d.Planet_num == j + 1]))


t_start = 1980.  # start of integration
t_end = 5500. # end of integration
dt = 0.1 # integration timestep
# jttv = JaxTTV(t_start, t_end, dt, list_of_obs_transit_times, period_guess, errorobs=list_of_transit_time_errs)
jttv = JaxTTV(t_start, t_end, dt, tcobs, period_guess, errorobs=errorobs, print_info=True)


# masses, and osculating orbital elements at t_start
e2s = (1) / (333000) # [solar mass]
# par_dict = {
#     "pmass": np.array([32.4*e2s, 10.8*e2s]), # planets' masses (solar mass)
#     # "period": np.array([7.9222, 11.8993]),
#     "period": np.array([7.92091, 11.8980]),
#     # "ecosw": np.array([0.02, 0.04]),
#     # "esinw": np.array([-0.19, -0.21]),
#     "ecc": np.array([0.2,0.21]),
#     "omega": np.array([3*np.pi/2, 3*np.pi/2]),
#     "tic": np.array([2027.9023, 2020.0007]), # time of inferior conjunction
#     ## default values when these parameters are not specified
#     "cosi": np.array([0, 0]),  # cosine of orbital inclination
#     "lnode": np.array([0, 0]), # longitude of ascending node
#     "smass": .88 # stellar mass
# }

par_dict = {
    "pmass": np.array([1.38177828e-04, 3.02486158e-05]), # planets' masses (solar mass)
    "period": np.array([7.9222, 11.8993]),
    # "period": np.array([ 7.91934959, 15.71986409]),
    "ecosw": np.array([-0.11691701, -0.07462371]),
    "esinw": np.array([-0.22205269, -0.24999952]),
    # "ecc": np.array([0.2,0.21]),
    # "omega": np.array([3*np.pi/2, 3*np.pi/2]),
    "tic": np.array([1980.38371386, 1984.27143437]), # time of inferior conjunction
    ## default values when these parameters are not specified
    "cosi": np.array([0.01, 0.01]),  # cosine of orbital inclination
    "lnode": np.array([0, 0]), # longitude of ascending node
    "smass": .88 # stellar mass
}


param_bounds = ttv_default_parameter_bounds(jttv, emax=0.25)

### initialize p_init: p1,p2,ecosw1,ecosw2,esinw1,esinw2,tic1,tic2,lnpmass1,lnpmass2
### initial guess from the fit for initial points 
# p_init = np.array([7.91975198, 11.90285385, -0.11691701, -0.07462371, -0.22205269, -0.24999952, 1980.38371386, 1984.27143437, -8.88696909, -10.40606013])
### adding other three petigura points for planet 1
# p_init = np.array([7.92012932, 11.90212229, -0.12788516, -0.08892373, -0.23193693, -0.24999966, 1980.38354037, 1984.27245905, -8.98233947, -9.96274083])
### adding the 2 petigura points for planet 2
# p_init = np.array([7.92008386, 11.90197663, -0.13080481, -0.09067855, -0.22900052, -0.24999992, 1980.38358244, 1984.27249886, -8.96152827, -10.02018136])
### best fit from first run with all points
p_init = np.array([7.91978264, 11.90381532, -0.12693783, -0.08689699, -0.14793586, -0.17598388, 1980.38363425, 1984.27272612, -9.07538167, -10.13796006])

popt = ttv_optim_curve_fit(jttv,param_bounds,p_init=p_init, plot=False)
# plt.show()
print(popt)
tcall = jttv.get_transit_times_all_list(popt,truncate=False)
jttv.plot_model(tcall, marker='.')
plt.show()


tc, _ = jttv.check_timing_precision(popt)

pdic_normal, pdic_student = jttv.check_residuals(popt)
plt.show()


### setup & run HMC
def model_scaled(sample_keys, param_bounds):
    """numpyro model for scaled parameters"""
    par = {}

    # sample parameters from priors
    for key in sample_keys:
        par[key+"_scaled"] = numpyro.sample(key+"_scaled", dist.Uniform(param_bounds[key][0]*0, param_bounds[key][0]*0+1.))
        par[key] = numpyro.deterministic(key, par[key+"_scaled"] * (param_bounds[key][1] - param_bounds[key][0]) + param_bounds[key][0])
    if "pmass" not in sample_keys:
        par["pmass"] = numpyro.deterministic("pmass", jnp.exp(par["lnpmass"]))
    
    # Jacobian for uniform ecc prior
    ecc = numpyro.deterministic("ecc", jnp.sqrt(par['ecosw']**2+par['esinw']**2))
    numpyro.factor("eprior", -jnp.log(ecc))

    # compute transit times
    tcmodel, ediff = jttv.get_transit_times_obs(par)
    numpyro.deterministic("ediff", ediff)
    numpyro.deterministic("tcmodel", tcmodel)
    
    # likelihood
    tcerrmodel = jttv.errorobs_flatten     
    numpyro.sample("obs", dist.Normal(loc=tcmodel, scale=tcerrmodel), obs=jttv.tcobs_flatten)

sample_keys = ["period","ecosw", "esinw", "pmass", "tic"] # uniform mass prior
# scaled parameters
pdic_scaled = scale_pdic(popt, param_bounds)
kernel = NUTS(model_scaled, 
            init_strategy=init_to_value(values=pdic_scaled), 
            dense_mass=True,
            #regularize_mass_matrix=False # this speeds up sampling for unknown reason
            )
mcmc = MCMC(kernel, num_warmup=50, num_samples=150, num_chains=num_chains)

# 4hr30min on M1 mac studio
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, sample_keys, param_bounds, extra_fields=('potential_energy', 'num_steps', 'adapt_state'))
mcmc.print_summary()

### plot models drawn from posterior
samples = mcmc.get_samples()
means, stds = jttv.sample_means_and_stds(samples)
jttv.plot_model(means, tcmodelunclist=stds)
plt.savefig("posterior_model.png", dpi=300, bbox_inches="tight") 
plt.show()

### trace and corner plots
import arviz as az
idata = az.from_numpyro(mcmc)
fig = az.plot_trace(mcmc, var_names=sample_keys, compact=False)
plt.tight_layout(pad=0.2)
plt.savefig("trace_plot.png", dpi=300, bbox_inches="tight")
plt.show()

idata.posterior['mu'] = idata.posterior['pmass'] / 3.003e-6
names = ["period", "tic", "ecosw", "esinw", "mu"]
fig = corner.corner(idata, var_names=names, show_titles=True)
plt.savefig("corner_plot.png", dpi=300, bbox_inches="tight") 
plt.show()



# transit_times, fractional_energy_error = jttv.get_transit_times_obs(popt)
# print("# observed")
# print(jttv.tcobs_flatten)
# print()
# print("# model")
# print(transit_times) 

# ### encouraged to check timing precision
# _, _ = jttv.check_timing_precision(popt)

# ### list of all transit times between t_start and t_end
# tc_all_list = jttv.get_transit_times_all_list(popt)
# jttv.plot_model(tc_all_list)
# plt.show()'
