#!/usr/bin/env python
# coding: utf-8

# # Jnkepler OMC Fitting

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import norm
import matplotlib

import jax.numpy as jnp
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


# In[2]:


### Read results
d = pd.read_csv("ttv_results.txt", sep="\s+", header=0, names=['Planet_num', 'Index', 'Tc', 'Tc_err', 'OMC', 'Source', 'Instrument'])


# In[3]:


### get times, errs from the data
list_of_obs_transit_times = []
list_of_transit_time_errs = []
# period_guess = [7.9222, 11.8993] # rough initial guess to track transit epochs (from Petigura 2020)
period_guess = [7.91975198, 11.90285385]
for j in range(2):
    list_of_obs_transit_times.append(np.array(d.Tc[d.Planet_num==j+1]))
    list_of_transit_time_errs.append(np.array(d.Tc_err[d.Planet_num==j+1]))
index_obs_b = np.array(d.Index[d.Planet_num==1])
index_obs_c = np.array(d.Index[d.Planet_num==2])

tcobs = []
errorobs = []
for j in range(2):
    tcobs.append(np.array(d.Tc[d.Planet_num == j + 1]))
    errorobs.append(np.array(d.Tc_err[d.Planet_num == j + 1]))


# In[4]:


### run JaxTTV sim
t_start = 1980.  # start of ~integration
t_end = 5500. # end of integration
#t_end = 7700.
dt = 0.25 # integration timestep
# jttv = JaxTTV(t_start, t_end, dt, list_of_obs_transit_times, period_guess, errorobs=list_of_transit_time_errs)
jttv = JaxTTV(t_start, t_end, dt, tcobs, period_guess, errorobs=errorobs, print_info=True)


# In[5]:


### set bounds for fit
param_bounds = ttv_default_parameter_bounds(jttv, emax=0.25)

### initialize p_init: p1,p2,ecosw1,ecosw2,esinw1,esinw2,tic1,tic2,lnpmass1,lnpmass2
### using best fit for run with all points
p_init = np.array([7.91978264, 11.90381532, -0.12693783, -0.08689699, -0.14793586, -0.17598388, 1980.38363425, 1984.27272612, -9.07538167, -10.13796006])

### fit
popt = ttv_optim_curve_fit(jttv,param_bounds,p_init=p_init, plot=False)
# plt.show()
print(popt)


# In[6]:


### plot fit
tcall = jttv.get_transit_times_all_list(popt,truncate=False)
jttv.plot_model(tcall, marker='.')
plt.show()


# In[7]:


### check precision and residuals 
tc, _ = jttv.check_timing_precision(popt)

pdic_normal, pdic_student = jttv.check_residuals(popt)
plt.show()


# ### Save results

# In[8]:


### save data
### jnkep model times
t_jnkep_b = tcall[0]
t_jnkep_c = tcall[1]

### jnkep obs times
t_obs_b = tcobs[0]
t_obs_c = tcobs[1]

### best fit params 
best_fit_period_b = popt['period'][0]
best_fit_period_c = popt['period'][1]
best_fit_tc_b = popt['tic'][0]
best_fit_tc_c = popt['tic'][1]


# In[9]:


import json

# Convert all arrays to regular lists
fit_data = {
    "jnkep_model_times": {
        "b": tcall[0].tolist(),
        "c": tcall[1].tolist()
    },
    "jnkep_observed_times": {
        "b": tcobs[0].tolist(),
        "c": tcobs[1].tolist()
    },
    "jnkep_observed_index": {
        "b": index_obs_b.tolist(),
        "c": index_obs_c.tolist()
    },
    "best_fit_params": {
        "period": {
            "b": float(popt['period'][0]),
            "c": float(popt['period'][1])
        },
        "tc": {
            "b": float(popt['tic'][0]),
            "c": float(popt['tic'][1])
        }
    }
}

# Save to JSON
with open("jnkep_fit_data.json", "w") as f:
    json.dump(fit_data, f, indent=4)

print("Data saved successfully.")


# # Set up and run HMC

# In[10]:



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


# In[11]:


# physical parameters to sample from
sample_keys = ["ecosw", "esinw", "pmass", "period", "tic"] # uniform mass prior


# In[12]:


# scaled parameters
pdic_scaled = scale_pdic(popt, param_bounds)


# In[13]:


kernel = NUTS(model_scaled, 
            init_strategy=init_to_value(values=pdic_scaled), 
            dense_mass=True,
            regularize_mass_matrix=False, # this speeds up sampling for unknown reason
            max_tree_depth=11
            )


# In[14]:
# To do, check point code

mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=num_chains)


# In[15]:

rng_key = random.PRNGKey(0)
mcmc.run(rng_key, sample_keys, param_bounds, extra_fields=('potential_energy', 'num_steps', 'adapt_state'))


# In[ ]:


mcmc.print_summary()


# In[ ]:


# save results
import dill
with open("jnkep_fit_k2-19_nburn-1000_nsteps-1000_tree-11.pkl", "wb") as f:
    dill.dump(mcmc, f)


# # Plot models drawn from posteriors

# In[ ]:


samples = mcmc.get_samples()


# In[ ]:


means, stds = jttv.sample_means_and_stds(samples)


# In[ ]:


jttv.plot_model(means, tcmodelunclist=stds)


# # Trace and corner plots

# In[ ]:


import arviz as az
idata = az.from_numpyro(mcmc)
fig = az.plot_trace(mcmc, var_names=sample_keys, compact=False)
plt.tight_layout(pad=0.2)


# In[ ]:


idata.posterior['mu'] = idata.posterior['pmass'] / 3.003e-6
names = ["period", "tic", "ecosw", "esinw", "mu"]
fig = corner.corner(idata, var_names=names, show_titles=True)


# In[ ]:




