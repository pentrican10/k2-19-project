import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
import numpy as np
import batman
from bls_fit import time, flux, flux_err

# Prepare time and flux
time = time.value
flux = flux.value
flux_err = flux_err.value

# Time conversion
petigura_offset = 2454833
tess_offset = 2457000

def convert_time_t2p(times):
    BTJD = times + tess_offset
    return BTJD - petigura_offset

time = convert_time_t2p(time)

# Transit parameters
per_b = 7.9222
rp_b = 0.0777
T14_b = 3.237 * 0.0416667
b_b = 0.17
q1_b = 0.4
q2_b = 0.3

# Transit windows
buffer = 1.5 * T14_b
transit_times = [
    4697.289029080774, 4713.127343841249, 4721.041829879279, 4728.959652590447,
    4736.880811974289, 4744.798634685457, 5433.878773628148, 5449.720258227197
]

cols = 3
rows = int(np.ceil(len(transit_times) / cols))

fig = plt.figure(figsize=(14, rows * 2.8))
outer = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.3, hspace=0.5)

for i, t0 in enumerate(transit_times):
    row = i // cols
    col = i % cols
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, col],
                                             height_ratios=[3, 1], hspace=0.0)  # Panels touch

    t_min = t0 - buffer
    t_max = t0 + buffer
    mask = (time >= t_min) & (time <= t_max)

    # Transit model
    theta_initial = [t0, per_b, rp_b, b_b, T14_b, q1_b, q2_b]
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.b, params.T14, q1, q2 = theta_initial
    params.u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]
    params.limb_dark = 'quadratic'

    m = batman.TransitModel(params, time[mask])
    model_flux = m.light_curve(params)
    residuals = flux[mask] - model_flux

    # Top panel: model
    ax1 = fig.add_subplot(inner[0])
    ax1.scatter(time[mask], flux[mask], s=5)
    ax1.plot(time[mask], model_flux, color='red')
    ax1.axvline(t0, color='green', linestyle='--')
    ax1.set_title(f'Tc {round(t0, 4)}')
    ax1.set_xlim(t_min, t_max)
    ax1.grid(True)
    ax1.tick_params(labelbottom=False)

    # Bottom panel: residuals
    ax2 = fig.add_subplot(inner[1], sharex=ax1)
    ax2.scatter(time[mask], residuals, s=5)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlim(t_min, t_max)
    ax2.set_xlabel(f"Time (BJD - {petigura_offset})")
    ax2.grid(True)

    # Only show y-axis labels for first column
    if col == 0:
        ax1.set_ylabel("Flux")
        ax2.set_ylabel("Res.")
    else:
        ax1.set_ylabel("")
        ax1.tick_params(labelleft=False)
        ax2.set_ylabel("")
        ax2.tick_params(labelleft=False)

    # Custom ticks
    ticks = np.round(np.linspace(t_min, t_max, 4), 2)
    ax2.set_xticks(ticks)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3}"))

plt.show()











'''
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import batman
from bls_fit import time, flux, flux_err

time = time.value
flux = flux.value
flux_err = flux_err.value

### Change time convention
petigura_offset = 2454833  # BJD - 2454833
tess_offset = 2457000  # BTJD - Barycentric TESS Julian Date (Julian Date - 2457000)

def convert_time_t2p(times):
    BTJD = times + tess_offset
    return BTJD - petigura_offset

time = convert_time_t2p(time)

### Transit parameters
per_b = 7.9222
rp_b = 0.0777
T14_b = 3.237 * 0.0416667  # Convert to days
b_b = 0.17
q1_b = 0.4
q2_b = 0.3

### Define a buffer around each transit time (e.g., 1.5 * transit duration)
buffer = 1.5 * T14_b

### List of transit times
transit_times = [
    4697.289029080774, 4713.127343841249, 4721.041829879279, 4728.959652590447,
    4736.880811974289, 4744.798634685457, 5433.878773628148, 5449.720258227197
]

### Create subplots
num_transits = len(transit_times)
cols = 3  # Number of columns in subplot grid
rows = (num_transits + 1) // cols  # Determine number of rows dynamically

fig, axes = plt.subplots(rows, cols, figsize=(14, 8), sharey=True)
axes = axes.flatten()  # Flatten in case of 2D array

### Loop through each transit time and plot in corresponding subplot
for i, t0 in enumerate(transit_times):
    t_min = t0 - buffer
    t_max = t0 + buffer

    ### Select only the data within this transit window
    mask = (time >= t_min) & (time <= t_max)

    ### Generate model
    theta_initial = [t0, per_b, rp_b, b_b, T14_b, q1_b, q2_b]
    
    ### Initialize batman parameters
    params = batman.TransitParams()
    params.t0, params.per, params.rp, params.b, params.T14, q1, q2 = theta_initial
    params.u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]  # Limb darkening coefficients
    params.limb_dark = 'quadratic'

    ### Generate transit model
    transit_model = batman.TransitModel(params, time[mask])
    model_flux = transit_model.light_curve(params)

    ### Plot in subplot
    ax = axes[i]
    ax.scatter(time[mask], flux[mask], label="TESS Data", s=5)
    ax.plot(time[mask], model_flux, color="red", label="Transit Model")
    ax.axvline(x=t0, color='green', linestyle='--', label=f'Tc {round(t0, 4)}')

    ax.set_xlabel(f'Time (BJD - {petigura_offset})')
    ax.set_xlim(t_min, t_max)
    ax.set_ylabel('Flux')
    ax.set_title(f'Tc {round(t0,4)}')
    ### set number of ticks and force to be in full format (not trunkated)
    ticks = np.round(np.linspace(t_min, t_max, 5), 2)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3f}"))
    #ax.legend(loc='lower left')
    ax.grid(True)

### Remove empty subplots (if any)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

### Adjust layout
plt.tight_layout()
plt.show()
'''

