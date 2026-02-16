import os
import copy
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from IPython.display import clear_output

from openoa.analysis.aep import MonteCarloAEP
from openoa.utils import plot

import project_ENGIE
project = project_ENGIE.prepare(
    './data/la_haute_borne', use_cleansed=False)

plot.column_histograms(project.meter, columns=["MMTR_SupWh"])
plot.column_histograms(project.curtail, columns=[
    "IAVL_DnWh", "IAVL_ExtPwrDnWh"])

project.meter.head()

pa = MonteCarloAEP(project, reanalysis_products=['era5', 'merra2'])

# View the monthly data frame
pa.aggregate.head()

pa.plot_normalized_monthly_reanalysis_windspeed(
    return_fig=False,
    xlim=(datetime(1996, 1, 1), datetime(2021, 12, 31)),
    ylim=(0.8, 1.2),
)

pa.plot_reanalysis_gross_energy_data(
    outlier_threshold=3, xlim=(4, 9), ylim=(0, 2), plot_kwargs=dict(s=60))

pa.plot_aggregate_plant_data_timeseries(
    xlim=(datetime(2013, 12, 1), datetime(2015, 12, 31)),
    ylim_energy=(0, 2),
    ylim_loss=(-0.1, 5.5)
)

# For illustrative purposes, let's suppose a few months aren't representative of long-term losses
pa.aggregate.loc['2014-11-01',
                 ['availability_typical', 'curtailment_typical']] = False
pa.aggregate.loc['2015-07-01',
                 ['availability_typical', 'curtailment_typical']] = False

# Run Monte Carlo based OA
pa.run(num_sim=2000, reanalysis_products=['era5', 'merra2'])

# Plot a distribution of AEP values from the Monte Carlo OA method
pa.plot_result_aep_distributions(
    xlim_aep=(8, 18),
    xlim_availability=(0.7, 1.3),
    xlim_curtail=(0.04, 0.09),
    ylim_aep=(0, 0.4),
    ylim_availability=(0, 9),
    ylim_curtail=(0, 120),
    annotate_kwargs={"fontsize": 12},
)

# Produce histograms of the various MC-parameters
mc_reg = pd.DataFrame(data={
    'slope': pa._mc_slope.ravel(),
    'intercept': pa._mc_intercept,
    'num_points': pa._mc_num_points,
    'metered_energy_fraction': pa.mc_inputs.metered_energy_fraction,
    'loss_fraction': pa.mc_inputs.loss_fraction,
    'num_years_windiness': pa.mc_inputs.num_years_windiness,
    'loss_threshold': pa.mc_inputs.loss_threshold,
    'reanalysis_product': pa.mc_inputs.reanalysis_product
})

plot.column_histograms(mc_reg)

# Produce scatter plots of slope and intercept values. Here we focus on the ERA-5 data
plot.set_styling()

plt.figure(figsize=(8, 6))
plt.plot(
    mc_reg.intercept[mc_reg.reanalysis_product == 'era5'],
    mc_reg.slope[mc_reg.reanalysis_product == 'era5'],
    '.', label="Monte Carlo Regression Values"
)
x = np.linspace(-2, 0, 3)
y = -0.2 * x + 0.135
plt.plot(x, y, label="y = -0.2x + 0.135")
plt.xlabel('Intercept (GWh)')
plt.ylabel('Slope (GWh / (m/s))')
plt.legend()
plt.xlim((-1.8, -0.4))
plt.ylim(0.2, 0.5)
plt.show()

pa.plot_aep_boxplot(x=mc_reg['reanalysis_product'],
                    xlabel="Reanalysis Product", ylim=(6, 18))

plot.plot_boxplot(
    y=pa.results.aep_GWh,
    x=mc_reg['num_years_windiness'],
    xlabel="Number of Years in the Windiness Correction",
    ylabel="AEP (GWh/yr)",
    ylim=(0, 20),
    plot_kwargs_box={"flierprops": dict(
        marker="x", markeredgecolor="tab:blue")}
)
fig, ax, boxes = pa.plot_aep_boxplot(
    x=mc_reg['num_years_windiness'],
    xlabel="Number of Years in the Windiness Correction",
    ylim=(0, 20),
    figure_kwargs=dict(figsize=(12, 6)),
    plot_kwargs_box={
        "flierprops": dict(marker="x", markeredgecolor="tab:blue"),
        "medianprops": dict(linewidth=1.5)
    },
    return_fig=True,
    with_points=True,
    points_label="Individual AEP Estimates",
    plot_kwargs_points=dict(alpha=0.5, s=2),
    legend_kwargs=dict(loc="lower left"),
)
