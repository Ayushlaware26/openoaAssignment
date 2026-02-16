import project_ENGIE
from openoa.utils import plot
from openoa.analysis.aep import MonteCarloAEP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse

import matplotlib
matplotlib.use("Agg")


app = FastAPI(title="OpenOA Full Monte Carlo API")

pa = None
project = None
mc_reg = None


# -----------------------------------
# Convert figure to base64
# -----------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img


# -----------------------------------
# Startup: Run everything once
# -----------------------------------
@app.on_event("startup")
def startup_event():
    global pa, project, mc_reg

    print("Preparing project...")
    project = project_ENGIE.prepare(
        "./data/la_haute_borne",
        use_cleansed=False
    )

    print("Initializing Monte Carlo...")
    pa = MonteCarloAEP(project, reanalysis_products=['era5', 'merra2'])

    # Mark months as non-typical
    pa.aggregate.loc['2014-11-01',
                     ['availability_typical', 'curtailment_typical']] = False
    pa.aggregate.loc['2015-07-01',
                     ['availability_typical', 'curtailment_typical']] = False

    print("Running Monte Carlo simulation...")
    pa.run(num_sim=10)

    # Build mc_reg dataframe
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

    print("Startup complete.")


# -----------------------------------
# Endpoint returning ALL plots
# -----------------------------------
@app.get("/plots")
def get_all_plots():

    plots = {}

    # 1️⃣ Meter Histogram
    fig1, _ = plot.column_histograms(
        project.meter,
        columns=["MMTR_SupWh"],
        return_fig=True
    )
    plots["meter_histogram"] = fig_to_base64(fig1)

    # 2️⃣ Curtail Histogram
    fig2, _ = plot.column_histograms(
        project.curtail,
        columns=["IAVL_DnWh", "IAVL_ExtPwrDnWh"],
        return_fig=True
    )
    plots["curtail_histogram"] = fig_to_base64(fig2)

    # 3️⃣ Normalized Monthly Windspeed
    fig3, _ = pa.plot_normalized_monthly_reanalysis_windspeed(
        return_fig=True,
        xlim=(datetime(1996, 1, 1), datetime(2021, 12, 31)),
        ylim=(0.8, 1.2),
    )
    plots["normalized_monthly_windspeed"] = fig_to_base64(fig3)

    # 4️⃣ Gross Energy Data
    fig4, _ = pa.plot_reanalysis_gross_energy_data(
        return_fig=True,
        outlier_threshold=3,
        xlim=(4, 9),
        ylim=(0, 2),
        plot_kwargs=dict(s=60)
    )
    plots["gross_energy_data"] = fig_to_base64(fig4)

    # 5️⃣ Aggregate Plant Data Timeseries
    fig5, _ = pa.plot_aggregate_plant_data_timeseries(
        return_fig=True,
        xlim=(datetime(2013, 12, 1), datetime(2015, 12, 31)),
        ylim_energy=(0, 2),
        ylim_loss=(-0.1, 5.5)
    )
    plots["aggregate_timeseries"] = fig_to_base64(fig5)

    # 6️⃣ AEP Distribution
    fig6, _ = pa.plot_result_aep_distributions(
        return_fig=True,
        xlim_aep=(8, 18),
        xlim_availability=(0.7, 1.3),
        xlim_curtail=(0.04, 0.09),
        ylim_aep=(0, 0.4),
        ylim_availability=(0, 9),
        ylim_curtail=(0, 120),
        annotate_kwargs={"fontsize": 12},
    )
    plots["aep_distribution"] = fig_to_base64(fig6)

    # 7️⃣ Monte Carlo Parameter Histograms
    fig7, _ = plot.column_histograms(
        mc_reg,
        return_fig=True
    )
    plots["mc_parameter_histograms"] = fig_to_base64(fig7)

    # 8️⃣ Intercept vs Slope Scatter
    plot.set_styling()
    fig8 = plt.figure(figsize=(8, 6))
    plt.plot(
        mc_reg.intercept[mc_reg.reanalysis_product == 'era5'],
        mc_reg.slope[mc_reg.reanalysis_product == 'era5'],
        '.',
        label="Monte Carlo Regression Values"
    )
    x = np.linspace(-2, 0, 3)
    y = -0.2 * x + 0.135
    plt.plot(x, y, label="y = -0.2x + 0.135")
    plt.xlabel('Intercept (GWh)')
    plt.ylabel('Slope (GWh / (m/s))')
    plt.legend()
    plt.xlim((-1.8, -0.4))
    plt.ylim(0.2, 0.5)

    plots["intercept_vs_slope_scatter"] = fig_to_base64(fig8)

    # 9️⃣ AEP Boxplot by Reanalysis Product
    fig9, _, _ = pa.plot_aep_boxplot(
        x=mc_reg['reanalysis_product'],
        xlabel="Reanalysis Product",
        ylim=(6, 18),
        return_fig=True
    )
    plots["aep_boxplot_reanalysis"] = fig_to_base64(fig9)

    # 🔟 AEP vs Num Years Windiness
    fig10, _, _ = plot.plot_boxplot(
        y=pa.results.aep_GWh,
        x=mc_reg['num_years_windiness'],
        xlabel="Number of Years in the Windiness Correction",
        ylabel="AEP (GWh/yr)",
        ylim=(0, 20),
        return_fig=True
    )
    plots["aep_vs_years_windiness"] = fig_to_base64(fig10)

    # 1️⃣1️⃣ Enhanced AEP Boxplot
    fig11, _, _ = pa.plot_aep_boxplot(
        x=mc_reg['num_years_windiness'],
        xlabel="Number of Years in the Windiness Correction",
        ylim=(0, 20),
        return_fig=True,
        with_points=True,
        points_label="Individual AEP Estimates"
    )
    plots["enhanced_aep_boxplot"] = fig_to_base64(fig11)

    return {
        "status": "success",
        "total_plots": len(plots),
        "plot_names": list(plots.keys()),
        "plots": plots
    }


@app.get("/heartbeat")
def heartbeat():
    return {"status": "alive"}
