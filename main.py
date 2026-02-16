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
import matplotlib
import gc

matplotlib.use("Agg")

app = FastAPI(title="OpenOA On-Demand Monte Carlo API")

project = None


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
# Startup: Only prepare project
# -----------------------------------
@app.on_event("startup")
def startup_event():
    global project

    print("Preparing project...")
    project = project_ENGIE.prepare(
        "./data/la_haute_borne",
        use_cleansed=False
    )
    print("Startup complete.")


# -----------------------------------
# Basic plots (lightweight)
# -----------------------------------
@app.get("/plots")
def get_basic_plots():

    plots = {}

    fig1, _ = plot.column_histograms(
        project.meter,
        columns=["MMTR_SupWh"],
        return_fig=True
    )
    plots["meter_histogram"] = fig_to_base64(fig1)

    fig2, _ = plot.column_histograms(
        project.curtail,
        columns=["IAVL_DnWh", "IAVL_ExtPwrDnWh"],
        return_fig=True
    )
    plots["curtail_histogram"] = fig_to_base64(fig2)

    return {
        "status": "success",
        "total_plots": len(plots),
        "plots": plots
    }


# -----------------------------------
# On-Demand Monte Carlo
# -----------------------------------
@app.get("/runmontecarlo")
def run_monte_carlo():

    num_sim = 1  # Always run 1 simulation

    print("Running Monte Carlo with 1 simulation...")

    pa = MonteCarloAEP(
        project,
        reanalysis_products=['era5', 'merra2']
    )

    pa.run(num_sim=num_sim)

    # AEP distribution plot
    fig, _ = pa.plot_result_aep_distributions(
        return_fig=True
    )

    plot_base64 = fig_to_base64(fig)

    # Extract summary results
    mean_aep = float(np.mean(pa.results.aep_GWh))
    p50 = float(np.percentile(pa.results.aep_GWh, 50))
    p90 = float(np.percentile(pa.results.aep_GWh, 10))

    # Clean memory
    del pa
    gc.collect()

    print("Monte Carlo complete and memory cleared.")

    return {
        "status": "success",
        "num_simulations": 1,
        "mean_aep_GWh": mean_aep,
        "p50_GWh": p50,
        "p90_GWh": p90,
        "aep_distribution_plot": plot_base64
    }


@app.get("/heartbeat")
def heartbeat():
    return {"status": "alive"}
