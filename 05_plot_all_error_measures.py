import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages

import os
import config

# matplotlib style
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use("science")

# hack to fix legend bug in tikz file
color_cycler = cycler(
    color=["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747"]
)
plt.rc("axes", prop_cycle=color_cycler)

##### Settings #####

tables_path = os.path.join(config.OUTPUT_PATH, "tables")
graphs_output_path = os.path.join(config.OUTPUT_PATH, "graphs")


if not os.path.exists(graphs_output_path):
    os.makedirs(graphs_output_path)

##### Data #####

n_datasets = len(config.DATASET_NAMES) if len(config.DATASET_NAMES) <= 3 else 3

if config.GEOLIFE in config.DATASET_NAMES:
    em_geolife = pd.read_csv(
        os.path.join(tables_path, config.GEOLIFE, config.GEOLIFE + "_mean.csv"),
        index_col="stat",
    )
    em_geolife_std = pd.read_csv(
        os.path.join(tables_path, config.GEOLIFE, config.GEOLIFE + "_std.csv"),
        index_col="stat",
    )
    measures = em_geolife.index
else:
    em_geolife = None
    em_geolife_std = None

if config.MADRID in config.DATASET_NAMES:
    em_madrid = pd.read_csv(
        os.path.join(tables_path, config.MADRID, config.MADRID + "_mean.csv"),
        index_col="stat",
    )
    em_madrid_std = pd.read_csv(
        os.path.join(tables_path, config.MADRID, config.MADRID + "_std.csv"),
        index_col="stat",
    )
    measures = em_madrid.index
else:
    em_madrid = None
    em_madrid_std = None

if config.BERLIN in config.DATASET_NAMES:
    em_berlin = pd.read_csv(
        os.path.join(tables_path, config.BERLIN, config.BERLIN + "_mean.csv"),
        index_col="stat",
    )
    em_berlin_std = pd.read_csv(
        os.path.join(tables_path, config.BERLIN, config.BERLIN + "_std.csv"),
        index_col="stat",
    )
    measures = em_berlin.index
else:
    em_berlin = None
    em_berlin_std = None

#### helper functions ###
def max_trips_from_key(key):
    return key.split("mt_")[1].split("_e")[0]


def eps_from_key(key):
    e = key.split("_e_")[1]
    return "999" if (e == "None") else e


#############################################
# Tables and Plots
#############################################


def df_for_single_dataset(df, measure):
    selection = df.loc[[measure],].T
    selection["pb"] = selection.index.to_series().apply(eps_from_key)
    selection["max_trips"] = selection.index.to_series().apply(max_trips_from_key)
    selection = pd.pivot_table(
        selection, index="pb", columns="max_trips", values=measure
    )
    selection.rename(index={"999": "withoutDp"}, inplace=True)
    selection.drop("event_level", axis=1, inplace=True)
    selection.columns = selection.columns.astype(int)
    selection.sort_index(axis=1, inplace=True)
    selection.set_axis(selection.columns.astype(str), axis=1, inplace=True)
    return selection.T


def create_subplot(axis, df, df_error, title, y_lim_max, legend=True):
    for c in df.columns:
        axis.errorbar(
            df.index,
            df[c],
            df_error[c],
            marker="s",
            ms=1,
            capsize=3,
            elinewidth=0.1,
            linewidth=2,
        )
    axis.plot(df, label=df.columns)
    # legend
    if legend:
        axis.legend(df.columns, ncol=3, frameon=True, bbox_to_anchor=(0.8, -0.3))

    # ylim
    if y_lim_max != None:
        axis.set_ylim(0, y_lim_max)
    else:
        axis.set_ylim(0)

    axis.set_title(title)
    return axis


def plot_all_datasets(
    d1, d2, d3, d1_error, d2_error, d3_error, measure, y_lim_max=None, sharey=False,
):
    fig, axes = plt.subplots(1, n_datasets, figsize=(7, 2), sharey=sharey)
    n_axis = 0
    if n_datasets == 1:
        ax = axes
    else:
        ax = axes[n_axis]

    if d1 is not None:
        df1 = df_for_single_dataset(d1, measure)
        df1_error = df_for_single_dataset(d1_error, measure)
        ax = create_subplot(ax, df1, df1_error, "GEOLIFE", y_lim_max, legend=False)
        n_axis += 1
        if n_axis < n_datasets:
            ax = axes[n_axis]

    if d2 is not None:
        df2 = df_for_single_dataset(d2, measure)
        df2_error = df_for_single_dataset(d2_error, measure)
        ax = create_subplot(ax, df2, df2_error, "MADRID", y_lim_max, legend=True)
        n_axis += 1
        if n_axis < n_datasets:
            ax = axes[n_axis]
    if d3 is not None:
        df3 = df_for_single_dataset(d3, measure)
        df3_error = df_for_single_dataset(d3_error, measure)
        ax = create_subplot(ax, df3, df3_error, "BERLIN", y_lim_max, legend=False)
    fig.suptitle(measure, y=1.2)
    pp.savefig()
    plt.close()


pp = PdfPages(os.path.join(graphs_output_path, "graphs_all_error_measures.pdf"))

for measure in measures:
    plot_all_datasets(
        em_geolife,
        em_madrid,
        em_berlin,
        em_geolife_std,
        em_madrid_std,
        em_berlin_std,
        measure,
    )

pp.close()
