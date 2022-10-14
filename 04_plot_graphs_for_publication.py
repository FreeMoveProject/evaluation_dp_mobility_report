import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages

# tikz format used for latex file output
import tikzplotlib

import os
import config

# matplotlib style
plt.rcParams.update(plt.rcParamsDefault)
#plt.style.use("science")

# hack to fix legend bug in tikz file
custom_cycler = (cycler(color=["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#b774d6", "#474747"]) + \
        cycler(linestyle=[(0,(1,2)), (0, (3, 1, 1, 1, 1, 1)), (0, (5, 1)), '-.' ,(0,(1,1)), '-']))
plt.rc("axes", prop_cycle=custom_cycler)

##### Settings #####
if (
    (config.GEOLIFE not in config.DATASET_NAMES)
    #| (config.MADRID not in config.DATASET_NAMES)
    #| (config.BERLIN not in config.DATASET_NAMES)
):
    print(
        "Not all three data sets used in publication are present. Plots for publication will therefore be skipped."
    )

else:
    tables_path = os.path.join(config.OUTPUT_PATH, "tables")
    graphs_output_path = os.path.join(config.OUTPUT_PATH, "graphs")

    if not os.path.exists(graphs_output_path):
        os.makedirs(graphs_output_path)

    ##### Data #####
    if config.GEOLIFE in config.DATASET_NAMES:
        um_geolife = pd.read_csv(
            os.path.join(tables_path, config.GEOLIFE, config.GEOLIFE + "_mean.csv"),
            index_col="stat",
        )
        um_geolife_std = pd.read_csv(
            os.path.join(tables_path, config.GEOLIFE, config.GEOLIFE + "_std.csv"),
            index_col="stat",
        )
    if config.MADRID in config.DATASET_NAMES:
        um_madrid = pd.read_csv(
            os.path.join(tables_path, config.MADRID, config.MADRID + "_mean.csv"),
            index_col="stat",
        )
        um_madrid_std = pd.read_csv(
            os.path.join(tables_path, config.MADRID, config.MADRID + "_std.csv"),
            index_col="stat",
        )
    if config.BERLIN in config.DATASET_NAMES:
        um_tapas = pd.read_csv(
            os.path.join(tables_path, config.BERLIN, config.BERLIN + "_mean.csv"),
            index_col="stat",
        )
        um_tapas_std = pd.read_csv(
            os.path.join(tables_path, config.BERLIN, config.BERLIN + "_std.csv"),
            index_col="stat",
        )

    #### helper functions ###
    def max_trips_from_key(key):
        return key.split("mt_")[1].split("_e")[0]

    def eps_from_key(key):
        e = key.split("_e_")[1]
        return "999" if (e == "None") else e

    #############################################
    # Tables and Plots
    #############################################

    ### Table item- vs. user-level privacy
    d1 = None
    d2 = None
    d3 = None
    if config.GEOLIFE in config.DATASET_NAMES:
        d1 = um_geolife[list(um_geolife.columns)]
        d1 = d1.round(2).loc[
            [
                "n_trips",
                "counts_per_tile_emd",
                "od_flows",
                "radius_gyration_quartiles",
            ]
        ]
        multi_columns = [
            np.array(["GEOLIFE", "GEOLIFE"]),
            np.array(["user-level", "item-level"]),
        ]
        d1.columns = multi_columns
    if config.MADRID in config.DATASET_NAMES:
        d2 = um_madrid[[list(um_madrid.columns)]]
        d2 = d2.round(2).loc[
            [
                "n_trips",
                "counts_per_tile_emd",
                "od_flows",
                "radius_gyration_quartiles",
            ]
        ]
        multi_columns = [
            np.array(["MADRID", "MADRID"]),
            np.array(["user-level", "item-level"]),
        ]
        d2.columns = multi_columns
        
    if config.BERLIN in config.DATASET_NAMES:
        d3 = um_tapas[list(um_tapas.columns)]
        d3 = d3.round(2).loc[
            [
                "n_trips",
                "counts_per_tile_emd",
                "od_flows",
                "radius_gyration_quartiles",
            ]
        ]
        multi_columns = [
            np.array(["BERLIN", "BERLIN"]),
            np.array(["user-level", "item-level"]),
        ]
        d3.columns = multi_columns
    table = None
    for d in [d1,d2,d3]:
       if d is not None:
           if table is not None:
               table = pd.concat([table,d],axis=1)
           else:
               table = d
    #table = pd.concat([d1, d2, d3], axis=1)
    table.rename(
        index=dict(
            n_trips="TripCountError",
            counts_per_tile_emd="LocationError",
            od_flows="OdFlowError",
            radius_gyration_quartiles="RadiusOfGyrationError",
        ),
        inplace=True,
    )
    table.index.name = "error measure"
    print(table.to_latex())

    def df_for_single_dataset(df, metric):
        selection = df.loc[[metric],].T
        selection["pb"] = selection.index.to_series().apply(eps_from_key)
        selection["max_trips"] = selection.index.to_series().apply(max_trips_from_key)
        selection = pd.pivot_table(
            selection, index="pb", columns="max_trips", values=metric
        )
        selection.rename(index={"999": "withoutDp"}, inplace=True)
        #selection.drop("event_level", axis=1, inplace=True)
        selection.columns = selection.columns.astype(int)
        selection.sort_index(axis=1, inplace=True)
        selection.set_axis(selection.columns.astype(str), axis=1, inplace=True)
        return selection.T

    # def create_subplot(axis, df, df_error, title, y_lim_max, legend=True):
    #     for c in df.columns:
    #         axis.errorbar(
    #             df.index,
    #             df[c],
    #             df_error[c],
    #             marker="s",
    #             ms=1,
    #             capsize=3,
    #             elinewidth=0.1,
    #             linewidth=2,
    #         )
    #     axis.plot(df, label=df.columns, linewidth=2)
    #     # legend
    #     if legend:
    #         axis.legend(df.columns, ncol=6, frameon=True, bbox_to_anchor=(3, -0.3))

    #     # ylim
    #     if y_lim_max != None:
    #         axis.set_ylim(0, y_lim_max)
    #     else:
    #         axis.set_ylim(0)

    #     axis.set_title(title)
    #     return axis

    # def plot_all_datasets(
    #     d1,
    #     d2,
    #     d3,
    #     d1_error,
    #     d2_error,
    #     d3_error,
    #     metric,
    #     file_name,
    #     y_lim_max=None,
    #     sharey=False,
    # ):
    #     df1 = df_for_single_dataset(d1, metric)
    #     df1_error = df_for_single_dataset(d1_error, metric)
    #     df2 = df_for_single_dataset(d2, metric)
    #     df2_error = df_for_single_dataset(d2_error, metric)
    #     df3 = df_for_single_dataset(d3, metric)
    #     df3_error = df_for_single_dataset(d3_error, metric)
    #     fig, axes = plt.subplots(1, 3, figsize=(7, 2), sharey=sharey)
    #     axes[0] = create_subplot(
    #         axes[0], df1, df1_error, "GEOLIFE", y_lim_max, legend=False
    #     )
    #     axes[1] = create_subplot(
    #         axes[1], df2, df2_error, "MADRID", y_lim_max, legend=True
    #     )
    #     axes[2] = create_subplot(
    #         axes[2], df3, df3_error, "BERLIN", y_lim_max, legend=False
    #     )
    #     tikzplotlib.save(os.path.join(graphs_output_path, file_name + ".tex"))
    #     pp.savefig()
    #     plt.close()

    def plot_single_dataset(ds_name, data, error, metric, y_label = None, y_axis = True, y_lim_min = 0, y_lim_max = None, legend = False):
        df = df_for_single_dataset(data, metric)
        df_error = df_for_single_dataset(error, metric)
        
        fig = plt.figure(figsize = (10, 7))
        ax = fig.add_subplot(1, 1, 1)
        for c in df.columns:
            ax.errorbar(
                    df.index,
                    df[c],
                    df_error[c],
                    marker="s",
                    ms=1,
                    capsize=3,
                    elinewidth=0.1,
                    linewidth=3,
                )
        #xlabel
        ax.set_xlabel("M")
        # ylabel
        if y_label is not None:
            ax.set_ylabel(y_label)
        # yticks
        if not y_axis:
            ax.tick_params(labelleft = False)
        # ylim
        if y_lim_max != None:
            ax.set_ylim(y_lim_min, y_lim_max)
        else:
            ax.set_ylim(y_lim_min)
        
        # legend
        if legend:
            plt.plot(df, label=df.columns, linewidth=3)
            plt.legend(df.columns, ncol=6, frameon=True, bbox_to_anchor=(1.1, -0.15), handlelength=3)
        
        tikzplotlib.save(os.path.join(graphs_output_path, metric + "_" + ds_name + ".tex"))
 
        pp.savefig()
        plt.close()

    pp = PdfPages(os.path.join(graphs_output_path, "graphs_for_publication.pdf"))

    if config.GEOLIFE in config.DATASET_NAMES:
        plot_single_dataset(
                "geolife",
            um_geolife,
            um_geolife_std,
            "n_trips",
            "TripCountError",
            y_axis=True,
            y_lim_min =-1.0,
            y_lim_max=5.0,
            legend = True
        )
    if config.MADRID in config.DATASET_NAMES:
        plot_single_dataset(
            "madrid",
            um_madrid,
            um_madrid_std,
            "n_trips",
            y_axis=False,
            y_lim_min =-1.0,
            y_lim_max=5.0,
            legend = False
        )
    if config.BERLIN in config.DATASET_NAMES:
        plot_single_dataset(
            "berlin",
            um_tapas,
            um_tapas_std,
            "n_trips",
            y_axis=False,
            y_lim_min =-1.0,
            y_lim_max=5.0,
            legend = False
        )


    if config.GEOLIFE in config.DATASET_NAMES:
        plot_single_dataset(
            "geolife",
            um_geolife,
            um_geolife_std,
            "counts_per_tile_emd",
            "LocationError",
            y_axis=True,
            legend = True
        )
    if config.MADRID in config.DATASET_NAMES:
        plot_single_dataset(
            "madrid",
            um_madrid,
            um_madrid_std,
            "counts_per_tile_emd",
            y_axis=True,
            legend = False
        )
    if config.BERLIN in config.DATASET_NAMES:
        plot_single_dataset(
            "berlin",
            um_tapas,
            um_tapas_std,
            "counts_per_tile_emd",
            y_axis=True,
            legend = False
        )
    if config.GEOLIFE in config.DATASET_NAMES:
        plot_single_dataset(
            "geolife",
            um_geolife,
            um_geolife_std,
            "rel_od_flows",
            "OdFlowError",
            y_lim_min =0,
            y_lim_max=2.2,
            y_axis=False,
            legend = True
        )
    if config.MADRID in config.DATASET_NAMES:
        plot_single_dataset(
            "madrid",
            um_madrid,
            um_madrid_std,
            "rel_od_flows",
            y_lim_min =0,
            y_lim_max=2.2,
            y_axis=True,
            legend = False
        )
    if config.BERLIN in config.DATASET_NAMES:
        plot_single_dataset(
            "berlin",
            um_tapas,
            um_tapas_std,
            "rel_od_flows",
            y_lim_min =0,
            y_lim_max=2.2,
            y_axis=True,
            legend = False
        )

    # plot_all_datasets(
    #     um_geolife,
    #     um_madrid,
    #     um_tapas,
    #     um_geolife_std,
    #     um_madrid_std,
    #     um_tapas_std,
    #     "n_trips",
    #     "n_trips",
    #     y_lim_max=5.0,
    #     sharey=True,
    # )
    # plot_all_datasets(
    #     um_geolife,
    #     um_madrid,
    #     um_tapas,
    #     um_geolife_std,
    #     um_madrid_std,
    #     um_tapas_std,
    #     "counts_per_tile_emd",
    #     "visits_per_location",
    # )
    # plot_all_datasets(
    #     um_geolife,
    #     um_madrid,
    #     um_tapas,
    #     um_geolife_std,
    #     um_madrid_std,
    #     um_tapas_std,
    #     "rel_od_flows",
    #     "od_flows",
    #     y_lim_max=2.2,
    #     sharey=True,
    # )

    pp.close()
