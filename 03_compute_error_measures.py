import os
from tqdm.auto import tqdm

import shelve
import pandas as pd
import numpy as np
import geopandas as gpd
import time

import config
import error_measures as em


def key(max_trips, eps, rep=None):
    if rep is None:
        return "mt_" + str(max_trips) + "_e_" + str(eps)
    else:
        return "mt_" + str(max_trips) + "_e_" + str(eps) + "_rep_" + str(rep)


for dataset_name in config.DATASET_NAMES:
    ds_report_path = os.path.join(config.REPORT_PATH, dataset_name)
    df_output_path = os.path.join(config.OUTPUT_PATH, "tables", dataset_name)
    tessellation = gpd.read_file(
        os.path.join(config.PROCESSED_DATA_PATH, dataset_name + "_tessellation.gpkg"),
        dtype={"tile_id": str},
    )

    # get config info
    d = shelve.open(os.path.join(ds_report_path, "config"))
    max_trips_array = sorted(d["max_trips"])
    privacy_budgets = d["eps"]
    reps = d["reps"]
    d.close()
    total_combinations = len(max_trips_array) * len(privacy_budgets)
    with tqdm(
        total=total_combinations, desc="Compute error measures for: " + dataset_name
    ) as pbar:  # progress bar
        # if export folder does not exist, create the folder
        if not os.path.exists(df_output_path):
            os.makedirs(df_output_path)

        error_measures = pd.DataFrame()
        error_measures_avg = pd.DataFrame()
        error_measures_std = pd.DataFrame()
        
        # get baseline report
        d = shelve.open(
            os.path.join(
                ds_report_path,
                "maxTrips_" + str(max(max_trips_array)) + "_eps_None",
            )
        )
        report_true = d[str(0)]
        d.close()

        # speed up compuation by only computing cost_matrix once
        tile_centroids = (
            tessellation.set_index("tile_id").to_crs(3395).centroid.to_crs(4326)
        )
        sorted_tile_centroids = tile_centroids.loc[
            report_true["counts_per_tile_section"].data.tile_id
        ]
        tile_coords = list(zip(sorted_tile_centroids.y, sorted_tile_centroids.x))
        cost_matrix = em._get_cost_matrix(tile_coords)

        # compute error measures
        for max_trips in max_trips_array:
            if max_trips == -99:
                max_trips = "event_level"
            for pb in privacy_budgets:
                d = shelve.open(
                    os.path.join(
                        ds_report_path,
                        "maxTrips_" + str(max_trips) + "_eps_" + str(pb),
                    )
                )
                for i in range(0, reps):
                    if str(i) in list(d.keys()):
                        error_measures[key(max_trips, pb, i)] = pd.Series(
                            em.compute_error_measures(
                                report_true, d[str(i)], tessellation, cost_matrix
                            )
                        ).round(3)
                error_measures_avg[key(max_trips, pb)] = (
                    error_measures.loc[
                        :, error_measures.columns.str.startswith(key(max_trips, pb))
                    ]
                    .T.mean()
                    .round(3)
                )
                error_measures_std[key(max_trips, pb)] = (
                    error_measures.loc[
                        :, error_measures.columns.str.startswith(key(max_trips, pb))
                    ]
                    .T.std()
                    .round(3)
                )
                d.close()
                pbar.update()

        error_measures.to_csv(
            os.path.join(config.df_output_path, dataset_name + "_all_reps.csv"),
            index_label="stat",
        )
        error_measures_avg.to_csv(
            os.path.join(config.df_output_path, dataset_name + "_mean.csv"),
            index_label="stat",
        )
        error_measures_std.to_csv(
            os.path.join(config.df_output_path, dataset_name + "_std.csv"),
            index_label="stat",
        )
