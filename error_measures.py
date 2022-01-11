import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance

import cv2
from haversine import haversine, Unit


def _moving_average(arr, size):
    return np.convolve(arr, np.ones(size), "valid") / size


def wasserstein_distance1D(hist1, hist2):
    u_values = _moving_average(hist1[1], 2)
    v_values = _moving_average(hist2[1], 2)
    u_weights = hist1[0]
    v_weights = hist2[0]
    if (sum(u_weights) == 0) | (sum(v_weights) == 0):
        return None
    return wasserstein_distance(u_values, v_values, u_weights, v_weights)


def symmetric_mape(true, estimate, n_true_positive_zeros=None):
    n = (
        len(true)
        if n_true_positive_zeros is None
        else (len(true) + n_true_positive_zeros)
    )
    return (
        1
        / n
        * np.sum(
            np.where(
                abs(true + estimate) == 0,
                0,  # return 0 if true and estimate are both 0
                np.divide(
                    abs(estimate - true),
                    ((abs(true) + abs(estimate)) / 2),
                    where=(abs(true + estimate) != 0),
                ),
            )
        )
    )


def rel_error(true, estimate):
    if estimate == None:
        estimate = 0
    if true == 0:
        # we can return the absolute error
        return np.abs(true - estimate)
        # or Relative Percent Difference
        # return(2*(estimate-true)/(np.abs(estimate)+np.abs(true)))
    return np.abs(true - estimate) / true


def rel_error_dict(true_dict, estimate_dict, round_res=False):
    re = dict()
    for key in true_dict:
        re[key] = rel_error(true_dict[key], estimate_dict[key])
        if round_res:
            re[key] = round(re[key], 2)
    return re


def get_prob(a):
    return a / sum(a)


def KL(a, b):
    return np.sum(
        np.where(a != 0, a * np.log(a / b, where=a != 0), 0)
    )  # double where to suppress warning


def compute_KL(p, q):
    p = get_prob(np.asarray(p, dtype=np.float))
    q = get_prob(np.asarray(q, dtype=np.float))
    return KL(p, q)


def compute_JS(p, q):
    p = get_prob(np.asarray(p, dtype=np.float))
    q = get_prob(np.asarray(q, dtype=np.float))
    m = (1.0 / 2.0) * (p + q)
    return (1.0 / 2.0) * KL(p, m) + (1.0 / 2.0) * KL(q, m)


### earth movers distance
def _get_cost_matrix(tile_coords):
    # get all potential combinations between all points from sig1 and sig2
    grid = np.meshgrid(range(0, len(tile_coords)), range(0, len(tile_coords)))
    tile_combinations = np.array([grid[0].flatten(), grid[1].flatten()])

    # create an empty cost matrix with the length of all possible combinations
    cost_matrix = np.empty(
        tile_combinations.shape[1], dtype=np.float32
    )  # float32 needed as input for cv2.emd!

    # compute haversine distance for all possible combinations
    for column in range(0, tile_combinations.shape[1]):
        tile_1 = tile_combinations[0, column]
        tile_2 = tile_combinations[1, column]
        cost_matrix[column] = haversine(
            tile_coords[tile_1], tile_coords[tile_2], unit=Unit.METERS
        )

    # reshape array to matrix
    return np.reshape(cost_matrix, (len(tile_coords), len(tile_coords)))


def earth_movers_distance(
    arr_true, arr_estimate, cost_matrix
):  # based on haversine distance
    # normalize input and assign needed type for cv2
    arr_true = (arr_true / arr_true.sum() * 100).round(2)
    sig_true = arr_true.astype(np.float32)
    if all(
        arr_estimate == 0
    ):  # if all values are 0, change all to 1 (otherwise emd cannot be computed)
        arr_estimate = np.repeat(1, len(arr_estimate))
    arr_estimate = (arr_estimate / arr_estimate.sum() * 100).round(2)
    sig_estimate = arr_estimate.astype(np.float32)

    emd_dist, _, _ = cv2.EMD(
        sig_true, sig_estimate, distType=cv2.DIST_USER, cost=cost_matrix
    )
    return emd_dist


def compute_error_measures(
    report_true, report_estimate, tessellation, cost_matrix=None
):
    error_measures = dict()

    ### overview ###
    error_measures = dict(
        **error_measures,
        **rel_error_dict(
            report_true["ds_statistics"],
            report_estimate["ds_statistics"],
            round_res=True,
        )
    )
    error_measures = dict(
        **error_measures,
        **rel_error_dict(
            report_true["missing_values"],
            report_estimate["missing_values"],
            round_res=True,
        )
    )
    trips_over_time = report_true["trips_over_time_section"].data.merge(
        report_estimate["trips_over_time_section"].data,
        how="outer",
        on="datetime",
        suffixes=("_true", "_estimate"),
    )
    trips_over_time.fillna(0, inplace=True)
    # error_measures["trips_over_time_js"] = compute_JS(trips_over_time.trip_count_true, trips_over_time.trip_count_estimate)
    error_measures["trips_over_time_mre"] = symmetric_mape(
        trips_over_time.trip_count_true, trips_over_time.trip_count_estimate
    )
    error_measures["trips_over_time_quartiles"] = symmetric_mape(
        report_true["trips_over_time_section"].quartiles.apply(lambda x: x.toordinal()),
        report_estimate["trips_over_time_section"].quartiles.apply(
            lambda x: x.toordinal()
        ),
    )

    trips_per_weekday = pd.concat(
        [report_true["trips_per_weekday"], report_estimate["trips_per_weekday"]],
        join="outer",
        axis=1,
    )
    trips_per_weekday.fillna(0, inplace=True)
    error_measures["trips_per_weekday"] = symmetric_mape(
        trips_per_weekday.iloc[:, 0], trips_per_weekday.iloc[:, 1]
    )

    trips_per_hour = report_true["trips_per_hour"].merge(
        report_estimate["trips_per_hour"],
        how="outer",
        on=["hour", "time_category"],
        suffixes=("_true", "_estimate"),
    )
    trips_per_hour.fillna(0, inplace=True)
    error_measures["trips_per_hour"] = symmetric_mape(
        trips_per_hour.count_true, trips_per_hour.count_estimate
    )

    ### place
    counts_per_tile = report_true["counts_per_tile_section"].data.merge(
        report_estimate["counts_per_tile_section"].data,
        how="outer",
        on="tile_id",
        suffixes=("_true", "_estimate"),
    )
    counts_per_tile.fillna(0, inplace=True)

    rel_counts_true = (
        counts_per_tile.visit_count_true / counts_per_tile.visit_count_true.sum()
    )
    rel_counts_estimate = (
        counts_per_tile.visit_count_estimate
        / counts_per_tile.visit_count_estimate.sum()
    )
    error_measures["counts_per_tile_smape"] = symmetric_mape(
        counts_per_tile.visit_count_true, counts_per_tile.visit_count_estimate
    )
    error_measures["rel_counts_per_tile_smape"] = symmetric_mape(
        rel_counts_true, rel_counts_estimate
    )

    # speed up evaluation: cost_matrix as input so it does not have to be recomputed every time
    if cost_matrix is None:
        tile_centroids = (
            tessellation.set_index("tile_id").to_crs(3395).centroid.to_crs(4326)
        )
        sorted_tile_centroids = tile_centroids.loc[counts_per_tile.tile_id]
        tile_coords = list(zip(sorted_tile_centroids.y, sorted_tile_centroids.x))
        # create custom cost matrix with distances between all tiles
        cost_matrix = _get_cost_matrix(tile_coords)
    error_measures["counts_per_tile_emd"] = earth_movers_distance(
        counts_per_tile.visit_count_true.to_numpy(),
        counts_per_tile.visit_count_estimate.to_numpy(),
        cost_matrix,
    )
    error_measures["counts_per_tile_outliers"] = rel_error(
        report_true["counts_per_tile_section"].n_outliers,
        report_estimate["counts_per_tile_section"].n_outliers,
    )
    error_measures["counts_per_tile_quartiles"] = symmetric_mape(
        report_true["counts_per_tile_section"].quartiles,
        report_estimate["counts_per_tile_section"].quartiles,
    )

    ## tile counts per timewindow
    counts_per_tile_timewindow_emd = []

    for c in report_true["counts_per_tile_timewindow"].columns:
        tw_true = report_true["counts_per_tile_timewindow"][c].loc[
            report_true["counts_per_tile_section"].data.tile_id
        ]  # sort accordingly for cost_matrix
        tw_true = tw_true / tw_true.sum()
        if c not in report_estimate["counts_per_tile_timewindow"].columns:
            tw_estimate = tw_true.copy()
            tw_estimate[:] = 0
        else:
            tw_estimate = report_estimate["counts_per_tile_timewindow"][c].loc[
                report_true["counts_per_tile_section"].data.tile_id
            ]
            tw_estimate = tw_estimate / tw_estimate.sum()
        tw = pd.merge(
            tw_true,
            tw_estimate,
            how="outer",
            right_index=True,
            left_index=True,
            suffixes=("_true", "_estimate"),
        )
        tw = tw[tw.notna().sum(axis=1) > 0]  # remove instances where both are NaN
        tw.fillna(0, inplace=True)
        counts_per_tile_timewindow_emd.append(
            earth_movers_distance(
                tw.iloc[:, 0].to_numpy(), tw.iloc[:, 1].to_numpy(), cost_matrix
            )
        )

    error_measures["counts_per_tile_timewindow_emd"] = np.mean(
        counts_per_tile_timewindow_emd
    )

    counts_timew_true = report_true["counts_per_tile_timewindow"][
        report_true["counts_per_tile_timewindow"].index != "None"
    ].unstack()
    counts_timew_estimate = report_estimate["counts_per_tile_timewindow"][
        report_estimate["counts_per_tile_timewindow"].index != "None"
    ].unstack()

    indices = np.unique(
        np.append(counts_timew_true.index.values, counts_timew_estimate.index.values)
    )

    counts_timew_true = counts_timew_true.reindex(index=indices)
    counts_timew_true.fillna(0, inplace=True)

    counts_timew_estimate = counts_timew_estimate.reindex(index=indices)
    counts_timew_estimate.fillna(0, inplace=True)

    rel_counts_timew_true = counts_timew_true / counts_timew_true.sum()
    rel_counts_timew_estimate = counts_timew_estimate / counts_timew_estimate.sum()

    error_measures["counts_per_tile_timewindow"] = symmetric_mape(
        counts_timew_true.to_numpy().flatten(),
        counts_timew_estimate.to_numpy().flatten(),
    )

    error_measures["rel_counts_per_tile_timewindow"] = symmetric_mape(
        rel_counts_timew_true.to_numpy().flatten(),
        rel_counts_timew_estimate.to_numpy().flatten(),
    )

    ### od
    all_od_combinations = pd.concat(
        [
            report_true["od_flows"][["origin", "destination"]],
            report_estimate["od_flows"][["origin", "destination"]],
        ]
    ).drop_duplicates()
    all_od_combinations["flow"] = 0
    n_true_positive_zeros = len(tessellation) ** 2 - len(all_od_combinations)

    true = (
        pd.concat([report_true["od_flows"], all_od_combinations])
        .drop_duplicates(["origin", "destination"], keep="first")
        .sort_values(["origin", "destination"])
        .flow
    )
    estimate = (
        pd.concat([report_estimate["od_flows"], all_od_combinations])
        .drop_duplicates(["origin", "destination"], keep="first")
        .sort_values(["origin", "destination"])
        .flow
    )

    rel_true = true / true.sum()
    rel_estimate = estimate / (estimate.sum())

    error_measures["od_flows"] = symmetric_mape(true.to_numpy(), estimate.to_numpy())
    error_measures["rel_od_flows"] = symmetric_mape(
        rel_true.to_numpy(), rel_estimate.to_numpy()
    )
    error_measures["od_flows_all_flows"] = symmetric_mape(
        true.to_numpy(), estimate.to_numpy(), n_true_positive_zeros
    )
    error_measures["rel_od_flows_all_flows"] = symmetric_mape(
        rel_true.to_numpy(), rel_estimate.to_numpy(), n_true_positive_zeros
    )
    error_measures["travel_time_emd"] = wasserstein_distance1D(
        report_true["travel_time_section"].data,
        report_estimate["travel_time_section"].data,
    )
    error_measures["travel_time_outliers"] = rel_error(
        report_true["travel_time_section"].n_outliers,
        report_estimate["travel_time_section"].n_outliers,
    )
    error_measures["travel_time_quartiles"] = symmetric_mape(
        report_true["travel_time_section"].quartiles,
        report_estimate["travel_time_section"].quartiles,
    )
    error_measures["jump_length_emd"] = wasserstein_distance1D(
        report_true["jump_length_section"].data,
        report_estimate["jump_length_section"].data,
    )
    error_measures["jump_length_outliers"] = rel_error(
        report_true["jump_length_section"].n_outliers,
        report_estimate["jump_length_section"].n_outliers,
    )
    error_measures["jump_length_quartiles"] = symmetric_mape(
        report_true["jump_length_section"].quartiles,
        report_estimate["jump_length_section"].quartiles,
    )

    ## user
    if report_estimate["traj_per_user_section"] is None:
        error_measures["traj_per_user_quartiles"] = None
        error_measures["traj_per_user_outliers"] = rel_error(
            report_true["traj_per_user_section"].n_outliers, 0
        )
    else:
        error_measures["traj_per_user_quartiles"] = symmetric_mape(
            report_true["traj_per_user_section"].quartiles,
            report_estimate["traj_per_user_section"].quartiles,
        )
        error_measures["traj_per_user_outliers"] = rel_error(
            report_true["traj_per_user_section"].n_outliers,
            report_estimate["traj_per_user_section"].n_outliers,
        )
    if report_estimate["user_time_delta_section"] is None:
        error_measures["user_time_delta_quartiles"] = None
        error_measures["user_time_delta_outliers"] = rel_error(
            report_true["user_time_delta_section"].n_outliers, 0
        )
    else:
        error_measures["user_time_delta_quartiles"] = symmetric_mape(
            (
                report_true["user_time_delta_section"].quartiles.apply(
                    lambda x: x.total_seconds() / 3600
                )
            ),
            report_estimate["user_time_delta_section"].quartiles.apply(
                lambda x: x.total_seconds() / 3600
            ),
        )
        error_measures["user_time_delta_outliers"] = rel_error(
            report_true["user_time_delta_section"].n_outliers,
            report_estimate["user_time_delta_section"].n_outliers,
        )
    error_measures["radius_gyration_emd"] = wasserstein_distance1D(
        report_true["radius_gyration_section"].data,
        report_estimate["radius_gyration_section"].data,
    )
    error_measures["radius_gyration_quartiles"] = symmetric_mape(
        report_true["radius_gyration_section"].quartiles,
        report_estimate["radius_gyration_section"].quartiles,
    )
    error_measures["radius_gyration_outliers"] = rel_error(
        report_true["radius_gyration_section"].n_outliers,
        report_estimate["radius_gyration_section"].n_outliers,
    )
    loc_entropy_per_tile = pd.DataFrame(report_true["location_entropy_section"]).merge(
        report_estimate["location_entropy_section"],
        how="outer",
        on="tile_id",
        suffixes=("_true", "_estimate"),
    )
    loc_entropy_per_tile.fillna(0, inplace=True)
    error_measures["location_entropy_mre"] = symmetric_mape(
        loc_entropy_per_tile.location_entropy_true,
        loc_entropy_per_tile.location_entropy_estimate,
    )
    error_measures["user_tile_count_emd"] = wasserstein_distance1D(
        report_true["user_tile_count_section"].data,
        report_estimate["user_tile_count_section"].data,
    )
    error_measures["user_tile_count_quartiles"] = symmetric_mape(
        report_true["user_tile_count_section"].quartiles,
        report_estimate["user_tile_count_section"].quartiles,
    )
    error_measures["uncorrelated_entropy_emd"] = wasserstein_distance1D(
        report_true["uncorrelated_entropy_section"].data,
        report_estimate["uncorrelated_entropy_section"].data,
    )
    error_measures["uncorrelated_entropy_quartiles"] = symmetric_mape(
        report_true["uncorrelated_entropy_section"].quartiles,
        report_estimate["uncorrelated_entropy_section"].quartiles,
    )
    return error_measures
