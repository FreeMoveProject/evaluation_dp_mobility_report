import os
from pathlib import Path
import random
import csv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from skmob import tessellation
from skmob.tessellation import tilers
import requests
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from tqdm.auto import tqdm


import config

raw_data_path = "data/raw/"
preprocessed_data_path = "data/processed/"

if not os.path.exists(raw_data_path):
    os.makedirs(raw_data_path)
if not os.path.exists(preprocessed_data_path):
    os.makedirs(preprocessed_data_path)


############ Download data ###############
# GEOLIFE
if config.GEOLIFE not in config.DATASET_NAMES:
    print("Geolife not selected in config. Download is skipped.")

elif not os.path.exists(os.path.join(raw_data_path, config.GEOLIFE)):
    with tqdm(total=1, desc="Download geolife data",) as pbar:  # progress bar

        os.makedirs(os.path.join(raw_data_path, config.GEOLIFE))

        url = "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(
                    os.path.join(raw_data_path, config.GEOLIFE)
                )
        pbar.update()
else:
    print("Geolife data already exists. Download is skipped.")

# MADRID
if config.MADRID not in config.DATASET_NAMES:
    print("Madrid not selected in config. Download is skipped.")
elif not os.path.exists(os.path.join(raw_data_path, config.MADRID)):
    with tqdm(total=3, desc="Download madrid data",) as pbar:  # progress bar

        os.makedirs(os.path.join(raw_data_path, config.MADRID))

        # INDIVIDUOS
        url = "https://crtm.maps.arcgis.com/sharing/rest/content/items/07dad41b543641d3964a68851fc9ad11/data"
        r = requests.get(url, allow_redirects=True)
        output = open(
            os.path.join(raw_data_path, "madrid/EDM2018INDIVIDUOS.xlsx"), "wb"
        )
        output.write(r.content)
        output.close()
        pbar.update()

        # VIAJES
        url = "https://crtm.maps.arcgis.com/sharing/rest/content/items/6afd4db8175d4902ada0803f08ccf50e/data"
        r = requests.get(url, allow_redirects=True)
        output = open(
            os.path.join(raw_data_path, "madrid/EDM2018VIAJES.xlsx"), "wb"
        )
        output.write(r.content)
        output.close()
        pbar.update()

        url = "https://opendata.arcgis.com/api/v3/datasets/97f83bab03664d4e9853acf0e431d893_0/downloads/data?format=shp&spatialRefId=3857"

        with urlopen(url) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(
                    os.path.join(raw_data_path, "madrid/ZonificacionZT1259-shp")
                )
        pbar.update()
else:
    print("Madrid data already exists. Download is skipped.")

########### Preprocess Geolife data ###############

# clean header of plt files and write all data into single csv
def geolife_clean_plt(root, user_id, input_filepath, traj_id):
    # read plt file
    with open(root + "/" + user_id + "/Trajectory/" + input_filepath, "rt") as fin:
        cr = csv.reader(fin)
        filecontents = [line for line in cr][6:]
        for l in filecontents:
            l.insert(0, traj_id)
            l.insert(0, user_id)
    return [
        filecontents[0],
        filecontents[-1],
    ]  # only return first and last item (start and end)
    # return filecontents


def geolife_data_to_df(dir):
    data = []
    col_names = ["uid", "tid", "lat", "lng", "-", "Alt", "dayNo", "date", "time"]
    user_id_dirs = [
        name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))
    ]
    for user_id in np.sort(user_id_dirs):
        tempdirs = os.listdir(dir + "/" + user_id + "/Trajectory")
        subdirs = []
        for item in tempdirs:
            if not item.endswith(".DS_Store"):
                subdirs.append(item)
        traj_id = 0
        for subdir in subdirs:
            data += geolife_clean_plt(dir, user_id, subdir, traj_id)
            traj_id = traj_id + 1
        pbar.update()
    return pd.DataFrame(data, columns=col_names)


## script ##
# GEOLIFE
if config.GEOLIFE not in config.DATASET_NAMES:
    print("Geolife not selected in config. Preprocessing is skipped.")

else:

    if Path(os.path.join(preprocessed_data_path, "geolife.csv")).exists():
        print("Geolife data is already preprocessed. Processing is skipped.")
    else:
        with tqdm(total=184, desc="Preprocess Geolife data",) as pbar:  # progress bar
            geolife_dir = os.path.join(
                raw_data_path, config.GEOLIFE, "Geolife Trajectories 1.3", "Data"
            )
            df = geolife_data_to_df(geolife_dir)

            # make sure start and end are not duplicate
            df.drop_duplicates(inplace=True)

            df["datetime"] = df.date + " " + df.time
            df["datetime"] = pd.to_datetime(df.datetime)

            ## fix datetime timezone
            df["datetime"] = (
                df["datetime"]
                .dt.tz_localize("GMT")
                .dt.tz_convert("Asia/Shanghai")
                .dt.tz_localize(None)
            )

            print(df.shape)
            df.to_csv(os.path.join(preprocessed_data_path, "geolife.csv"), index=False)
            pbar.update()

            ### create tessellation
            # set boundaries of tessellation
            polygon_geom = Polygon(
                zip([116.08, 116.69, 116.69, 116.08], [39.66, 39.66, 40.27, 40.27])
            )
            base_shape = gpd.GeoDataFrame(index=[0], crs=4326, geometry=[polygon_geom])
            tessellation = tilers.tiler.get(
                "h3_tessellation", base_shape=base_shape, meters=2000
            )
            tessellation.rename(columns=dict(tile_ID="tile_id"), inplace=True)
            tessellation.to_file(
                os.path.join(preprocessed_data_path, "geolife_tessellation.gpkg"),
                driver="GPKG",
            )
            pbar.update()

###### Preprocess Madrid Data ######
def madrid_preprocess_to_csv(dir):

    dir_ind = dir + "EDM2018INDIVIDUOS.xlsx"
    dir_via = dir + "EDM2018VIAJES.xlsx"
    ind = pd.read_excel(dir_ind, engine="openpyxl")
    via = pd.read_excel(
        dir_via, dtype={"VORIHORAINI": str, "VDESHORAFIN": str}, engine="openpyxl",
    )

    ind.rename(columns={"DMES": "MONTH", "DDIA": "DAY"}, inplace=True)

    ind["ref_date"] = pd.to_datetime(ind.assign(YEAR=2018)[["YEAR", "MONTH", "DAY"]])
    via["start_time"] = pd.to_datetime(via.VORIHORAINI, format="%H%M").dt.time

    via.set_index(["ID_HOGAR", "ID_IND"], inplace=True)
    ind.set_index(["ID_HOGAR", "ID_IND"], inplace=True)

    ind = ind.join(via, rsuffix="_via", how="inner")

    ind["date_arrival"] = ind.ref_date

    ind.loc[ind.VDESHORAFIN.astype(int) >= 2400, "ref_date"] = ind.loc[
        ind.VDESHORAFIN.astype(int) >= 2400, "ref_date"
    ] + timedelta(days=1)
    ind.loc[ind.VDESHORAFIN.astype(int) >= 2400, "VDESHORAFIN"] = (
        ind.loc[ind.VDESHORAFIN.astype(int) >= 2400, "VDESHORAFIN"].astype(int) - 2400
    )
    ind["VDESHORAFIN"] = ind.VDESHORAFIN.astype(str).apply(lambda x: x.zfill(4))

    ind["end_time"] = pd.to_datetime(ind.VDESHORAFIN, format="%H%M").dt.time

    ind.reset_index(inplace=True)
    ind["uid"] = ind.ID_HOGAR.astype(str) + "_" + ind.ID_IND.astype(str)

    ind["tid"] = (
        ind.ID_HOGAR.astype(str)
        + "_"
        + ind.ID_IND.astype(str)
        + "_"
        + ind.ID_VIAJE.astype(str)
    )
    ind["Datetime_start"] = ind.apply(
        lambda x: datetime.combine(x["ref_date"], x["start_time"]), 1
    )
    ind["Datetime_end"] = ind.apply(
        lambda x: datetime.combine(x["date_arrival"], x["end_time"]), 1
    )

    df_start = ind[["uid", "tid", "Datetime_start", "VORIZT1259",]].copy()
    df_end = ind[["uid", "tid", "Datetime_end", "VDESZT1259"]].copy()

    df_start.rename(
        columns={"Datetime_start": "datetime", "VORIZT1259": "tile_id"}, inplace=True,
    )
    df_end.rename(
        columns={"Datetime_end": "datetime", "VDESZT1259": "tile_id"}, inplace=True
    )

    df = pd.concat([df_start, df_end])
    return df[["uid", "tid", "datetime", "tile_id"]]


if config.MADRID not in config.DATASET_NAMES:
    print("Madrid not selected in config. Preprocessing is skipped.")
elif Path(os.path.join(preprocessed_data_path, "madrid.csv")).exists():
    print("Madrid data is already preprocessed. Processing is skipped.")
else:
    with tqdm(total=1, desc="Preprocess Madrid data",) as pbar:  # progress bar
        madrid_path = raw_data_path + "madrid/"
        df = madrid_preprocess_to_csv(madrid_path)
        tessellation = gpd.read_file(
            madrid_path + "/ZonificacionZT1259-shp/ZonificacionZT1259.shp"
        )[["ZT1259", "geometry"]]
        tessellation.rename(columns={"ZT1259": "tile_id"}, inplace=True)
        centroids = tessellation.to_crs(3035).centroid.to_crs(4326)
        tessellation["lng"] = centroids.x
        tessellation["lat"] = centroids.y
        df = pd.merge(
            df,
            tessellation[["tile_id", "lat", "lng"]],
            left_on="tile_id",
            right_on="tile_id",
            how="left",
        )
        df.to_csv(os.path.join(preprocessed_data_path, "madrid.csv"), index=False)

        tessellation = tessellation.to_crs(4326)
        tessellation[["tile_id", "geometry"]].to_file(
            os.path.join(preprocessed_data_path, "madrid_tessellation.gpkg"),
            driver="GPKG",
        )
        pbar.update()


###### Preprocess Berlin Data ######
if config.BERLIN not in config.DATASET_NAMES:
    print("Berlin not selected in config. Preprocessing is skipped.")

elif Path(os.path.join(preprocessed_data_path, config.BERLIN + ".csv")).exists():
    print("Berlin data is already preprocessed. Processing is skipped.")
else:
    with tqdm(total=1, desc="Preprocess Berlin data",) as pbar:  # progress bar

        df = pd.read_csv(
            os.path.join(raw_data_path, config.BERLIN, "berlin_raw.csv"),
            delimiter=";",
        )  
        # sample of 10%
        uids = df.p_id.unique()
        sample_10_perc = np.random.choice(uids, size = int(0.1*len(uids)), replace = False)
        df = df[df.p_id.isin(sample_10_perc)]
        df["tid"] = range(0, len(df))


        starts = df.loc[
            :, ["p_id", "tid", "lon_start", "lat_start", "start_time_min", "name_mct"]
        ]
        starts.rename(
            columns=dict(
                p_id="uid",
                lon_start="lng",
                lat_start="lat",
                start_time_min="datetime",
                name_mct="traffic_mode",
            ),
            inplace=True,
        )
        ends = df.loc[:, ["p_id", "tid", "lon_end", "lat_end", "end_time", "name_mct"]]
        ends.rename(
            columns=dict(
                p_id="uid",
                lon_end="lng",
                lat_end="lat",
                end_time="datetime",
                name_mct="traffic_mode",
            ),
            inplace=True,
        )

        df = starts.append(ends).sort_values("tid").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(
            (pd.Timestamp("2018-04-19").timestamp() + df["datetime"] * 60), unit="s"
        )

        tessellation = gpd.read_file(os.path.join(raw_data_path, config.BERLIN, "verkehrszellen_berlin.geojson"))
        tessellation.rename(
            columns={"vz_vbz_typ": "tile_id", "name": "tile_name"}, inplace=True
        )
        tessellation[["tile_id", "tile_name", "geometry"]].to_file(
            preprocessed_data_path + config.BERLIN + "_tessellation.gpkg", driver="GPKG"
        )

        df.to_csv(preprocessed_data_path + config.BERLIN + ".csv", index=False)
        pbar.update()
