"""
Load required data for trip generation.
** All required data should be placed in the same folder.
** Some dataset might be subjected to specific formats. The required columns for each data file can be found at XXX

The required data are listed as follows:
    1. stochastic daily activity data created by Markov Chain (csv format)
    2. household location for each driver (no. of activities == no. household locations) (csv format)
    3. TAZ data in the study area (shp format)
    4. POI data in the study area (csv)
    5. OD folder with OD distribution data in different time period (csv format)
    6. OSM network (by inputting the bounding box txt file (txt format)
"""

import logging
import pathlib
from pathlib import Path
import pickle

import geopandas as gpd
import networkx
import osmnx as ox
import pandas as pd
from shapely import wkt

logging.basicConfig(level=logging.INFO)


def load_stochastic_activity_data(activity_filepath: pathlib.WindowsPath) -> pd.DataFrame:
    """
    load activity data.

    Parameters
    --------------------
    activity_filepath_filename: pathlib.WindowsPath
        file path for activity file

    Returns
    --------------------
    activity_df : pandas.DataFrame
        return the activity dataframe
    """
    trip_df = pd.read_csv(activity_filepath, index_col=0)
    logging.info("========activity data is successfully loaded.========")
    return trip_df


def load_household_location_data():
    pass



def load_taz_data(taz_filepath: pathlib.WindowsPath) -> gpd.GeoDataFrame:
    """
    load TAZ shapefile.
    'centroid' column is the centroid point for each TAZ, and 'geometry' column is the Polygon feature.

    Parameters
    --------------------
    taz_filename: pathlib.WindowsPath
        file path for taz shp file

    Returns
    --------------------
    taz_gpd : gpd.GeoDataFrame
        return the taz gdf
    """
    taz_df = pd.read_csv(taz_filepath)
    taz_df['centroid'] = taz_df['centroid'].apply(wkt.loads)
    taz_df['geometry'] = taz_df['geometry'].apply(wkt.loads)
    taz_gdf = gpd.GeoDataFrame(taz_df, geometry = "geometry")
    logging.info("========TAZ data is successfully loaded.=========")
    return taz_gdf


def load_poi_data(poi_filepath: pathlib.WindowsPath) -> pd.DataFrame:
    """
    load POI data. POI data should contain geometry feature for each point.

    Parameters
    --------------------
    taz_filename: pathlib.WindowsPath
        file path for POI file

    Returns
    --------------------
    poi_df : pandas.DataFrame
        return the POI dataframe
    """
    poi_df = pd.read_csv(poi_filepath)
    poi_df['geometry'] = poi_df['geometry'].apply(wkt.loads)
    logging.info("========POI data is successfully loaded.=========")
    return poi_df


def load_od_data(od_folder_path):
    """
    The od_folder must include 4 OD csv files: od_1.csv, od_2.csv, od_3.csv, od_4.csv

    :param od_folder_path:
    :return:
    """
    od_files = [f for f in od_folder_path.iterdir() if f.suffix == ".csv"]
    od_dict = dict()
    for od_file in od_files:
        od_dict[od_file.stem] = pd.read_csv(od_file)
    logging.info("========OD data is successfully loaded.=========")
    return od_dict


def load_precomputed_travel_time(precomputed_tt_path):
    """
    The precomputed travel time file is a pkl saving the dictionary, where the key is "node_A-node_B",
    and the value is the shortest travel time from node_A to node_B.
    """
    with open(precomputed_tt_path,"rb") as f:
        precomputed_dict = pickle.load(f)
    return precomputed_dict


def load_osm_network(north: float, south: float, east: float, west: float, crs: str = "epsg:26912")\
        -> networkx.MultiDiGraph:
    """
    This function resorts to osmnx package to fetch drivable road network in the bounding box.
    Missing speed and travel time are computed.

    Parameters
    ----------------
    north: float
        northern lat of bounding box
    south:float
        southern lat of bounding box
    east: float
        eastern lon of bounding box
    west: float
        western lon of bounding box
    crs: string
        the projected coordinate system
    Returns
    -----------------
    proj_graph: networkx.MultiDiGraph
        the multi directional graph representing the road network in OpenStreetMap.
    """
    graph = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    proj_graph = ox.project_graph(graph, to_crs=crs)
    proj_graph = ox.add_edge_speeds(proj_graph)
    proj_graph = ox.add_edge_travel_times(proj_graph)
    logging.info("========Network graph is successfully fetched from OSM.=========")
    return proj_graph


def load_required_dataset(folder_path, **kwargs):
    logging.info("********start to load datasets********")
    if "taz" in kwargs:
        taz_filename = kwargs["taz"]
    else:
        taz_filename = "taz.shp"

    if "network_bbox" in kwargs:
        network_filename = kwargs["network_bbox"]
    else:
        network_filename = "network_bbox.txt"

    if "stochastic_activity" in kwargs:
        activity_filename = kwargs["stochastic_activity"]
    else:
        activity_filename = "stochastic_activity.csv"

    if "poi" in kwargs:
        poi_filename = kwargs["poi"]
    else:
        poi_filename = "poi.csv"

    if "od" in kwargs:
        od_folder = kwargs["od"]
    else:
        od_folder = "od_folder"

    if "precomputed_tt" in kwargs:
        precomputed_tt_filename = kwargs["precomputed_tt"]
    else:
        precomputed_tt_filename = "precomputed_tt.pkl"

    # check if those files exist
    folder_dir = Path(folder_path)
    od_dir = folder_dir.joinpath(od_folder)
    taz_fp = folder_dir.joinpath(taz_filename)
    poi_fp = folder_dir.joinpath(poi_filename)
    network_fp = folder_dir.joinpath(network_filename)
    activity_fp = folder_dir.joinpath(activity_filename)
    precomputed_tt_fp = folder_dir.joinpath(precomputed_tt_filename)
    # add hh distribution later
    if not folder_dir.is_dir():
        raise FileNotFoundError("The input file directory does not exist.")
    if not od_dir.is_dir():
        raise FileNotFoundError("The OD folder does not exist.")
    if not taz_fp.is_file():
        raise FileNotFoundError("The taz file does not exist.")
    if not poi_fp.is_file():
        raise FileNotFoundError("The POI file does not exist.")
    if not network_fp.is_file():
        raise FileNotFoundError("The network file does not exist.")
    if not activity_fp.is_file():
        raise FileNotFoundError("The activity file does not exist.")
    if not precomputed_tt_fp.is_file():
        raise FileNotFoundError("The precomputed travel time pkl file does not exist.")
    # add check for hh distribution later

    # start to load each data
    activity_df = load_stochastic_activity_data(activity_fp)
    poi_df = load_poi_data(poi_fp)
    taz_gdf = load_taz_data(taz_fp)
    od_dict = load_od_data(od_dir)
    precomputed_travel_time_dict = load_precomputed_travel_time(precomputed_tt_fp)
    with open(network_fp) as f:
        bbox_string_list = f.read().split(",")
        north, south, east, west = list(map(float, bbox_string_list))
        network_graph = load_osm_network(north, south, east, west)
    # return the packed dataset in a dictionary
    logging.info("********all required data are successfully loaded.********")
    data_dict ={
        "activity_df" : activity_df,
        "poi_df" : poi_df,
        "taz_gdf" : taz_gdf,
        "od_dict" : od_dict,
        "network_graph" : network_graph,
        "precomputed_travel_time_dict" : precomputed_travel_time_dict
    }
    return data_dict


if __name__ == "__main__":
    filepath = "/Users/zhiyan_yi/Desktop/input_data"
    load_required_dataset(filepath)