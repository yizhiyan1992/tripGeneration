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

import geopandas as gpd
import networkx
import pandas as pd
from shapely import wkt


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
    activity_df = pd.read_csv(activity_filepath)
    return activity_df


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
    poi_df['geometry'] = poi_df['geometry'].apply(wtk.loads)
    return poi_df


def load_od_data():
    pass


def load_osm_network(north: float, south: float, east: float, west: float) -> networkx.MultiDiGraph:
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

    Returns
    -----------------
    proj_graph: networkx.MultiDiGraph
        the multi directional graph representing the road network in OpenStreetMap.
    """
    graph = ox.graph_from_bbox(north, south, east, west, network_type = "drive")
    proj_graph = ox.prject_graph(graph, to_crs = crs)
    proj_graph = ox.add_edge_speeds(proj_graph)
    proj_graph = ox.add_edge_travel_times(proj_graph)
    return proj_graph


def load_required_dataset(folder_path, **kwargs):
    if "taz" in kwargs:
        taz_filename = kwargs["taz"]
    else:
        taz_filename = "taz.csv"

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

    # check if those files exist

    # start to load each data

    # return the packed dataset in a dictionary
    return
