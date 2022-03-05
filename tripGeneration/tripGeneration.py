"""
Preprocess the trip
"""
from datetime import datetime
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from . import constants as c
from .utils import (get_nearest_edge,
                    find_qualified_tazs_using_shortestpath,
                    find_qualified_tazs_with_poi,
                    get_random_taz_destination,
                    select_poi)

logging.basicConfig(level=logging.INFO)


def map_single_trip(
        home_loc: list,
        parsed_trip: list,
        taz_gdf: gpd.GeoDataFrame,
        nodes: pd.DataFrame,
        new_edges: pd.DataFrame,
        poi_df: pd.DataFrame,
        od_dict: dict,
        balltree_taz,
        balltree_nodes,
        shortest_path_graph):

    start_loc = home_loc
    work_loc = None
    output = []
    last_activity = datetime.strptime("2022/01/01 04:00:00", "%Y/%m/%d %H:%M:%S")
    home_edge_id, home_edge_index = get_nearest_edge(home_loc, balltree_nodes, nodes, new_edges)
    output.append([home_loc[0], home_loc[1],last_activity, None, 1, home_edge_id, home_edge_index,None,None])

    for transient in parsed_trip:
        # calculate activity duration
        activity_hour = (transient[3] - last_activity).seconds//3600
        activity_minute = (transient[3] - last_activity).seconds%3600//60
        activity_duration = str(activity_hour) + ":" +str(activity_minute)
        last_activity = transient[3]

        try:
            start_taz = taz_gdf[taz_gdf['geometry'].contains(Point(start_loc))]['TAZID'].values[0]
        except IndexError:
            print("Cannot find TAZ.")
            return None
        start_taz_nearest_node = taz_gdf[taz_gdf['geometry'].contains(Point(start_loc))]['nearest_node'].values[0]
        driving_time = transient[2]
        trip_purpose = transient[1]

        # Because home and workplace are fixed. So if the transient's destination is workplace or home, \
        # it will ba mapped automatically and skip the following matching process. (for workplace, it has\
        # to be mapped for the first time.
        if trip_purpose == 1: #home
            start_loc = home_loc
            output.append([home_loc[0], home_loc[1], transient[3], activity_duration, trip_purpose, home_edge_id,home_edge_index, driving_time,None])
            continue
        if trip_purpose == 2 and work_loc is not None: # work
            start_loc = work_loc
            output.append([work_loc[0], work_loc[1], transient[3], activity_duration, trip_purpose, work_edge_id,work_edge_index, driving_time,None])
            continue

        # ==================mapping process=======================
        # 1. use ball tree to shrink searching area
        qualified_tazs_index = balltree_taz.query_radius([start_loc], r=driving_time*c.SECONDS*c.RADIUS_SPEED)
        qualified_tazs = taz_gdf.iloc[qualified_tazs_index[0],:]

        # 2. search qualified TAZs that can be rearched around the driving time
        iter_time = 0
        while True:
            qualified_time_tazs = find_qualified_tazs_using_shortestpath(start_taz_nearest_node, qualified_tazs, driving_time*c.SECONDS,c.THRESHOLD,shortest_path_graph)
            if len(qualified_time_tazs) != 0:
                break
            driving_time=driving_time//2

        # 3. qualified TAZs must contain POIs that match the trip purpose
        qualified_time_poi_tazs=find_qualified_tazs_with_poi(qualified_time_tazs, poi_df, trip_purpose)
        qualified_time_poi_tazs_list = list(qualified_time_poi_tazs["TAZID"])

        # 4. randomly select a TAZ based on OD distribution
        hour = int(transient[3].hour)
        next_taz = get_random_taz_destination(start_taz, qualified_time_poi_tazs_list, od_dict, hour)
        est_time=qualified_time_poi_tazs[qualified_time_poi_tazs['TAZID']==next_taz]['est_time'].values[0]/60
        # 5. randomly select a POI in that TAZ
        poi_index, poi_x, poi_y = select_poi(next_taz, poi_df, trip_purpose)

        # 6. update variables for next iteration
        start_loc = [poi_x, poi_y]
        nearest_edge_id, nearest_edge_index = get_nearest_edge(start_loc,balltree_nodes,nodes,new_edges)
        if trip_purpose==2 and work_loc is None:
            work_loc = [poi_x, poi_y]
            work_edge_id = nearest_edge_id
            work_edge_index = nearest_edge_index
        output.append([poi_x,poi_y,transient[3],activity_duration,trip_purpose,nearest_edge_id,nearest_edge_index,driving_time,est_time])

    return output