"""
Basic util functions
"""

from datetime import datetime, timedelta
import logging
import pathlib
from pathlib import Path

from dijkstar import Graph, find_path
import geopandas as gpd
import numpy as np
import networkx
import osmnx as ox
import pandas as pd
from shapely import wkt
from sklearn.neighbors import BallTree

logging.basicConfig(level=logging.INFO)


# utils for trip pre-processing
def is_connected_by_driving(trip_list: list) -> bool:
    """
    For a valid trip, two different places must be connected by driving process.
    i.e. home->driving->office. cases like home->office is invalid trip.
    """
    for i in range(1,len(trip_list)):
        if trip_list[i]!=trip_list[i-1] and (trip_list[i]!=0 and trip_list[i-1]!=0):
            return False
    return True


def if_driving_today(trip_list: list) -> bool:
    """
    If driving activity happens today, return True, otherwise False
    """
    return 0 in trip_list


def is_valid_start(trip_list: list) -> bool:
    """
    Valid start is defined as a trip starting from home.
    """
    if trip_list[0]!=1:
        return False
    return True


def is_valid_end(trip_list):
    """
    Valid end is defined as a trip ending at home.
    """
    if trip_list[-1]!=1:
        return False
    return True

def fix_trip_start(trip_list: list) -> list:
    """
    If the start is not from home,
    set the start place as home, and 10 mins drive to the next place
    """
    trip_list[0]=1
    trip_list[1]=0
    return trip_list

def fix_trip_end(trip_list: list) -> list:
    """
    If the end is not at home,
    set the end place at home, and 10 mins drive to home.
    """
    trip_list[-1]=1
    trip_list[-2]=0
    return trip_list


def parse_daily_activity(activity_list):
    """
    Parse the valid daily trip into specific format: list[(start_purpose, end_purpose, time_duration, start_time),...]
    A valid trip should be:
        1. activity starts at 4:00 am and ends at 4:00 in the next day.
        2. start place must be home and end place must be home as well.
        3. every continuous two locations must be connected by "driving"

    """
    moving_list = []
    start_time = "2022/01/01 04:00:00"
    time_format = "%Y/%m/%d %H:%M:%S"
    cur_time = datetime.strptime(start_time, time_format)
    time_arr = [cur_time]
    driving_start = None
    driving_end = None

    for idx, activity in enumerate(activity_list):
        # deal with corner case idx==0 and idx==len(act)-1
        if activity == 0 and (idx == 0 or activity_list[idx - 1] != activity):
            driving_start = idx
        if activity == 0 and (idx == len(activity_list) - 1 or activity_list[idx + 1] != activity):
            driving_end = idx
        if driving_start != None and driving_end != None:
            moving_list.append([activity_list[driving_start - 1], activity_list[driving_end + 1],
                                (driving_end - driving_start + 1) * 10, time_arr[driving_start]])
            driving_start = None
            driving_end = None

        cur_time += timedelta(minutes=10)
        time_arr.append(cur_time)
    return moving_list


# utils for location mapping
def get_nearest_edge(loc,balltree_nodes,nodes_df,new_edges_df) ->int:
    """
    Given the location [x,y], find the id of its nearest edge.
    """
    _,index=balltree_nodes.query([loc])
    index=index[0][0]
    edge_id=new_edges_df[(new_edges_df['u']==nodes_df.index[index]) |\
                         (new_edges_df['v']==nodes_df.index[index])]['osmid'].values[0]
    edge_index = new_edges_df[(new_edges_df['u'] == nodes_df.index[index]) | \
                           (new_edges_df['v'] == nodes_df.index[index])].index.values[0]
    if isinstance(edge_id,list): # might find duplicates
        edge_id=edge_id[0]
    return edge_id, edge_index


def find_qualified_tazs_using_shortestpath(
        start_node,
        qualified_tazs_gdf,
        driving_time,
        threshold,
        graph):
    """
    :param start_node:
    :param qualified_tazs_gdf:
    :param precomputed_travel_time_dict: a dict that records the pre-computed shortest travel time between each TAZ centroid
    :param driving_time: the driving time in the MC simulated result.
    :param threshold: determine the upper and lower bounds (percentage)
    :return: qualified TAZs given the constraint of driving time
    """
    taz_ids = list(qualified_tazs_gdf['TAZID'])
    taz_nereast_nodes = list(qualified_tazs_gdf['nearest_node'])
    upper_bound = driving_time*(1+threshold)
    lower_bound = driving_time*(1-threshold)
    output_index = []
    est_time=[]
    for i, n in enumerate(taz_nereast_nodes):
        try:
            result = find_path(graph, start_node, n)
            travel_time = result.total_cost
        except:
            travel_time = 100000
        if lower_bound<=travel_time<=upper_bound:
            output_index.append(taz_ids[i])
            est_time.append(travel_time)
    output_gdf= qualified_tazs_gdf[qualified_tazs_gdf['TAZID'].isin(output_index)]
    output_gdf['est_time']=est_time
    return output_gdf


def make_graph(edges_df):
    graph = Graph()
    u = list(edges_df['u'])
    v = list(edges_df['v'])
    t = list(edges_df['travel_time'])
    osmid = list(edges_df['osmid'])
    osm_edge_dict = {}
    for i in range(len(u)):
        graph.add_edge(u[i], v[i], t[i])
        osm_edge_dict[str(u[i]) + ":" + str(v[i])] = osmid[i]
    return graph,osm_edge_dict


def find_qualified_tazs_with_poi(
        qualified_tazs_gdf,
        poi_df,
        trip_purpose):
    taz_with_corresponding_poi=poi_df[poi_df['purpose_index']==trip_purpose]['TAZID'].unique()
    return qualified_tazs_gdf[qualified_tazs_gdf['TAZID'].isin(taz_with_corresponding_poi)]


def get_random_taz_destination(start_taz, candidate_tazs_list, od_dict, hour):
    prob_accu_list = []
    trip_sum = 0
    if 6 <= hour < 9:
        od_df = od_dict['od_1']
    elif 9 <= hour < 15:
        od_df = od_dict['od_2']
    elif 15 <= hour < 19:
        od_df = od_dict['od_3']
    else:
        od_df = od_dict['od_4']

    # accumulate the probability for all TAZs
    for end_taz in candidate_tazs_list:
        try:
            trip_count = od_df[(od_df['i'] == start_taz) & (od_df['j'] == end_taz)]['VTrips'].values[0]
        except IndexError:
            trip_count = 0
        trip_sum += trip_count
        prob_accu_list.append(trip_sum)

    # generate a random number, and randomly assign a destination TAZ
    rand_num = np.random.uniform(0, trip_sum)

    for i in range(len(prob_accu_list)):
        if i == 0 and prob_accu_list[i] > rand_num:
            result_idx = 0
            break
        elif i == len(prob_accu_list) - 1:
            result_idx = i
            break
        else:
            if prob_accu_list[i - 1] < rand_num <= prob_accu_list[i]:
                result_idx = i
                break
    return candidate_tazs_list[result_idx]


def select_poi(taz_id,poi_df,trip_purpose):
    possible_pois=poi_df[(poi_df['TAZID']==taz_id) & (poi_df['purpose_index']==trip_purpose)]
    rand_num=int(np.random.uniform(0,len(possible_pois)))
    poi_idx=possible_pois.iloc[rand_num]['Index']
    poi_x,poi_y=list(possible_pois.iloc[rand_num]['geometry'].coords)[0]
    return (poi_idx,poi_x,poi_y)


# auxiliary
def parse_output(output: list)->pd.DataFrame:
    x_list = []
    y_list = []
    nearest_id_list = []
    nearest_index_list = []
    end_time_list = []
    activity_duration_list = []
    trip_purpose_list = []
    driving_duration_list = []
    est_time_list=[]

    for o in output:
        x_list.append(o[0])
        y_list.append(o[1])
        end_time_list.append(o[2].strftime("%H:%M"))
        activity_duration_list.append(o[3])
        trip_purpose_list.append(o[4])
        nearest_id_list.append(o[5])
        nearest_index_list.append(o[6])
        driving_duration_list.append(o[7])
        est_time_list.append(o[8])

    df = pd.DataFrame(np.array(
        [x_list, y_list, end_time_list, activity_duration_list, trip_purpose_list, nearest_id_list,
         nearest_index_list, driving_duration_list, est_time_list]).T,
                      columns=['x', 'y', 'end_time', 'duration', 'purpose', 'nearest_edge_id','nearest_edge_index', 'driving_minutes','est_time'])
    return df


def get_osm_travel_time(trip_df,new_edges,shortest_path_graph):
    """
    supplement a new column for the output dataframe: the estimated travel time in OSM
    """
    edge_index_list=list(trip_df['nearest_edge_index'])
    travel_time_list=[None]
    for i in range(1,len(edge_index_list)):
        start_node = new_edges.iloc[edge_index_list[i - 1], :]['u']
        end_node = new_edges.iloc[edge_index_list[i], :]['v']
        result=find_path(shortest_path_graph,start_node,end_node)
        travel_time=round(result.total_cost/60,1)
        travel_time_list.append(travel_time)
    trip_df['actual_travel_time_osm']=travel_time_list
    return trip_df


def generate_new_edges_df(edges):
    new_edges=edges.copy()
    new_edges=new_edges.reset_index()
    edge_dict = {}
    u = list(new_edges['u'])
    v = list(new_edges['v'])
    dup_idx_list = []
    for i in range(len(new_edges)):
        s = str(u[i]) + ":" + str(v[i])
        if s in edge_dict:
            dup_idx_list.append(i)
        edge_dict[s] = 0
    new_edges = new_edges.drop(index=dup_idx_list)
    new_edges['osmid'] = list(range(len(new_edges)))
    return new_edges
