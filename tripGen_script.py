"""
Preprocess the trip
"""
import logging
import numpy as np
from pathlib import Path
import pickle

from sklearn.neighbors import BallTree
import osmnx as ox
import pandas as pd

from tripGeneration.trip_preprocess import process_batch_trips
from tripGeneration.dataloader import load_required_dataset
from tripGeneration.tripGeneration import map_single_trip
from tripGeneration import utils


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # step 1 : load data
    logging.info("==========================================")
    logging.info("=======       load data          =========")
    logging.info("==========================================")
    file_path = "C:/Users/zhiyan/Desktop/sample_data"
    data_dict = load_required_dataset(file_path)
    trip_df = data_dict["activity_df"]
    poi_df = data_dict["poi_df"]
    taz_gdf = data_dict["taz_gdf"]
    od_dict = data_dict["od_dict"]
    network_graph = data_dict["network_graph"]
    nodes, edges = ox.graph_to_gdfs(network_graph)
    new_edges = edges.reset_index()

    logging.info("==========================================")
    logging.info("=======       pre process          =======")
    logging.info("==========================================")

    # step 2 : preprocess the trip and build BallTree
    logging.info("*******Start to process daily trips*******")
    parsed_trips_list=process_batch_trips(trip_df=trip_df,resolution=10)
    logging.info("*******Daily trips are preprocessed*******")
    logging.info(f"raw trips number: {len(trip_df)}, valid trips number: {len(parsed_trips_list)}")

    logging.info("*******Start to build BallTree for TAZs and graph nodes*******")
    taz_coor_np = np.array([[p.x, p.y] for p in list(taz_gdf['centroid'])]).reshape((-1, 2))
    balltree_taz = BallTree(taz_coor_np, metric="minkowski")
    balltree_nodes = BallTree(nodes[["x","y"]], metric="minkowski")
    logging.info("*******BallTrees are successfully built*******")

    logging.info("==========================================")
    logging.info("=======       location matching    =======")
    logging.info("==========================================")
    with open("C:/Users/Zhiyan/Desktop/pre_computed_travel_time.pkl", "rb") as f:
        precomputed_travel_time_dict = pickle.load(f)

    home_locs = [[415000,4496719],[417000,4491719],[418000,4491719],[420000,4493719],[410000,4492319],[415000,4496719],[417000,4491719],[418000,4491719],[420000,4493719],[410000,4492319]]
    for person_id in range(10):
        print(f"^^^^^^^^matching person {person_id} ^^^^^^^^^^")
        output = map_single_trip(home_locs[person_id], parsed_trips_list[person_id], taz_gdf, nodes,new_edges,poi_df,\
                             od_dict,precomputed_travel_time_dict,balltree_taz,balltree_nodes)
        if output is None:
            continue
        output_df = utils.parse_output(output)
        print(output_df)
        #output_df_with_osm_time = utils.get_osm_travel_time(output_df,network_graph,new_edges,edges)
        #print(output_df_with_osm_time)