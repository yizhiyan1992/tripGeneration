"""
Preprocess the trip
"""
import logging
import numpy as np
from pathlib import Path

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
    household_df = data_dict["household_df"]
    poi_df = data_dict["poi_df"]
    taz_gdf = data_dict["taz_gdf"]
    od_dict = data_dict["od_dict"]
    #precomputed_travel_time_dict = data_dict["precomputed_travel_time_dict"]
    network_graph = data_dict["network_graph"]
    nodes, edges = ox.graph_to_gdfs(network_graph)
    #new_edges = edges.reset_index()
    new_edges = utils.generate_new_edges_df(edges)
    shortest_path_graph,_ =utils.make_graph(new_edges)

    logging.info("==========================================")
    logging.info("=======       pre process          =======")
    logging.info("==========================================")

    # step 2 : preprocess the trip and build BallTree
    logging.info("*******Start to process daily trips*******")
    parsed_trips_list=process_batch_trips(trip_df=trip_df,resolution=10)
    if len(parsed_trips_list)<len(household_df):
        raise ValueError(f"""
        The number of valid trips should not be smaller than the number of person.
        Got valid trips: {len(parsed_trips_list)}, number of households: {len(household_df)}.
        """)
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

    output_list = []
    #for person_id in range(len(household_df)):
    for person_id in range(20):
        home_loc = [household_df["x"][person_id],household_df["y"][person_id]]
        print(f"^^^^^^^^matching person {household_df['id'][person_id]} ^^^^^^^^^^")
        try:
            map_result = map_single_trip(home_loc, parsed_trips_list[person_id], taz_gdf, nodes,new_edges,poi_df,\
                             od_dict,balltree_taz,balltree_nodes,shortest_path_graph)
            if map_result is None:
                print(f"{household_df['id'][person_id]} matching failed.")
                continue
            map_result_df = utils.parse_output(map_result)
            map_result_df["person_id"] = household_df['id'][person_id]
            print(map_result_df)
            output_list.append(map_result_df)
        except:
            print(f"{household_df['id'][person_id]} matching failed.")
    output_df = pd.concat(output_list, axis=0)
    output_df.to_csv("C:/Users/Zhiyan/Desktop/output_df.csv")