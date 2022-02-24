"""
Preprocess the trip
"""

import logging
import pandas as pd

from . import utils

logging.basicConfig(level=logging.INFO)


def process_single_trip(daily_trip, trip_id = None, verbose =False):
    # preprocess the trip and filter invalid.

    # step 1: check if driving happens in this activity, and if two places are connected by driving.
    if not (utils.if_driving_today(daily_trip) and utils.is_connected_by_driving(daily_trip)):
        if verbose:
            logging.info(f"Trip {trip_id} is not valid!")
        return None

    # step 2: fix the trip start and end if the trip does not meet the requirement.
    if not utils.is_valid_start(daily_trip):
        daily_trip=utils.fix_trip_start(daily_trip)
    if not utils.is_valid_end(daily_trip):
        daily_trip=utils.fix_trip_end(daily_trip)

    # step 3: parse the activity into specific format
    parsed_trip=utils.parse_daily_activity(activity_list=daily_trip)
    return parsed_trip


def process_batch_trips(trip_df: pd.DataFrame, resolution: int) ->list:
    """

    :param trip_df:
    :param resolution:
    :return:
    """
    trip_num, time_step = trip_df.shape
    if time_step * resolution != 24 * 60:
        raise ValueError("The time step of activity does not match with resolution!")

    trip_list = []
    invalid_counter = 0
    for i in range(len(trip_df)):
        daily_trip = list(trip_df.iloc[i, :])
        parsed_daily_trip = process_single_trip(daily_trip=daily_trip, trip_id=i)
        if parsed_daily_trip is None:
            invalid_counter += 1
        else:
            trip_list.append(parsed_daily_trip)
    return trip_list
