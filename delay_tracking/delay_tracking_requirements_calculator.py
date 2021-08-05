"""
Script to calculate MeerKAT Extension delay tracking requirements.

This script loads the MeerKAT Extension antenna coordinates, determines the maximum baseline length from this and then
calculates the MeerKAT Extension delay tracking requirements introduced by this distance.

The requirements that this script is meant to determine include:
1. The maximum baseline length that needs to be accounted for.
2. The maximum delay that the system needs to compensate for.
3. The maximum and minumum rate of change of delay.
"""

__author__ = "SARAO DSP Group"

import argparse
import csv
import itertools
from typing import Tuple

import numpy
import scipy
import scipy.constants
from geopy import distance


def get_coordinates_from_csv(
    path_to_csv: str, delimiter: str = ",", col_labels: bool = True, latitude_first: bool = False
) -> dict:
    """
    Read antenna coordinates from csv file into a dictionary.

    Assumes coordinates are formatted as decimal degrees.
    Expected format of each entry in the csv: ANT_NAME, Latitude, Longitude OR ANT_NAME, Longitude, Latitude
    Column labels/headings in the CSV file are optional.

    :param path_to_csv: path to csv file containing dish positions specified by geographic coordinates
    :param delimiter: specify what delimiter is used in the csv file, default=,
    :param col_labels: set to False if csv file does not contain column labels
    :param latitude_first: set to True if latitude coodinate is specified first in the csv file
    :return: dict() Each item in the returned dict has the structure key=ant_name, value=(latitude, longitude)
    """
    tmp = list()
    with open(path_to_csv, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            tmp.append(row)

    if col_labels:
        tmp.pop(0)  # remove column labels if present

    if latitude_first:
        ant_coords = {ant[0]: (ant[1], ant[2]) for ant in tmp}
    else:
        ant_coords = {ant[0]: (ant[2], ant[1]) for ant in tmp}

    return ant_coords


def calculate_baselines(antenna_pos_coords: dict) -> dict:
    """
    Calculate baselines for a given set of antennas described by their geographic coordinates.

    :param antenna_pos_coords: dictionary containing antenna names as keys and their geographic coords,
    in a tuple, as values
    :return: dict() each item in the returned dict has the structure key=(ant01, ant02), value=baseline_in_km
    """
    # each item in baselines is of form {key=(ant01, ant02), value=baseline_in_km}
    baselines = dict()

    # iterate through all pairs of antennas
    for ant_pair in itertools.product(antenna_pos_coords, repeat=2):
        ant_a = (
            antenna_pos_coords[ant_pair[0]][0],
            antenna_pos_coords[ant_pair[0]][1],
        )  # get lat and long of first ant
        ant_b = (
            antenna_pos_coords[ant_pair[1]][0],
            antenna_pos_coords[ant_pair[1]][1],
        )  # get lat and long of second ant

        # calculate baseline between antenna pair
        # NOTE: Technically the distance that we want is a chord, not a geodesic. Geopy doesn't
        # make it easy to get this though, and the difference is trivial on these distance scales, so
        # we have opted to leave this as-is.
        baselines[ant_pair] = distance.GeodesicDistance(ant_a, ant_b).kilometers

    return baselines


def find_longest_baseline(baselines: dict) -> Tuple[str, float]:
    """
    Retrieve the longest baseline from a set of baselines calculated for an arrangment of antennas.

    :param baselines: dictionary containing antennas pairs as keys and baselines as values
    :return:  vanteanna_pair_with_longest_baselines, baseline_in_km
    """
    # get the maximum value and return the corresponding key and value
    longest_baseline_ants = max(baselines, key=lambda x: x[1])
    longest_baseline = baselines[longest_baseline_ants]

    return (longest_baseline_ants, longest_baseline)


def calculate_delay_from_source_elevation(baseline: float, source_elevation_degrees: float) -> float:
    """
    Calculate the delay between two antennas in seconds based on a source's elevation.

    See "Theory" section in the readme.md for further explation of how this is calculated.

    :param baseline: distance between the two antennas considered, for specifications, use the maximum baseline, in m
    :param source_elevation_degrees: elevation to the point source, specificed in degrees
    """
    if source_elevation_degrees > 90 or source_elevation_degrees < 0:
        raise TypeError("source_elevation_degrees needs to be within: 0<=x<=90")

    # Calculate the delay distance using properties of the triangle described in the docstring above
    delay_length_m = baseline * numpy.cos(source_elevation_degrees / 180 * scipy.constants.pi)
    delay_length_s = delay_length_m / scipy.constants.c

    return delay_length_s


def calculate_delay_rate_of_change(
    baseline: float, elevation: float, elevation_change_per_second: float, max_or_min: str
) -> float:
    """
    Calculate delay rate of change required for a source point at a given elevation.

    :param baseline: distance between the two antennas considered, for specifications, use the maximum baseline, in m
    :param elevation: elevation to the point source, specificed in degrees
    :param elevation_change_per_second: change in elevation, degrees per second
    :param max_or_min: specify whether the maximum or minimum rate of change is being calculated
    """
    delay1_s = calculate_delay_from_source_elevation(baseline, elevation)
    if max_or_min == "max":
        delay2_s = calculate_delay_from_source_elevation(baseline, elevation - elevation_change_per_second)
    else:
        delay2_s = calculate_delay_from_source_elevation(baseline, elevation + elevation_change_per_second)

    delay_rate_of_change = abs((delay2_s - delay1_s) * 1000 * 1000 * 1000)

    return delay_rate_of_change


def calculate_delay_tracking_requirements(longest_baseline: float) -> dict:
    """
    Calculate the delay tracking requirements for a given maximum baseline.

    :param longest_baseline: maximum baseline to be considered in the derivation of requirements, in km
    :return: dict() containing all delay tracking specifications
    """
    # calculate maximum delay compensation required

    # The maximum baseline length determines the maximum coarse delay - this corresponds to an object just on the
    # horizon where the wavefront needs to travel almost directly along the baseline.
    longest_baseline_m = longest_baseline * 1000  # convert to metres
    max_coarse_delay_s = longest_baseline_m / scipy.constants.c

    # Technically what was done for MeerKAT is to take "max_coarse_delay_s" and double it - this is done to account for
    # a virtual reference antenna. It is not absolutly necessary to double it, this is just the way we have
    # implemented it. You could just as easily have added a constant of some us to get a delay
    # of "max_coarse_delay_s + x us"

    # We then need to account for the different propagation times in the PPS signal. This signal originates in the
    # central MeerKAT data centre (The KAPB). The cable lengths this signal is sent on are not equal. The difference in
    # time that needs to be accounted for is the difference in propagation time between the shortest and longest cable.
    # A more than worst case estimate for this length is the length of the longest baseline. As such, the maximum delay
    # compensation required becomes "2 * max_coarse_delay_s + ~max_coarse_delay_s ~= 3 * max_coarse_delay_s"

    max_coarse_delay_s *= 3

    # calculate range of rate of change of delay

    # The model here assumes the earth remains stationary, and the source moves across the sky changing the wavefront
    # angle at a uniform rate.
    # Calculation is for an elevation change by 90 degrees over six hours.
    elevation_change_per_second = 90 / (3600 * 6)

    # maximum and minimum elevation:
    # maximum rate of change of delay occurs at 90 degrees elevation
    # minumum rate of change of delay occurs at 15 degrees elevation
    elevation = {"min": 15, "max": 90}

    # rate of change is specific in picoseconds/second
    range_of_rate_of_change = {
        k: calculate_delay_rate_of_change(
            baseline=longest_baseline_m,
            elevation=v,
            elevation_change_per_second=elevation_change_per_second,
            max_or_min=k,
        )
        for (k, v) in elevation.items()
    }

    return {
        "longest_baseline_km": longest_baseline_m / 1000,
        "max_coarse_delay_us": max_coarse_delay_s * 1000 * 1000,
        "range_of_rate_of_change_of_delay_ns_s": range_of_rate_of_change,
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ant_pos", type=str, help="csv file containing " "antenna positions defined by geographic coordinates"
    )
    parser.add_argument("-d", dest="delimiter", type=str, help="delimter in csv " "file", default=",")
    parser.add_argument(
        "-l",
        dest="lat_first",
        action="store_true",
        help="specify that latitude coords appear " "first in the given csv file",
    )
    parser.add_argument(
        "-n", dest="no_col_labels", action="store_true", help="specify that no column " "labels are present"
    )

    args = parser.parse_args()

    ant_coords = get_coordinates_from_csv(
        path_to_csv=args.ant_pos,
        delimiter=args.delimiter,
        col_labels=not args.no_col_labels,
        latitude_first=args.lat_first,
    )

    baselines = calculate_baselines(antenna_pos_coords=ant_coords)

    longest_baseline_ants, longest_baseline_km = find_longest_baseline(baselines=baselines)

    delay_tracking_reqs = calculate_delay_tracking_requirements(longest_baseline=longest_baseline_km)

    print("================================================================")
    print("Correlator Delay Tracking Requirements")
    print("================================================================")
    print(f"Longest Baseline: {longest_baseline_km:.2f} km")
    print(f"Longest Baseline Antenna Pair: {longest_baseline_ants}")
    print(f"Maximum Delay Compensation: {delay_tracking_reqs['max_coarse_delay_us']:.2f} us")
    print(
        "Range of rate of change of delay: ",
        f"<={delay_tracking_reqs['range_of_rate_of_change_of_delay_ns_s']['min']:.2f}",
        f"ns/s to >= {delay_tracking_reqs['range_of_rate_of_change_of_delay_ns_s']['max']:.2f} ns/s",
    )
    print("================================================================")
