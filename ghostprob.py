import argparse
import numpy as np
import tensorflow as tf
from ROOT import TFile, RDataFrame


# File description
columns = ["x", "y", "tx", "ty", "best_qop", "best_pt", "kalman_ip_chi2",
           "kalman_docaz", "chi2", "chi2V", "chi2UT", "chi2T",
           "ndof", "ndofV", "ndofT", "nUT", "ghost"]

# Bounds for the values
bounds = {"x": (-10., 10.),
          "y": (-10., 10.),
          "tx": (-0.3, 0.3),
          "ty": (-0.3, 0.3),
          "best_pt": (0, 15000),
          "kalman_ip_chi2": (-0.5, 10000.5),
          "kalman_docaz": (-0.5, 25.5),
          "chi2dof": (0, 400),
          "chi2Vdof": (0, 150),
          "chi2Tdof": (0, 150)}


def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="File with validator data", type=str, required=True)
    return parser.parse_args()


def __main__():
    arguments = command_line()

    # Load file into a table
    kalman_file = TFile(arguments.filename)
    df = RDataFrame("kalman_validator/kalman_ip_tree", kalman_file, columns).Define("p", "abs(1.f/best_qop)")

    # Filter out of bounds data
    for column in columns:
        if column in bounds:
            lower, upper = bounds[column]
            df = df.Filter(f"{column} > {lower} && {column} < {upper}")

    # Convert table to numpy
    np_df = df.AsNumpy()
    print(f"Entries in the table: {len(np_df)}")


if __name__ == "__main__":
    __main__()
