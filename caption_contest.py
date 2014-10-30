import os
import sys
import csv

import numpy as np
import pandas as pd


def read_csvs():
    """
    read all csvs in data/raw/
    return all rows
    """

    paths  = ['data/raw/' + f for f in os.listdir('data/raw/')]
    rows   = []

    for fp in paths:
        with open(fp, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                rows.append(row)

    return rows



if __name__ == "__main__":
    rows = read_csvs()
    