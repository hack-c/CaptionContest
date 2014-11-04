import os
import sys
import csv
import string
import nltk

import numpy as np
import pandas as pd

from utils import tokenize
from utils import remove_nonascii
from utils import remove_uppercase


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


def construct_df(rows):
    """
    read in the data, put into a dataframe, and run preprocessing chain
    """
    df            = pd.DataFrame(rows)
    df.columns    = df.ix[0,:]
    df            = remove_uppercase(df[df.ContestID != 'ContestID'])  # remove repeated header rows

    df['tokens']  = df.CaptionText.apply(remove_nonascii).apply(unicode).apply(tokenize)




if __name__ == "__main__":
    df = construct_df(read_csvs())
    








