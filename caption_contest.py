import os
import sys
import csv
import string
import nltk

import numpy as np
import pandas as pd


nonascii_table = {i: None for i in range(128,65375)}
punctuation_table = {ord(c): None for c in string.punctuation}



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


def tokenize(raw_string):
    """
    take in a string, return a tokenized and normalized list of words
    """
    if raw_string == '':
        return [u'']
    if not isinstance(raw_string, unicode):
        raise TypeError("%s is not unicode." % raw_string)
    return filter(
        lambda x: x not in nltk.corpus.stopwords.words('english'), 
        nltk.word_tokenize(raw_string.lower().translate(punctuation_table))
    )


def remove_nonascii(s):
    """
    strip out nonascii chars
    """
    return s.translate(nonascii_table)


def remove_uppercase(df):
    """
    drop every all-upper-case submission, because they universally suck
    """
    return df[df.CaptionText != df.CaptionText.apply(string.upper)]



if __name__ == "__main__":
    rows = read_csvs()
    df = pd.DataFrame(rows)
    df.columns = df.ix[0,:]
    df = df[df.ContestID != 'ContestID']
    df = remove_uppercase(df)
    df['tokens'] = df.CaptionText.apply(remove_nonascii).apply(unicode).apply(tokenize)








