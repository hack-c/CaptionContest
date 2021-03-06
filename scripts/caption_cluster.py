"""
===================================================
Clustering Caption Contest Submissions with k-means
===================================================
by Charlie Hack 
<charlie@205consulting.com>


This script uses scikit-learn to cluster Caption Contest submissions
by topics, using spherical k-means.
"""
from __future__ import print_function

import sys
import logging
import string
import nltk
import numpy as np
import pandas as pd
from time import time
from optparse import OptionParser

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans

import settings
from utils import asciidammit

captiontext = "CaptionText"


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--no-hashing",
              action="store_true", default=False,
              help="Disable hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")



if __name__ == "__main__":

    ###################################################################
    #############################[ Args ]##############################
    ###################################################################

    (opts, args) = op.parse_args()

    print(__doc__)
    op.print_help()
    
    
    if len(args) < 2:
        op.error("Please supply a path to a csv or xls file and k value.")
        sys.exit(1)

    filepath  = args[0]
    extension = filepath[-3:]

    k = int(args[1])  # TODO: find a better way to accept args





    ###################################################################
    ############################[ Gametime ]###########################
    ###################################################################

    print()
    print()
    print("Reading file...")
    print()
    if extension == "csv":
        df = pd.read_csv(filepath).fillna("")

    elif extension == "xls":
        df         = read_xls(path)
        assert isinstance(df, pd.DataFrame)

    else:
        op.error("Unrecognized filetype. Please specify a path to a csv or xls file.")
        sys.exit(1)


    assert len(df.columns) > 0, "Please supply a file with stuff in it!"

    if len(df.columns) == 1 and df.columns[0] != "CaptionText":
        assert list(df.ix[:,0].apply(type).unique()) == [str], "Unrecognized stuff in here. Are you sure this is Caption Contest csv?"
        df.columns = ["CaptionText"]
    else:
        assert "CaptionText" in df.columns, "Unrecognized columns. Are you sure this is a Caption Contest csv?"



    # drop all-upper-case submissions, because they universally suck
    # but save them because they're also hilarious
    uppercased  = df[df[captiontext] == df[captiontext].apply(string.upper)]
    df          = df[df[captiontext] != df[captiontext].apply(string.upper)]

    # scrub non-ascii characters
    df[captiontext] = df[captiontext].apply(asciidammit)

    dataset     = [c for c in df[captiontext]]

    print("%d documents." % len(dataset))
    print()

    print("Extracting features from the training dataset using a sparse vectorizer...")
    t0 = time()
    if not opts.no_hashing:
        if opts.use_idf:
            # perform IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english', non_negative=True,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                                           stop_words='english',
                                           non_negative=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                     min_df=2, stop_words='english',
                                     use_idf=opts.use_idf)

    X = vectorizer.fit_transform(dataset)

    print("done in %fs." % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    if opts.n_components:
        print("Performing dimensionality reduction using LSA...")
        t0 = time()
        # vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(opts.n_components)
        lsa = make_pipeline(svd, Normalizer(copy=False))

        X = lsa.fit_transform(X)

        print("done in %fs." % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        print()


    # do the actual clustering
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

    print("Clustering sparse data with %s..." % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs." % (time() - t0))
    print()

    df['cluster']        = km.labels_
    df['captionlength']  = df[captiontext].apply(len)
    df                   = df.sort(['cluster', 'captionlength'])

    filename = filepath.split('/')[-1][:-4]

    # dump the processed df to csv
    # df.to_csv('data/processed/' + filename + '_processed.csv', index=False)


    print_top_terms = False
    if not (opts.n_components or (not opts.no_hashing)):  # no_hashing is ON, n_components is OFF
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        print_top_terms = True

    def print_cluster(n, print_top_terms=True):
        print("cluster {}:".format(n+1), end='')
        if print_top_terms:
            for ix in order_centroids[i,:2]:
                print(' {}'.format(terms[ix]), end='')
        print("\n======================")
        for c in sorted(df[df.cluster == n].CaptionText, key=len):
            print(c)
        print()
        print("%i captions." % len(df[df.cluster == n].CaptionText))
        print()


    for i in range(k):
        print_cluster(i, print_top_terms)
        print()






    print()
    print("and finally, just because:")
    print("--------------------------")
    for c in uppercased.CaptionText:
        print(c)
    print()


