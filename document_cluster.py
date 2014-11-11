"""
===================================================
Clustering Caption Contest Submissions with k-means
===================================================
by Charlie Hack <charlie@205consulting.com>


This script uses scikit-learn to cluster Caption Contest submission documents
by topics, using a bag-of-words approach. 

There are two methods available.

  - TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
    frequent words to features indices and hence compute a word occurrence
    frequency (sparse) matrix. The word frequencies are then reweighted using
    the Inverse Document Frequency (IDF) vector collected feature-wise over
    the corpus.

  - HashingVectorizer hashes word occurrences to a fixed dimensional space,
    possibly with collisions. The word count vectors are then normalized to
    each have l2-norm equal to one (projected to the euclidean unit-ball) which
    seems to be important for k-means to work in high dimensional space.

It's also possible to transform the corpus with Latent Semantic Analysis before
applying the clustering. 
"""
from __future__ import print_function

import sys
import logging
from optparse import OptionParser
from time import time
import string
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--csv",
              dest="csv_path", type="str",
              help="Path to the csv for the relevant caption contest.")
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--n-clusters", type=int, default=9,
              help="Number of clusters in the contest"
                   " as annotated by Stokes.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)
elif opts.csv_path is None:
    op.error("please specify the path to the caption contest csv.")
    sys.exit(1)


print("loading docs...")
df          = pd.read_csv(opts.csv_path).fillna("")
# drop explained_variance_ratio_y all-upper-case submission, because they universally suck
uppercased  = df[df.CaptionText == df.CaptionText.apply(string.upper)]
df          = df[df.CaptionText != df.CaptionText.apply(string.upper)]
dataset     = [c for c in df.CaptionText]

print("%d documents." % len(dataset))
print()

true_k = opts.n_clusters

print("Extracting features from the training dataset using a sparse vectorizer...")
t0 = time()
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
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
    # Vectorizer results are normalized, which makes KMeans behave as
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


###############################################################################
# Do the actual clustering

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
            verbose=opts.verbose)

print("Clustering sparse data with %s..." % km)
t0 = time()
km.fit(X)
print("done in %0.3fs." % (time() - t0))
print()

df['cluster']        = km.labels_
df['captionlength']  = df.CaptionText.apply(len)
df                   = df.sort(['cluster', 'captionlength'])

filename = opts.csv_path.split('/')[-1][:-4]

df.to_csv('data/processed/' + filename + '_processed.csv')

def print_cluster(n):
    for c in sorted(df[df.cluster == n].CaptionText, key=len):
      print(c)
    print()
    print("%i captions." % len(df[df.cluster == n].CaptionText))
    print()


for i in range(true_k):
    print("cluster %d:" % (i+1))
    print("===========")
    print_cluster(i)
    print()

if not (opts.n_components or opts.use_hashing):
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % (i+1), end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

print()
print("and finally, just because:")
print("--------------------------")
for c in uppercased.CaptionText:
    print(c)
print()




for row in places.iterrows():
    try:
        result = Geocoder.geocode(row[1].City + ', ' + row[1].State + ', ' + row[1].Country)
        coords.append(result[0].coordinates)
        sys.stdout.write('.'); sys.stdout.flush()
    except GeocoderError:
        sys.stdout.write('E'); sys.stdout.flush()
        continue

rounded_coords = [(round(x[0], 2), round(x[1], 2)) for x in coords]



places['coordinates'] = places.apply(lambda row: Geocoder.geocode(row.City + ', ' + row.State + ', ' + row.Country)[0].coordinates, axis=1)




