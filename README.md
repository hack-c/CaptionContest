Clustering The New Yorker Caption Contest Submissions with k-means
==================================================================
by Charlie Hack <charlie@205consulting.com>


This script uses scikit-learn to cluster Caption Contest submission documents
by topics, using a bag-of-words approach. 

There are two methods available:

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
applying the clustering, though empirically this doesn't seem to show a marked
improvement in the quality of the clusters.

It's hard to measure quantitatively how well the algorithm is doing at categorizing,
so it's easiest to just play around with different values for k and see what works.
Luckily the datasets are small (< 10000 documents) so that the script runs fast.

This is inspired by the scikit-learn documentation by Peter Prettenhofer and Lars 
Buitinck.


