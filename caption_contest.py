import os
import sys
import argparse
import itertools
import csv
import string
import nltk
import gensim
import numpy as np
import pandas as pd

from utils import tokenize
from utils import remove_nonascii
from utils import remove_uppercase


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--csvdir", help="specify path to dir containing Caption Contest csvs, no trailing slash",
                    action="store")
args = parser.parse_args()


class CaptionCollection(object):
    """
    Class for a collection of Caption Contests.
    """

    def __init__(self):
        self.dirpath     = None
        self.num_topics  = None
        self.name        = None
        self.df          = None
        self.doc_map     = None
        self.docs_list   = None
        self.corpus      = None
        self.dictionary  = None


    def get_num_topics(self, dirpath):
        """
        as bob mankoff is fond of saying, there's two funny things in each cartoon. 
        """
        return 2*len(os.listdir(dirpath))


    def read_csvs(self, dirpath):
        """
        read all csvs in data/raw/
        return all rows
        """
        print "\n\nloading in csv data..."
        self.dirpath  = dirpath
        paths         = [dirpath + '/' + f for f in os.listdir(dirpath)]
        self.name     = '+'.join(os.listdir(dirpath))
        rows          = []        

        for fp in paths:
            with open(fp, 'rb') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    rows.append(row)

        return rows


    def construct_df(self, rows):
        """
        read in the data, put into a dataframe, and run preprocessing chain
        """
        print "\n\npreprocessing..."
        self.df            = pd.DataFrame(rows)
        self.df.columns    = self.df.ix[0,:]
        self.df            = remove_uppercase(self.df[self.df.ContestID != 'ContestID'])  # remove repeated header rows
        self.df['tokens']  = self.df.CaptionText.apply(remove_nonascii).apply(unicode).apply(tokenize)
        return self.df


    def get_contest_docs(self):
        """
        return a dict mapping contest IDs to long lists of tokens
        """
        docs_list  = [list(itertools.chain.from_iterable(contest['tokens'])) for contest in [self.df[self.df.ContestID == cid] for cid in self.df.ContestID.unique()]]
        doc_map    = dict(list(enumerate(self.df.ContestID.unique())))  # later we'll need to know which numbered doc corresponds to which subreddit

        return docs_list, doc_map


    def build_corpus(self):
        """
        serialize and return gensim corpus of contest-documents
        """
        print "\n\nbuilding corpus..."
        self.docs_list, self.doc_map  = self.get_contest_docs()

        print "\n\nbuilding dictionary..."
        self.dictionary  = gensim.corpora.Dictionary(self.docs_list)
        self.dictionary.save('data/processed/' + self.name + '.dict')

        print "\n\nserializing corpus..."
        corpus       = [self.dictionary.doc2bow(doc) for doc in self.docs_list]
        gensim.corpora.MmCorpus.serialize('data/processed/' + self.name + '.mm', corpus)
        self.corpus  = gensim.corpora.MmCorpus('data/processed/' + self.name + '.mm')

        print "\n\ndone."

        return self.corpus


    def tfidf_transform(self, corpus):
        """
        transform gensim corpus to normalized tf-idf
        """
        print "\n\ntransforming corpus to tfidf..."
        self.tfidf         = gensim.models.TfidfModel(corpus, normalize=True)
        self.corpus_tfidf  = self.tfidf[corpus]

        return self.corpus_tfidf


    def lda_transform(self, corpus):
        """
        fit lda 
        """
        print "\n\nfitting lda..."
        self.num_topics  = self.get_num_topics(self.dirpath)
        self.lda         = gensim.models.LdaModel(corpus, id2word=self.dictionary, num_topics=self.num_topics)
        self.corpus_lda  = self.lda[corpus]

        return self.corpus_lda


if __name__ == "__main__":

    c = CaptionCollection()

    if args.csvdir is not None:
        df = c.construct_df(c.read_csvs(args.csvdir))
    else:
        print "please specify a -c flag with path to csv directory."
        exit()

    corpus  = c.lda_transform(c.build_corpus())
    








