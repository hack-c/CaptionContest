import string
import nltk

num_topics         = 50
min_len            = 8
punctuation_table  = {ord(c): None for c in string.punctuation}
stopwords          = nltk.corpus.stopwords.words('english')