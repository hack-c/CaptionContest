import sys
import string
import nltk

useless_words = [
    'dont',
    'get',
    'going',
    'hes',
    'im',
    'know',
    'like',
    'one',
    'think',
    'hell',
    'ill',
    'really',
    'said',
    'youre',
    'feel',
    'go',
    'got',
    'see',
    'well']

useless_words += nltk.corpus.stopwords.words('english')



asciis             = frozenset(string.ascii_lowercase + string.ascii_uppercase + ' ')
nonascii           = ''.join([unichr(i) for i in range(128,65375)])
punctuation_table  = {ord(c): None for c in string.punctuation}


def tokenize(raw_string):
    """
    take in a string, return a tokenized and normalized list of words
    """
    sys.stdout.write('.')
    sys.stdout.flush()
    if raw_string == '':
        return [u'']
    if not isinstance(raw_string, unicode):
        raise TypeError("%s is not unicode." % raw_string)
    return filter(
        lambda x: x not in useless_words, 
        nltk.word_tokenize(raw_string.lower().translate(punctuation_table))
    )


def remove_nonascii(s):
    """
    strip out nonascii chars
    """
    return filter(asciis.__contains__, s)


def remove_uppercase(df):
    """
    drop every all-upper-case submission, because they universally suck
    """
    return df[df.CaptionText != df.CaptionText.apply(string.upper)]