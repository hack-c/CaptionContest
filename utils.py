import nltk


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