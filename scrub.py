import pandas as pd
from optparse import OptionParser


if __name__ == "__main__":
    op = OptionParser()
    (opts, args) = op.parse_args()

    path = args[0]
    name = args[0][:-3]
    ext  = args[0][-3:]

    keep_cols = [u'CaptionEntryID', u'ContestID', u'CaptionText', u'FirstName',]

    df = pd.read_csv(path)
    df = df[keep_cols]

    df.to_csv(path, index=False)
