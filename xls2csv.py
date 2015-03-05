import pandas as pd
from optparse import OptionParser

captiontext = "CaptionText"

if __name__ == "__main__":
    op = OptionParser()
    (opts, args) = op.parse_args()

    path = args[0]
    name = args[0][:-3]
    ext  = args[0][-3:]

    assert ext == "xls", "Supply a path to a .xls file."

    df         = pd.read_html(path)[0].fillna("")  # read_html returns a singleton list for some reason...
    df.columns = list(df.ix[0])
    df         = df.drop(df.index[0])
    
    assert isinstance(df, pd.DataFrame)
    assert captiontext in list(df.columns), """Please use a New Yorker Caption Contest formatted spreadsheet.
                                                    Columns not recognized: {}.""".format(set(df.columns) - (set(settings.columns) & set(df.columns)))

    df.CaptionText = df.CaptionText.apply(asciidammit)
    pd.to_csv(name + "csv")