import pandas as pd
from optparse import OptionParser

from utils import asciidammit, scrub, read_xls

captiontext = "CaptionText"

if __name__ == "__main__":
    op = OptionParser()
    (opts, args) = op.parse_args()

    path = args[0]
    name = args[0][:-3]
    ext  = args[0][-3:]

    assert ext == "xls", "Supply a path to a .xls file."


    df = read_xls(path)
    
    assert isinstance(df, pd.DataFrame)
    # assert captiontext in list(df.columns), """Please use a New Yorker Caption Contest formatted spreadsheet.
                                                     # Columns not recognized: {}.""".format(set(df.columns) - (set(settings.columns) & set(df.columns)))


    ##################[ scrub personal data ]####################

    df = scrub(df)

    df.CaptionText    = df.CaptionText.apply(asciidammit)
    df.FirstName      = df.FirstName.apply(asciidammit)

    df.to_csv(name + "csv")

