###################################################
###############[ Module: xls2csv ]#################
###################################################
"""
Commmand line utility for converting and cleaning Cartoon Dept. spreadsheets. 

Run 

    $ xls2csv --help

for args, options and usage examples.
"""
import os
import click
import pandas as pd

from captioncontest import utils


CAPTIONTEXT = "CaptionText"


@click.command()
@click.argument("path", type=click.Path(exists=True), required=True)
@click.argument("outpath", type=click.Path(exists=False), required=False)
@click.option("--captions/--nocaptions", "captions", default=False, help="Validate 'MyReport'-style caption spreadsheet format.")
@click.option("--scrub/--noscrub", "scrub", default=False, help="Scrub personal data.")
def main(path, outpath, captions, scrub):
    """
    Command line utility for The New Yorker Cartoon Dept. spreadsheets. 

    Specify PATH to a .xls file.

    Example:

                $ xls2csv ./data/raw/MyReport-72.xls --scrub --captions

                    # outputs clean file ./data/raw/MyReport-72.csv 
    """

    ##################[ check extension ]####################
    
    filename, ext = os.path.splitext(path)
    assert ext == ".xls", "Supply a path to a .xls file."


    ##################[ read in xls ]####################

    df = utils.read_xls(path)
    assert isinstance(df, pd.DataFrame)


    ##################[ validate format ]####################

    if captions:
        assert CAPTIONTEXT in list(df.columns), """Please use a New Yorker Caption Contest formatted spreadsheet.
                                                         # Columns not recognized: {}.""".format(set(df.columns) - (set(settings.columns) & set(df.columns)))


    ##################[ scrub personal data ]####################

    if scrub:
        df = utils.scrub(df)


    ##################[ coerce ascii ]##################

    df = df.apply(utils.asciidammit)


    ##################[ write csv ]##################

    if outpath is None:
        outpath = filename + ".csv"

    df.to_csv(outpath)

