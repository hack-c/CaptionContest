import sys
from optparse import OptionParser
import numpy as np
import pandas as pd
from pygeocoder import Geocoder, GeocoderError


op = OptionParser()
op.add_option("--csv",
              dest="csv_path", type="str",
              help="Path to the csv for the relevant caption contest.")

print(__doc__)
op.print_help()
(opts, args) = op.parse_args()


if __name__ == '__main__':

    fname = opts.csv_path.split('/')[-1][:-4] + '_coords.csv'

    print("preprocessing...")
    print()

    places          = pd.read_csv(opts.csv_path)[['City', 'State', 'Country']]
    places.City     = places.City.apply(str.upper)
    places.State    = places.City.apply(str.upper)
    places.Country  = places.City.apply(str.upper)
    places          = places[places.apply(lambda s: type(s.City) is str and type(s.State) is str and type(s.Country) is str, axis=1)]

    print("geocoding...")
    print()

    coords = []

    for row in places.iterrows():
        try:
            result = Geocoder.geocode(row[1].City + ', ' + row[1].State + ', ' + row[1].Country)
            coords.append(result[0].coordinates)
            sys.stdout.write('.'); sys.stdout.flush()
        except GeocoderError:
            sys.stdout.write('E'); sys.stdout.flush()
            continue

    print("writing to data/processed/" + fname + "...")
    print()

    coords, counts = np.unique(np.array([(round(x[0], 2), round(x[1], 2)) for x in coords]), return_counts=True)
    pd.DataFrame(zip(coords,counts)).to_csv('data/processed/' + fname)

    print("done.")
    print()


