import os
import sys
import time
import json
import argparse
import numpy as np

from plot_utils import compare_spectra


def verify_spectra(spectra):

    spectra = spectra.split(',')

    for spectrum in spectra:
        assert os.path.isfile(spectrum), f'{spectrum} not found!'

    return spectra

def plot_spectra(spectra, min_keV, max_keV):

    keV = []
    hits = []
    titles = []

    # load all spectra into memory
    for spectrum in spectra:
        spec = json.load(open(spectrum, 'r'))
        keV.append(spec['KEV'])
        hits.append(spec['HIT'])
        titles.append(os.path.basename(spectrum).replace('.json',''))

    # use first spectrum's calibration
    base_keV = np.array(keV[0])

    # verify all other spectra have the same calbration
    for i in range(1, len(keV)):
        assert np.array_equal(base_keV, keV[i]), 'calibrations do not match between all spectra'

    # plot all spectra
    compare_spectra(base_keV, hits, titles, min_keV, max_keV)

def parse_args():

    parser = argparse.ArgumentParser(description='Plot multiple spectra')
    parser.add_argument('--spectra', type=str, default='data/mcnp_spectra/preproc_spectra/137Cs_10kc_1s-compton.json,data/mcnp_spectra/preproc_spectra/137Cs_10kc_1s-no-compton.json', help='Comma seperated list of spectra to compare')
    parser.add_argument('--min_keV', type=float, default=0.0, help='minimum keV to plot')
    parser.add_argument('--max_keV', type=float, default=1500.0, help='maximum keV to plot')
    args = parser.parse_args()

    return args

def main(args):
    start = time.time()

    spectra = verify_spectra(args.spectra)

    plot_spectra(spectra, args.min_keV, args.max_keV)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
