import sys
import time
import json
import argparse
import numpy as np

from spectra_utils import data_load_normalized
from spectra_utils import generate_spectrum_SNR


def save_spectrum(keV, hits, spec_file, outfile):

    spec = json.load(open(spec_file, 'r'))

    spec['KEV'] = keV.tolist()
    spec['HIT'] = hits.tolist()

    json.dump(spec, open(outfile, 'w'))
    
def sum_spectra(spec1, spec2, snr):

    # load main signal spectra
    keV, hits1 = data_load_normalized(spec1)

    # load additional spectra
    _, hits2 = data_load_normalized(spec2)

    # no compton
    compton_hits = np.zeros_like(hits1)

    # generate new spectrum with sum of hits1 and hits2 at SNR params 
    # where hits1 is signal and hits2 is noise
    _ , noisy_spectrum, _ = generate_spectrum_SNR(hits1, hits2, compton_hits, snr)

    return keV, noisy_spectrum

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec1", type=str, help="main spectrum used for signal term", required=True)
    parser.add_argument("--spec2", type=str, help="spectrum used for noise term", required=True)
    parser.add_argument("--outfile", type=str, help="name for new spectrum", default='snr_spectrum.json')
    parser.add_argument('--snr', type=int, default=0, help='SNR for addition of spec1 and spec2')
    return parser.parse_args()


def main(args):
    start = time.time()

    keV, hits = sum_spectra(args.spec1, args.spec2, args.snr)

    save_spectrum(keV, hits, args.spec1, args.outfile)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
