import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from dtraUtil import data_load_normalized


def compare_denoising(spectra, denoised_spectra):
    for spec in glob(os.path.join(denoised_spectra, '*.json')):
        raw_spec = os.path.join(spectra, os.path.basename(spec).replace('-denoised', ''))
        print(f'loading {raw_spec}')
        keV, hits = data_load_normalized(raw_spec)
        print(f'loading {spec}')
        keVDN, hitsDN = data_load_normalized(spec)

        # background subtraction
        print(f'loading {raw_spec} background')
        back_keV, back_hits = data_load_normalized(os.path.join(spectra, 'background.json'))
        # don't want negative intensities
        hits = np.clip(hits-back_hits, a_min=0, a_max=None)  

        # normalize magnitude
        hits = np.array(hits)
        hits /= (hits ** 2).sum() ** 0.5
        hitsDN = np.array(hitsDN)
        hitsDN /= (hitsDN ** 2).sum() ** 0.5

        plt.plot(keV, hits, label='raw spectrum')
        plt.plot(keVDN, hitsDN, label='denoised spectrum', alpha=0.6)
        plt.legend()
        plt.show()

def parse_args():
    parser = argparse. ArgumentParser(description='Compare results of denosing and background subtraction')
    parser.add_argument('--spectra', type=str, default='spectra/NaI/Uranium', help='directory of spectra or spectrum in json format')
    parser.add_argument('--denoised_spectra', type=str, default='spectra/NaI/Uranium', 
                        help='directory containing denoised spectra')
    parser.add_argument('--savefigs', help='saves plots of results', default=False, action='store_true')
    args = parser.parse_args()

    assert os.path.exists(args.spectra), 'Directory containing raw spectra not found!'
    assert os.path.exists(args.denoised_spectra), 'Directory containing denoised spectra not found!'

    return args

def main(args):
    start = time.time()

    compare_denoising(args.spectra, args.denoised_spectra)


    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
