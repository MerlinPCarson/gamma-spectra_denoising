import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from spectra_utils import data_load_normalized


def compare_denoising(spectra, denoised_spectra):
    for spec in glob(os.path.join(denoised_spectra, '*.json')):
        raw_spec = os.path.join(spectra, os.path.basename(spec).replace('-denoised', ''))
        print(f'loading {raw_spec}')
        keV, hits = data_load_normalized(raw_spec)
        print(f'loading {spec}')
        keVDN, hitsDN = data_load_normalized(spec)
        hitsDN = np.clip(hitsDN, a_min=0, a_max=None)


        # background subtraction
        print(f'loading {raw_spec} background')
        back_keV, back_hits = data_load_normalized(os.path.join(spectra, 'background.json'))
        # don't want negative intensities
        hitsBS = np.clip(hits-back_hits, a_min=0, a_max=None)

        # normalize magnitude of each version of the spectrum
        hits /= (hits ** 2).sum() ** 0.5
        hitsBS /= (hitsBS ** 2).sum() ** 0.5
        hitsDN /= (hitsDN ** 2).sum() ** 0.5

        # plot comparisons side by side
        fig, ax = plt.subplots(1,2)
        ax[0].plot(keV, hits, color='red', label='raw spectrum')
        ax[0].plot(keV, hitsBS, color='green', label='background subtracted spectrum', alpha=0.6)
        ax[0].legend()
        ax[0].set_xlabel('energy (keV)')
        ax[0].set_ylabel('Intensity')
        ax[1].plot(keV, hits, color='red', label='raw spectrum')
        ax[1].plot(keV, hitsDN, color='green', label='DNN denoised spectrum', alpha=0.6)
        ax[1].set_xlabel('energy (keV)')
        ax[1].set_ylabel('Intensity')
        ax[1].legend()

        #plt.xlabel('energy (keV)')
        #plt.ylabel('Intensity')
        plt.legend()
        plt.tight_layout()
        plt.show()

def parse_args():
    parser = argparse. ArgumentParser(description='Compare results of denosing and background subtraction')
    parser.add_argument('--spectra', type=str, default='../DTRA_SSLCA/psu_dtra/data/NaI-8-21-20/Uranium', help='directory of spectra or spectrum in json format')
    parser.add_argument('--denoised_spectra', type=str, default='denoised', 
                        help='directory containing denoised spectra in json format')
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
