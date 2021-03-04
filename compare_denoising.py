import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from spectra_utils import data_load_normalized


def compare_denoising(spectra, denoised_spectra, minkev, maxkev, show_figs=True, save_figs=False):
    for spec in glob(os.path.join(denoised_spectra, '*.json')):
        raw_spec = os.path.join(spectra, os.path.basename(spec).replace('-denoised', ''))
        print(f'loading {raw_spec}')
        keV, hits = data_load_normalized(raw_spec)
        print(f'loading {spec}')
        keVDN, hitsDN = data_load_normalized(spec)

        # don't want negative intensities
        hitsDN = np.clip(hitsDN, a_min=0, a_max=None)


        # background subtraction
        print(f'loading {raw_spec} background')
        back_keV, back_hits = data_load_normalized(os.path.join(spectra, 'background.json'))
        # don't want negative intensities
        hitsBS = np.clip(hits-back_hits, a_min=0, a_max=None)

        # normalize magnitude of each version of the spectrum
        hitsNorm = hits / (hits ** 2).sum() ** 0.5
        hitsBSNorm = hitsBS / (hitsBS ** 2).sum() ** 0.5
        hitsDNNorm = hitsDN / (hitsDN ** 2).sum() ** 0.5

        # plot comparisons side by side
        fig, ax = plt.subplots(2,2)

        # compare raw and noise reduced spectrum
        ax[0][0].plot(keV, hits, color='blue', label='raw spectrum', alpha=0.6)
        ax[0][0].plot(keV, hitsBS, color='green', label='background subtracted spectrum', alpha=0.6)
        ax[0][0].legend()
        ax[0][0].set_xlabel('energy (keV)')
        ax[0][0].set_ylabel('Intensity')
        ax[0][0].set_xlim(minkev,maxkev)
        ax[0][1].plot(keV, hits, color='blue', label='raw spectrum', alpha=0.6)
        ax[0][1].plot(keVDN, hitsDN, color='red', label='DNN denoised spectrum', alpha=0.6)
        ax[0][1].set_xlabel('energy (keV)')
        ax[0][1].set_ylabel('Intensity')
        ax[0][1].set_xlim(minkev,maxkev)
        ax[0][1].legend()

        # compare background subtracted with DNN denoised spectrum
        ax[1][0].plot(keV, hitsBS, color='green', label='background subtracted spectrum', alpha=0.6)
        ax[1][0].plot(keV, hitsDN, color='red', label='DNN denoised spectrum', alpha=0.6)
        ax[1][0].legend()
        ax[1][0].set_xlabel('energy (keV)')
        ax[1][0].set_ylabel('Intensity')
        ax[1][0].set_xlim(minkev,maxkev)
        ax[1][1].plot(keV, hitsBSNorm, color='green', label='normalized background subtracted spectrum', alpha=0.6)
        ax[1][1].plot(keVDN, hitsDNNorm, color='red', label='normalized DNN denoised spectrum', alpha=0.6)
        ax[1][1].set_xlabel('energy (keV)')
        ax[1][1].set_ylabel('Intensity')
        ax[1][1].set_xlim(minkev,maxkev)
        ax[1][1].legend()

        fig.canvas.set_window_title(f"{os.path.basename(spec).replace('-denoised.json','')} spectrum")

        # make image full screen
        plt.tight_layout()
        fig = plt.gcf()
        fig.set_size_inches(18, 11)

        if save_figs:
            out_file = os.path.join(denoised_spectra, os.path.basename(spec).replace('.json', '.png'))
            print(f'Saving figure to {out_file}')
            plt.savefig(out_file, orientation='landscape', dpi=300)

        if show_figs:
            plt.show()

def convert_map(map_file, out_dir):
    df = pd.read_csv(map_file)

    # append denoised to end of each spectra name in map file
    df['Name'] = df['Name']+'-denoised'

    out_file = os.path.join(out_dir, os.path.basename(map_file))
    print(f'Saving SSLCA map file for denoised spectra to {out_file}')
    df.to_csv(out_file, index=False)

def parse_args():
    parser = argparse. ArgumentParser(description='Compare results of denosing and background subtraction')
    parser.add_argument('--spectra', type=str, default='../DTRA_SSLCA/psu_dtra/data/NaI-8-21-20/Uranium', help='directory of spectra or spectrum in json format')
    parser.add_argument('--denoised_spectra', type=str, default='denoised', 
                        help='directory containing denoised spectra in json format')
    parser.add_argument('--minkev', type=int, help='Min keV to plot', default=0)
    parser.add_argument('--maxkev', type=int, help='Max keV to plot', default=1500)
    parser.add_argument('--savefigs', help='saves plots of comparisons', default=False, action='store_true')
    parser.add_argument('--showfigs', help='show plots of comparisons', default=False, action='store_true')
    parser.add_argument('--map_file', type=str, help='SSLCA map file to convert to denoised spectra')
    args = parser.parse_args()

    assert os.path.exists(args.spectra), 'Directory containing raw spectra not found!'
    assert os.path.exists(args.denoised_spectra), 'Directory containing denoised spectra not found!'

    return args

def main(args):
    start = time.time()

    # create comparison figures of denoising algorithms
    if args.savefigs or args.showfigs:
        compare_denoising(args.spectra, args.denoised_spectra, args.minkev, args.maxkev, 
                          show_figs=args.showfigs, save_figs=args.savefigs)

    # convert SSLCA map file of spectra to map file of denoised spectra
    if args.map_file:
        convert_map(args.map_file, args.denoised_spectra)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
