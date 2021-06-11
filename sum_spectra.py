import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from spectra_utils import data_load_normalized
from spectra_utils import generate_spectrum_SNR
from spectra_utils import split_radionuclide_name


def compare_spectra(keV, spectra, titles, min_keV=0, max_keV=1500, outfile=None, savefigs=False, showfigs=True):

    colors = ['blue', 'red', 'purple']

    min_idx = np.searchsorted(keV, min_keV, side='left')
    max_idx = np.searchsorted(keV, max_keV, side='left')

    #plt.figure(figsize=(20,10))
    # plot comparisons side by side
    fig, ax = plt.subplots(len(spectra), 1)
    for i, (spectrum, title, color) in enumerate(zip(spectra, titles, colors)):
        ax[i].plot(keV[min_idx:max_idx], spectrum[min_idx:max_idx], color=color, label=f'{title}')
    
        #ax[i] = plt.gca()
        ax[i].set_xlabel('Energy (keV)', fontsize=16, fontweight='bold', fontname='cmtt10')
        ax[i].set_ylabel('Intensity', fontsize=16, fontweight='bold', fontname='cmtt10')
        ax[i].set_xticks(np.arange(keV[min_idx], keV[max_idx], 100))
        ax[i].set_xticks(np.arange(keV[min_idx], keV[max_idx], 20), minor=True)
        ax[i].set_xlim([min_keV, max_keV])
        ax[i].grid(axis='x', which='major', alpha=0.5)
        ax[i].grid(axis='x', which='minor', alpha=0.2)

        ax[i].legend(fancybox=True, shadow=True, fontsize=14)

    plt.tight_layout()

    fig.set_size_inches(8.5, 11)
    if savefigs and outfile:
        plt.savefig(outfile, format='pdf', dpi=300)

    if showfigs:
        plt.show()

    plt.close()

def save_spectrum(keV, hits, spec_file, outfile):

    spec = json.load(open(spec_file, 'r'))

    spec['KEV'] = keV.tolist()
    spec['HIT'] = hits.tolist()

    json.dump(spec, open(outfile, 'w'))
    
def sum_spectra(spec1, spec2, snr, rn, outfile, showfigs=False, savefigs=False, smooth=True):

    # load main signal spectra
    keV, hits1 = data_load_normalized(spec1)
    #hits1 = hits1 / np.sqrt(np.sum(hits1**2))

    # load additional spectra
    _, hits2 = data_load_normalized(spec2)

    # no compton
    compton_hits = np.zeros_like(hits1)

    # generate new spectrum with sum of hits1 and hits2 at SNR params 
    # where hits1 is signal and hits2 is noise
    _ , noisy_spectrum, noise = generate_spectrum_SNR(hits1, hits2, compton_hits, snr)

    if smooth: 
        # 2 smoothing window size, 1 for each half
        windowsize = 10 
        noisy_spectrum[:len(noisy_spectrum)//2] = np.convolve(noisy_spectrum[:len(noisy_spectrum)//2], np.ones((windowsize,))/windowsize, mode='same').tolist()
        windowsize = 40 
        noisy_spectrum[len(noisy_spectrum)//2:] = np.convolve(noisy_spectrum[len(noisy_spectrum)//2:], np.ones((windowsize,))/windowsize, mode='same').tolist()
        plt.plot(keV[:4233], noisy_spectrum[:4233], color='red', linewidth=5.0)
        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(18, 8.5) 
        plt.savefig('smooth.png', bbox_inches='tight', pad_inches=0)
        plt.show()

    if showfigs or savefigs:
        fig_titles = []
        titles = ['source', 'background', f'noisy spectrum ({snr} dB)']
        rn_num, rn_name = split_radionuclide_name(rn)
        for title in titles:
            fig_titles.append("${}^{"+rn_num+"}{"+rn_name+"}$ " + title)

        # remove RN from background title
        fig_titles[1] = titles[1]

        outfile = os.path.join('figures', outfile.replace('.json', '.pdf'))
        compare_spectra(keV, [hits1, noise, noisy_spectrum], fig_titles, outfile=outfile, savefigs=savefigs, showfigs=showfigs)

    return keV, noisy_spectrum

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec1", type=str, default='data/mcnp_spectra/preproc_spectra/152Eu_1kc_1s-nocompton.json', help="main spectrum used for signal term")
    parser.add_argument("--spec2", type=str, default='background/NaI/BG1200s-U.json', help="spectrum used for noise term")
    parser.add_argument("--outfile", type=str, help="name for new spectrum", default='snr_spectrum.json')
    parser.add_argument('--snr', type=int, default=-15, help='SNR for addition of spec1 and spec2')
    parser.add_argument('--rn', type=str, default='152Eu', help='Radionuclide present in spectrum 1')
    parser.add_argument("-sf", "--showfigs", help="show plots of each spectrum", default=True, action="store_true")
    parser.add_argument("-sp", "--savefigs", help="save plots of each spectrum", default=True, action="store_true")
    return parser.parse_args()


def main(args):
    start = time.time()

    outfile = f'{args.rn}-{args.outfile}'

    keV, hits = sum_spectra(args.spec1, args.spec2, args.snr, args.rn, outfile, args.showfigs, args.savefigs)

    save_spectrum(keV, hits, args.spec1, args.outfile)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
