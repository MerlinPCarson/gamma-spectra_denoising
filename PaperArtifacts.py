import os
import sys
import time
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_utils import compare_spectra
from spectra_utils import split_radionuclide_name
from spectra_utils import data_smooth
from spectra_utils import data_load_normalized


def compare_spectra(keV, spectra, titles, min_keV=-10, max_keV=1500, outfile=None, savefigs=False, showfigs=True):

    colors = ['blue', 'red', 'green']

    min_idx = np.searchsorted(keV, min_keV, side='left')
    max_idx = np.searchsorted(keV, max_keV, side='left')

    plt.figure(figsize=(20,10))
    for spectrum, title, color in zip(spectra, titles, colors):
        plt.plot(keV[min_idx:max_idx], spectrum[min_idx:max_idx], color=color, label=f'{title}')
    
    ax = plt.gca()
    ax.set_xlabel('Energy (keV)', fontsize=24, fontweight='bold', fontname='cmtt10')
    ax.set_ylabel('Intensity', fontsize=24, fontweight='bold', fontname='cmtt10')
    ax.set_xticks(np.arange(keV[min_idx], keV[max_idx], 100))
    ax.set_xticks(np.arange(keV[min_idx], keV[max_idx], 20), minor=True)
    ax.set_xlim([min_keV, max_keV])
    ax.grid(axis='x', which='major', alpha=0.5)
    ax.grid(axis='x', which='minor', alpha=0.2)

    plt.legend(fancybox=True, shadow=True, fontsize=24)
    plt.tight_layout()

    if savefigs and outfile:
        plt.savefig(outfile, format='pdf', dpi=300)

    if showfigs:
        plt.show()

    plt.close()

def load_spectrum(spectrum, normalize=False):
    
    spec = json.load(open(spectrum, 'r'))
    keV = spec['KEV']
    hits = spec['HIT']
    try:
        rn = spec['RADIONUCLIDE']
    except:
        rn = None

    if normalize:
        hits = np.array(hits, dtype=np.float)
        hits /= np.sum(hits)

    return keV, hits, rn

def show_spectra(keV, hits, rn, titles, min_keV, max_keV, outfile):

    fig_titles = []

    rn_num, rn_name = split_radionuclide_name(rn)

    for title in titles:
        fig_titles.append("${}^{"+rn_num+"}{"+rn_name+"}$ " + title)

    compare_spectra(keV, hits, fig_titles, min_keV, max_keV, outfile, savefigs=True)

def nai_effeciency(keV, cutoff=121.8, max_eff=1400):
    if keV <= cutoff:
        eff = 0.0036*keV + 0.5641
        return max(eff,0.0)
    else:
        eff = 8.9946*(keV**(-0.462))
        return max(eff,0.0)

def plot_spectra_snrs(spec1, spec2, min_keV, max_keV, outdir):


    keV, hits1 = data_load_normalized(spec1)
    _, hits2 = data_load_normalized(spec2)

    _, hits_back = data_load_normalized(os.path.join(os.path.dirname(spec1), 'background.json'))

    hits1_sub = hits1 - hits_back
    hits2_sub = hits2 - hits_back

    rms1 = np.sqrt(np.mean(hits1_sub**2))
    rms2 = np.sqrt(np.mean(hits2_sub**2))
    rms_back = np.sqrt(np.mean(hits_back**2))

    snr1 = 20*np.log10(rms1/rms_back)
    snr2 = 20*np.log10(rms2/rms_back)

    titles = [f'spectrum ({snr1:.2f} dB)', f'spectrum ({snr2:.2f} dB)']
    rn = '235U'

    fig_titles = []
    rn_num, rn_name = split_radionuclide_name(rn)
    for title in titles:
        fig_titles.append("${}^{"+rn_num+"}{"+rn_name+"}$ " + title)
    outfile = os.path.join(outdir, f'{rn}_spectra_with_snrs.pdf')

    # plot comparisons side by side
    fig, ax = plt.subplots(2,1)

    keV, hits1 = data_load_normalized(spec1, normalize=False)
    _, hits2 = data_load_normalized(spec2, normalize=False)

    ax[0].plot(keV, hits1, color='red', label=f'{fig_titles[0]}')
    ax[1].plot(keV, hits2, color='blue', label=f'{fig_titles[1]}')

    for i in range(len(ax)):
        ax[i].set_xlabel('Energy (keV)', fontsize=14, fontweight='bold', fontname='cmtt10')
        ax[i].set_ylabel('Counts', fontsize=14, fontweight='bold', fontname='cmtt10')
        ax[i].set_xticks(np.arange(min_keV, max_keV, 100))
        ax[i].set_xticks(np.arange(min_keV, max_keV, 20), minor=True)
        ax[i].set_xlim([min_keV, max_keV])
        ax[i].grid(axis='x', which='major', alpha=0.5)
        ax[i].grid(axis='x', which='minor', alpha=0.2)
        ax[i].legend(fancybox=True, shadow=True, fontsize=12)

    #fig.set_size_inches(8.5, 11)
    plt.tight_layout()
    plt.savefig(outfile, format='pdf', dpi=300)
    plt.show()

    
def parse_args():

    parser = argparse.ArgumentParser(description='Plot multiple spectra')
    parser.add_argument("-cf", "--configfile", help="configuration file for generating data", default="config_data_real.json")
    parser.add_argument("-out", "--outdir", help="output directory for figures", default="figures")
    parser.add_argument("-det", "--dettype", help="detector type", default="NaI")
    parser.add_argument('--nndc_temp', type=str, default='data/mcnp_spectra/spectra/compton/152Eu_1kc_1s-no-compton.json', 
                        help='photoelectric peaks for a radionuclide')
    parser.add_argument('--nndc_temp_compton', type=str, default='data/mcnp_spectra/spectra/compton/152Eu_1kc_1s-compton-only.json', 
                        help='Compton for a radionuclide')
    parser.add_argument('--nndc_preproc_temp', type=str, default='data/mcnp_spectra/preproc_spectra/152Eu_1kc_1s-nocompton.json', 
                        help='photoelectric peaks for a radionuclide')
    parser.add_argument('--nndc_preproc_temp_compton', type=str, default='data/mcnp_spectra/preproc_spectra/152Eu_1kc_1s-compton-only.json', 
                        help='Compton for a radionuclide')
    parser.add_argument('--wide_temp', type=str, default='data/mcnp_spectra/preproc_spectra/152Eu_1mc_1s-nocompton.json', 
                        help='photoelectric peaks for a radionuclide')
    parser.add_argument('--temp_compton', type=str, default='figures/152Eu_template.npy', 
                        help='photoelectric peaks for a radionuclide')
    parser.add_argument('--highsnr_spec', type=str, default='../DTRA_SSLCA/psu_dtra/data/NaI-8-21-20/Uranium/U2in300s.json', 
                        help='photoelectric peaks for a radionuclide')
    parser.add_argument('--lowsnr_spec', type=str, default='../DTRA_SSLCA/psu_dtra/data/NaI-8-21-20/Uranium/U24in60s.json', 
                        help='photoelectric peaks for a radionuclide')
    parser.add_argument('--spec_augment', type=str, default='generated_spectra/152Eu_snr_spectrum.json', 
                        help='spectrum for showing data augmentation')
    parser.add_argument('--min_keV', type=float, default=0.0, help='minimum keV to plot')
    parser.add_argument('--max_keV', type=float, default=1500.0, help='maximum keV to plot')
    parser.add_argument('--normalize', default=False, action='store_true', help='normalize all spectra')
    args = parser.parse_args()

    return args

def main(args):
    start = time.time()

#    compton = pickle.load(open(args.temp_compton, 'rb'))
#    nocompton = pickle.load(open(args.temp_compton.replace('.npy','_nocompton.npy'), 'rb'))
#
#    # plot a template 
#    titles = ['SSLCA template without Compton', 'SSLCA template with compton']
#    outfile = os.path.join(args.outdir, 'template_compton.pdf')
#    show_spectra(compton['keV'], [nocompton['intensity'], compton['intensity']], '152Eu', titles, args.min_keV, args.max_keV, outfile)
#
#    # load configuration parameters
#    with open(args.configfile, 'r') as cfile:
#        config = json.load(cfile)['DETECTORS'][args.dettype.upper()]
#
#    # load a radionuclide template
#    keV, hits, rn, = load_spectrum(args.nndc_temp, True)
#
#    # plot a raw template 
#    titles = ['NNDC gamma-ray table']
#    outfile = os.path.join(args.outdir, 'template.pdf')
#    show_spectra(keV, [hits], rn, titles, args.min_keV, args.max_keV, outfile)
#
#    # apply efficiency
#    hits_eff = [nai_effeciency(eV)*counts for eV, counts in zip(keV, hits)]
#
#    # compare template pre/post efficiency 
#    titles = ['NNDC gamma-ray table', 'table scaled for efficiency']
#    outfile = os.path.join(args.outdir, 'compare_efficiency.pdf')
#    show_spectra(keV, [hits, hits_eff], rn, titles, args.min_keV, args.max_keV, outfile)
#
#    # apply Gaussian broadening
#    hits_broad = np.array(data_smooth(keV, hits, **config['SMOOTH_PARAMS']))
#    hits_broad /= np.sqrt(np.sum(hits_broad**2))
#
#    # compare template pre/post efficiency 
#    titles = ['table scaled for efficiency', 'with Gaussian broadening']
#    outfile = os.path.join(args.outdir, 'compare_eff_broad.pdf')
#    show_spectra(keV, [hits_eff, hits_broad], rn, titles, args.min_keV, args.max_keV, outfile)
#
#    # plot 2 spectra and associated SNRs
#    plot_spectra_snrs(args.lowsnr_spec, args.highsnr_spec, 0, 1500, args.outdir)
#
#    # compare PE and Compton in MCNP simulations 
#    # load PE intensity 
#    keV, hits_pe, rn, = load_spectrum(args.nndc_temp, normalize=True)
#    # load Compton only intensity 
#    _, hits_comp, rn, = load_spectrum(args.nndc_temp_compton, normalize=True)
#
#    titles = ['photoelectric', 'Compton']
#    outfile = os.path.join(args.outdir, 'compare_compton_pe.pdf')
#    show_spectra(keV, [hits_pe, hits_comp], rn, titles, args.min_keV, args.max_keV, outfile)
#
#    # compare broadened PE and Compton in MCNP simulations 
#    # load PE intensity 
#    keV, hits_pe, rn, = load_spectrum(args.nndc_preproc_temp, normalize=False)
#    # load Compton only intensity 
#    _, hits_comp, rn, = load_spectrum(args.nndc_preproc_temp_compton, normalize=False)
#
#    titles = ['photoelectric', 'Compton']
#    outfile = os.path.join(args.outdir, 'compare_compton_pe_broad.pdf')
#    show_spectra(keV, [hits_pe, hits_comp], rn, titles, args.min_keV, args.max_keV, outfile)

    # compare broadened PE and Compton in MCNP simulations 
    # load PE intensity 
    keV, hits_pe, rn, = load_spectrum(args.spec_augment, normalize=False)
    hits_pe_left = np.roll(hits_pe, -15)
    hits_pe_right = np.roll(hits_pe, 15)

    titles = ['spectrum', 'left shifted', 'right shifted']
    outfile = os.path.join(args.outdir, 'shift_augment.pdf')
    show_spectra(keV, [hits_pe, hits_pe_left, hits_pe_right], rn, titles, args.min_keV, args.max_keV, outfile)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
