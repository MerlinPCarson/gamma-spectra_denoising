import os
import numpy as np
import matplotlib.pyplot as plt

from spectra_utils import split_radionuclide_name


def compare_spectra(keV, spectra, titles, min_keV=-10, max_keV=1500, outfile=None, savefigs=False, showfigs=True):

    min_idx = np.searchsorted(keV, min_keV, side='left')
    max_idx = np.searchsorted(keV, max_keV, side='left')

    plt.figure(figsize=(20,10))
    for spectrum, title in zip(spectra, titles):
        plt.plot(keV[min_idx:max_idx], spectrum[min_idx:max_idx], label=f'{title}')
    
    ax = plt.gca()
    ax.set_xlabel('Energy (keV)', fontsize=18, fontweight='bold', fontname='cmtt10')
    ax.set_ylabel('Intensity', fontsize=18, fontweight='bold', fontname='cmtt10')
    ax.set_xticks(np.arange(keV[min_idx], keV[max_idx], 100))
    ax.set_xticks(np.arange(keV[min_idx], keV[max_idx], 20), minor=True)
    ax.set_xlim([keV[min_idx], keV[max_idx]])
    ax.grid(axis='x', which='major', alpha=0.5)
    ax.grid(axis='x', which='minor', alpha=0.2)
    plt.legend(fancybox=True, shadow=True, fontsize=11)
    
    #plt.axis('off')
    plt.tight_layout()

    if savefigs and outfile:
        plt.savefig(outfile)

    if showfigs:
        plt.show()

    plt.close()

def plot_data(dataset):
    for i in range(len(dataset["name"])):
        plt.plot(dataset["keV"], dataset["spectrum"][i], label='clean spectrum') 
        plt.plot(dataset["keV"], dataset["noisy_spectrum"][i], label='noisy spectrum') 
        plt.plot(dataset["keV"], dataset["noise"][i], label='noise') 
        rn_num, rn_name = split_radionuclide_name(dataset["name"][i].decode('utf-8'))
        rn = "${}^{"+rn_num+"}{"+rn_name+"}$"
        plt.title(f'{rn} with Compton scale: {dataset["compton_scale"][i]}, SNR {dataset["SNR"][i]}')
        plt.legend()
        plt.show()

def plot_spectrum(keV, intensity, rn, outdir, title='', show_plot=False):

    rn_num, rn_name = split_radionuclide_name(rn)

    plt.figure(figsize=(20,10))
    plt.plot(keV, intensity, label="${}^{"+rn_num+"}{"+rn_name+"}$ " +title+ " Spectrum", color='blue')
    
    ax = plt.gca()
    ax.set_xlabel('Energy (keV)', fontsize=18, fontweight='bold', fontname='cmtt10')
    ax.set_ylabel('Intensity', fontsize=18, fontweight='bold', fontname='cmtt10')
    ax.set_xticks(np.arange(keV[0], keV[-1], 50))
    ax.set_xticks(np.arange(keV[0], keV[-1], 10), minor=True)
    ax.grid(axis='x', which='major', alpha=0.5)
    ax.grid(axis='x', which='minor', alpha=0.2)

    plt.legend(fancybox=True, shadow=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, rn) + '.png', format='png')

    if show_plot:
        plt.show()

    plt.close()

def compare_three_spectra(keV, spectrum, noisy_spectrum, noise, rn, outdir, title1 = '', title2= '', title3='', show_plot=False):

    rn_num, rn_name = split_radionuclide_name(rn)

    plt.figure(figsize=(20,10))
    plt.plot(keV, spectrum, label="${}^{"+rn_num+"}{"+rn_name+"}$ "+title1+ " Spectrum", color='blue')
    plt.plot(keV, noisy_spectrum, alpha=0.5, label="${}^{"+rn_num+"}{"+rn_name+"}$ "+title2+ " Spectrum", color='red')
    plt.plot(keV, noise, alpha=0.5, label=title3, color='green')
    
    ax = plt.gca()
    ax.set_xlabel('Energy (keV)', fontsize=18, fontweight='bold', fontname='cmtt10')
    ax.set_ylabel('Intensity', fontsize=18, fontweight='bold', fontname='cmtt10')
    ax.set_xticks(np.arange(keV[0], keV[-1], 100))
    ax.set_xticks(np.arange(keV[0], keV[-1], 20), minor=True)
    ax.set_xlim([0,keV[-1]])
    ax.grid(axis='x', which='major', alpha=0.5)
    ax.grid(axis='x', which='minor', alpha=0.2)

    plt.legend(fancybox=True, shadow=True, fontsize=11)
    plt.tight_layout()

    if show_plot:
        plt.show()

    plt.close()

def compare_results(keV, spectrum, noisy_spectrum, denoised_spectrum, snr_improve, outdir, fname, show_plot=False):

    plt.figure(figsize=(20,10))
    plt.plot(keV, noisy_spectrum, label="Noisy Spectrum", color='green', linestyle='-.')
    plt.plot(keV, spectrum, label="Target Spectrum", color='blue')
    plt.plot(keV, denoised_spectrum, alpha=0.8, label=f"Denoised Spectrum ({snr_improve:.2f} dB)", color='red', linestyle='--')
    
    ax = plt.gca()
    ax.set_xlabel('Energy (keV)', fontsize=18, fontweight='bold', fontname='cmtt10')
    ax.set_ylabel('Intensity', fontsize=18, fontweight='bold', fontname='cmtt10')
    ax.set_xticks(np.arange(keV[0], keV[-1], 100))
    ax.set_xticks(np.arange(keV[0], keV[-1], 20), minor=True)
    ax.grid(axis='x', which='major', alpha=0.5)
    ax.grid(axis='x', which='minor', alpha=0.2)

    plt.legend(fancybox=True, shadow=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname) + '-results.png', format='png')

    if show_plot:
        plt.show()

    plt.close()