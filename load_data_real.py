import os
import sys
import time
import h5py
import json
import argparse
import numpy as np
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as psnr

from spectra_utils import split_radionuclide_name
from plot_utils import plot_data


def load_spectrum(specfile):
    with open(specfile) as spec:
        spectrum = json.load(spec)

    return spectrum['KEV'], spectrum['HIT']

def load_spectra(spectra):

    test_data = {"keV": [], "hits": [], "spec_name": []}

    if os.path.isfile(spectra):
        test_files = [spectra]
    elif os.path.isdir(spectra):
        test_files = glob(os.path.join(spectra, '*.json'))
    else:
        sys.exit('Spectra path not found')

    for spectrum in test_files:
        keV, hits = load_spectrum(spectrum)
        test_data["keV"].append(keV)
        test_data["hits"].append(hits)
        test_data["spec_name"].append(spectrum)

    return test_data, test_files


def load_data(datafile, det, show_data=False):
    with h5py.File(datafile, 'r') as h5f:
        assert h5f[det]["spectrum"].shape == h5f[det]["noisy_spectrum"].shape, f'Mismatch between training examples and target examples'
        dataset = {"name": h5f[det]["name"][()], "keV": h5f[det]["keV"][()], "spectrum": h5f[det]["spectrum"][()], \
                            "noisy_spectrum": h5f[det]["noisy_spectrum"][()], "noise": h5f[det]["noise"][()], \
                            "compton_scale": h5f[det]["compton_scale"][()], "SNR": h5f[det]["SNR"][()]}
    if show_data:
        plot_data(dataset)

    return dataset

def dataset_stats(dataset, det):
    print(f'Dataset {det}:')
    print(f'\tfeatures: {dataset["keV"].shape}')
    print(f'\tclean spectra: {dataset["spectrum"].shape}')
    print(f'\tnoisy spectra: {dataset["noisy_spectrum"].shape}')
    print(f'\tnoise: {dataset["noise"].shape}')
    print(f'\tmin Compton scale: {np.min(dataset["compton_scale"])}')
    print(f'\tmax Compton scale: {np.max(dataset["compton_scale"])}')
    print(f'\tmin Noise scale: {np.min(dataset["SNR"])}')
    print(f'\tmax Noise scale: {np.max(dataset["SNR"])}')

    noisy_spectra = dataset['noisy_spectrum']
    clean_spectra = dataset['spectrum']

    min_psnr = 9999.0
    max_psnr = 0.0
    for clean, noisy in zip(clean_spectra, noisy_spectra):
        noisy_psnr = psnr(clean, noisy)
        if noisy_psnr < min_psnr:
            min_psnr = noisy_psnr
        if noisy_psnr > max_psnr:
            max_psnr = noisy_psnr

    print(f'\tmax PSNR {max_psnr:.2f} dB')
    print(f'\tmin PSNR {min_psnr:.2f} dB')


def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--datafile", help="data file containing templates", default="data/training.h5")
    parser.add_argument("-det", "--dettype", help="detector type", default="HPGe")
    parser.add_argument("-sf", "--showfigs", help="saves plots of data", default=False, action="store_true")
    arg = parser.parse_args()

    print(f'Loading data set from {arg.datafile}')
    dataset = load_data(arg.datafile, arg.dettype.upper(), show_data=arg.showfigs)

    print(f'{len(dataset["name"])} examples in dataset.')

    dataset_stats(dataset, arg.dettype)

    print(f'\nScript completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
