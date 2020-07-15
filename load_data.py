import sys
import time
import h5py
import argparse
import numpy as np

import matplotlib.pyplot as plt

from spectra_utils import split_radionuclide_name


def load_data(datafile, det, show_data=False):
    with h5py.File(datafile, 'r') as h5f:
        assert h5f[det]["spectrum"].shape == h5f[det]["noisy_spectrum"].shape, 'Mismatch between training examples and target examples'
        dataset = {"name": h5f[det]["name"][()], "keV": h5f[det]["keV"][()], "clean": h5f[det]["spectrum"][()], \
                            "noisy": h5f[det]["noisy_spectrum"][()], "noise": h5f[det]["noise"][()], \
                            "compton_scale": h5f[det]["compton_scale"][()], "noise_scale": h5f[det]["noise_scale"][()]}
    if show_data:
        plot_data(dataset)

    return dataset

def plot_data(dataset):
    for i in range(len(dataset["name"])):
        plt.plot(dataset["keV"], dataset["clean"][i], label='clean spectrum') 
        plt.plot(dataset["keV"], dataset["noisy"][i], label='noisy spectrum') 
        plt.plot(dataset["keV"], dataset["noise"][i], label='noise') 
        rn_num, rn_name = split_radionuclide_name(dataset["name"][i].decode('utf-8'))
        rn = "${}^{"+rn_num+"}{"+rn_name+"}$"
        plt.title(f'{rn} with Compton scale: {dataset["compton_scale"][i]}, noise scale {dataset["noise_scale"][i]}')
        plt.legend()
        plt.show()

def dataset_stats(dataset, det):
    print(f'Dataset {det}')
    print(f'\tfeatures: {dataset["keV"].shape}')
    print(f'\tclean spectra: {dataset["clean"].shape}')
    print(f'\tnoisy spectra: {dataset["noisy"].shape}')
    print(f'\tnoise: {dataset["noise"].shape}')
    print(f'\tmin Compton scale: {np.min(dataset["compton_scale"])}')
    print(f'\tmax Compton scale: {np.max(dataset["compton_scale"])}')
    print(f'\tmin Noise scale: {np.min(dataset["noise_scale"])}')
    print(f'\tmax Noise scale: {np.max(dataset["noise_scale"])}')

def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--datafile", help="data file containing templates", default="data/training.h5")
    parser.add_argument("-det", "--dettype", help="detector type", default="HPGe")
    parser.add_argument("-sf", "--showfigs", help="saves plots of data", default=False, action="store_true")
    arg = parser.parse_args()

    dataset = load_data(arg.datafile, arg.dettype.upper(), show_data=arg.showfigs)

    print(f'{len(dataset)} loaded.')

    dataset_stats(dataset, arg.dettype)

    print(f'\nScript completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
