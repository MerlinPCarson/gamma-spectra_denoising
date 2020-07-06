import sys
import time
import h5py
import argparse
import numpy as np

import matplotlib.pyplot as plt


def load_data(datafile, show_data=False):
    datasets = {}
    with h5py.File(datafile, 'r') as h5f:
        print(f'Loading data set for detector {h5f.keys()}')
        for det in h5f.keys():
            assert h5f[det]["spectrum"].shape == h5f[det]["noisy_spectrum"].shape, 'Mismatch between training examples and target examples'
            datasets[det] = {"keV": h5f[det]["keV"][()], "clean": h5f[det]["spectrum"][()], "noisy": h5f[det]["noisy_spectrum"][()]}

    if show_data:
        plot_data(datasets)

    return datasets

def plot_data(datasets):
    for det in datasets.keys():
        for spectrum, noisy_spectrum in zip(datasets[det]["clean"], datasets[det]["noisy"]):
            plt.plot(datasets[det]["keV"], spectrum) 
            plt.plot(datasets[det]["keV"], noisy_spectrum) 
            plt.show()

def datasets_stats(datasets):
    for det in datasets.keys():
        print(f'Dataset {det}')
        print(f'\tfeatures: {datasets[det]["keV"].shape}')
        print(f'\tclean spectra: {datasets[det]["clean"].shape}')
        print(f'\tnoisy spectra: {datasets[det]["clean"].shape}')

def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--datafile", help="data file containing templates", default="data/training.h5")
    arg = parser.parse_args()

    datasets = load_data(arg.datafile)

    print(f'{len(datasets)} loaded.')

    datasets_stats(datasets)

    print(f'\nScript completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
