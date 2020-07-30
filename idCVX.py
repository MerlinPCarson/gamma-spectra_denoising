import sys
import time
import h5py
import argparse
import numpy as np

from spectra_utils import split_radionuclide_name, plot_data
from load_templates import load_templates
import cvx

def load_data(datafile, det, show_data=False):
    with h5py.File(datafile, 'r') as h5f:
        assert h5f[det]["spectrum"].shape == h5f[det]["noisy_spectrum"].shape, 'Mismatch between training examples and target examples'
        dataset = {"name": h5f[det]["name"][()], "keV": h5f[det]["keV"][()], "clean": h5f[det]["spectrum"][()], \
                            "noisy": h5f[det]["noisy_spectrum"][()], "noise": h5f[det]["noise"][()], \
                            "compton_scale": h5f[det]["compton_scale"][()], "noise_scale": h5f[det]["noise_scale"][()]}
    if show_data:
        plot_data(dataset)

    return dataset

def load_template(datafile, det, show_data=False):
    with h5py.File(datafile, 'r') as h5f:
        dataset = {"name": h5f[det]["name"][()], "keV": h5f[det]["keV"][()], "intensity": h5f[det]["spectrum"][()]}

    if show_data:
        plot_data(dataset)

    return dataset

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
    parser.add_argument("-df", "--datafile", help="data file containing dataset", default="data/training.h5")
    parser.add_argument("-dt", "--templatefile", help="data file containing templates", default="data/templates.h5")
    parser.add_argument("-det", "--dettype", help="detector type", default="HPGe")
    parser.add_argument("--norm", type=str,
                  choices=["L1", "L2", "Linf"],
                  default="L1",
                  help="error norm (default L1) -- see docs")
    parser.add_argument("--complete", action='store_true',
                  help="require prevalences to sum to 1")
    parser.add_argument("-sf", "--showfigs", help="saves plots of data", default=False, action="store_true")
    arg = parser.parse_args()

    # Pick a valid norm.
    norms = {"L1" : 1, "L2" : 2, "Linf" : "inf"}
    norm = norms[arg.norm]

    # load bases, clean radionuclide templates
    templates = load_templates(arg.templatefile, arg.dettype.upper())
    
    dataset = load_data(arg.datafile, arg.dettype.upper(), show_data=arg.showfigs)

    print(f'{len(dataset)} loaded.')

    dataset_stats(dataset, arg.dettype)

    #noisy_spectra = dataset['noisy']

    bases = np.array(templates['intensity'])

    preds = []
    for i, (name, spectra) in enumerate(zip(dataset['name'], dataset['noisy']), start=1):
        # Decompose measured spectrum
        (ampl0, q) = cvx.decompose(bases, spectra, arg.complete, norm)

        pred = np.argmax(ampl0)

        preds.append(pred)

        print(f"{i}: Pred is {templates['name'][pred]}, Target is {name.decode('utf-8')}")

    preds = np.array([templates['name'][pred] for pred in preds])
    targets = np.array([dataset['name'][i].decode('utf-8') for i in range(len(dataset['name']))])

    accuracy = np.sum(preds==targets)/len(preds)
    print(f'Identification Accuracy: {accuracy}')

    print(f'\nScript completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
