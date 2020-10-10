import os
import sys
import time
import h5py
import argparse
import numpy as np

from spectra_utils import split_radionuclide_name, plot_data
from load_templates import load_templates
from load_data_real import load_data, dataset_stats
import cvx


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

    print(f'{len(dataset["name"])} loaded.')

    dataset_stats(dataset, arg.dettype)

    bases = np.array(templates['intensity'])

    preds = []
    for i, (name, spectrum) in enumerate(zip(dataset['name'], dataset['noisy_spectrum']), start=1):
        # Decompose measured spectrum
        (ampl0, q) = cvx.decompose(bases, spectrum, arg.complete, norm)

        # determine which template matched the noisy spectrum best
        pred = np.argmax(ampl0)

        preds.append(pred)

        print(f"{i}: Pred is {templates['name'][pred]}, Target is {name.decode('utf-8')} ({ampl0[pred]})")

    preds = np.array([templates['name'][pred] for pred in preds])
    targets = np.array([dataset['name'][i].decode('utf-8') for i in range(len(dataset['name']))])

    accuracy = np.sum(preds==targets)/len(preds)
    print(f'Identification Accuracy: {accuracy}')
    
    results = preds == targets
    outname = os.path.basename(arg.datafile).replace('.h5','.npy')
    with open(outname, 'wb') as f:
        np.save(f, results)

    print(f'\nScript completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
