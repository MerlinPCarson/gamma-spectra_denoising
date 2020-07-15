import os
import sys
import time
import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm

from spectra_utils import load_radionuclide_nndc, generate_spectrum, compare_spectra


def load_nndc_tables(nndc_dir, radionuclides):

    nndc_tables = {}

    for rn in radionuclides:
        keV, intensity = load_radionuclide_nndc(nndc_dir, rn)
        nndc_tables[rn] = {"keV": keV, "intensity": intensity}
        
    return nndc_tables
    

def generate_spectra(config, nndc_tables, outdir, savefigs, params):

    spectra = {"name": [], "spectrum": [], "noisy_spectrum": [], "noise": [], "compton_scale": [], "noise_scale": []} 
    for rn_name, rn_values in tqdm(nndc_tables.items()):
        for compton_scale, noise_scale in params:
            spectrum_keV, spectrum, noisy_spectrum, noise = generate_spectrum(rn_values, config, compton_scale=compton_scale, noise_scale=noise_scale)
            spectra["name"].append(rn_name.encode('utf-8'))
            spectra["spectrum"].append(spectrum)
            spectra["noisy_spectrum"].append(noisy_spectrum)
            spectra["noise"].append(noise)
            spectra["compton_scale"].append(compton_scale)
            spectra["noise_scale"].append(noise_scale)
            if savefigs:
                compare_spectra(spectrum_keV, spectrum, noise, rn_name, outdir, show_plot=False)

    spectra["keV"] = spectrum_keV

    return spectra 
        
def save_dataset(dettype, dataset, outfile):
    with h5py.File(outfile, 'a') as h5f:
        try:
            h5f.create_group(dettype)
        except:
            pass
        for k, v in dataset.items():
            try:
                h5f[dettype].create_dataset(k, data=v)
            except:
                del h5f[dettype][k]
                h5f[dettype].create_dataset(k, data=v)


def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--configfile", help="configuration file for generating data", default="config_data.json")
    parser.add_argument("-out", "--outfile", help="output file for data", default="data/training.h5")
    #parser.add_argument("-det", "--dettype", help="detector type", default="HPGe,NaI,CZT")
    parser.add_argument("-det", "--dettype", help="detector type", default="HPGe")
    parser.add_argument("-nndc", "--nndctables", help="location of NNDC tables data",  default="nuclides-nndc")
    parser.add_argument("-sf", "--savefigs", help="saves plots of templates", action="store_true")
    parser.add_argument("-maxn", "--maxnoise", help="maximum noise scale", default=1.0, type=float)
    parser.add_argument("-minn", "--minnoise", help="minimum noise scale", default=0.1, type=float)
    parser.add_argument("-nn", "--numnoise", help="number of noise scales between 0.0 and max noise scale", default=2, type=int)
    parser.add_argument("-maxc", "--maxcompton", help="maximum compton sacle", default=0.5, type=float)
    parser.add_argument("-minc", "--mincompton", help="minimum compton sacle", default=0.1, type=float)
    parser.add_argument("-nc", "--numcompton", help="number of compton scales between 0.0 and max compton scale", default=2, type=int)
    arg = parser.parse_args()

    outdir = os.path.dirname(arg.outfile)
    outfile = arg.outfile

    # load configuration parameters
    with open(arg.configfile, 'r') as cfile:
        config = json.load(cfile)

    # make output dir if it does not exist
    os.makedirs(outdir, exist_ok=True)

    # load NNDC tables for radionuclides
    nndc_tables = load_nndc_tables(arg.nndctables, config["RADIONUCLIDES"])

    # determines size of dataset based on number of noise and compton scales
    noise_scales = np.linspace(arg.minnoise, arg.maxnoise, arg.numnoise)
    compton_scales = np.linspace(arg.mincompton, arg.maxcompton, arg.numcompton)

    params = [(compton, noise) for compton in compton_scales for noise in noise_scales]

    for dettype in arg.dettype.split(','):
        dettype = dettype.upper()
        print(f'Generating templates for detector {dettype}')
        if arg.savefigs:
            os.makedirs(os.path.join(outdir, dettype), exist_ok=True)
        dataset = generate_spectra(config["DETECTORS"][dettype], nndc_tables, os.path.join(outdir, dettype), arg.savefigs, params)
        save_dataset(dettype, dataset, outfile)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
