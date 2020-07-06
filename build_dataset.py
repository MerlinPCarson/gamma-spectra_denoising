import os
import sys
import time
import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm

from spectra_utils import load_radionuclide_nndc, generate_spectrum, plot_spectra, plot_noisy_spectra


def load_nndc_tables(nndc_dir, radionuclides):

    nndc_tables = {}

    for rn in radionuclides:
        keV, intensity = load_radionuclide_nndc(nndc_dir, rn)
        nndc_tables[rn] = {"keV": keV, "intensity": intensity}
        
    return nndc_tables
    

def generate_spectra(config, nndc_tables, outdir, savefigs, compton_scale=0.5):

    spectra = {"spectrum": [], "noisy_spectrum": []} 
    for rn_name, rn_values in tqdm(nndc_tables.items()):
        #print(f"building template for {rn_name}")
        spectrum_keV, spectrum, noisy_spectrum = generate_spectrum(rn_values, config, compton_scale=compton_scale)
        spectra["spectrum"].append(spectrum)
        spectra["noisy_spectrum"].append(noisy_spectrum)
        if savefigs:
            plot_noisy_spectra(spectrum_keV, spectrum, noisy_spectrum, rn_name, outdir, show_plot=False)

    spectra["keV"] = spectrum_keV

    return spectra 
        
def save_dataset(dettype, dataset, outfile):
    with h5py.File(outfile, 'w') as h5f:
        h5f.create_group(dettype)
        for k, v in dataset.items():
            h5f[dettype].create_dataset(k, data=v)
        

 #           for k2, v2 in v.items():
 #               try:
 #                   h5f[dettype][k].create_dataset(k2, data=v2)
 #               except: # overwrites existing data if data already exists
 #                   data = h5f[dettype][k][k2] 
 #                   data[...]= v2 


def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--rnlistfile", help="file containing list of radionuclides to use", default="ANSI_N42.34.json")
    parser.add_argument("-cf", "--configfile", help="configuration file for generating data", default="config_data.json")
    parser.add_argument("-out", "--outfile", help="output file for data", default="data/training.h5")
    #parser.add_argument("-det", "--dettype", help="detector type", default="HPGe,NaI,CZT")
    parser.add_argument("-det", "--dettype", help="detector type", default="HPGe")
    parser.add_argument("-nndc", "--nndctables", help="location of NNDC tables data",  default="nuclides-nndc")
    parser.add_argument("-sf", "--savefigs", help="saves plots of templates", action="store_true")
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

    for dettype in arg.dettype.split(','):
        dettype = dettype.upper()
        print(f'Generating templates for detector {dettype}')
        if arg.savefigs:
            os.makedirs(os.path.join(outdir, dettype), exist_ok=True)
        dataset = generate_spectra(config["DETECTORS"][dettype], nndc_tables, os.path.join(outdir, dettype), arg.savefigs)
        save_dataset(dettype, dataset, outfile)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
