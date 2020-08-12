import os
import sys
import time
import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm

from spectra_utils import load_radionuclide_nndc, generate_spectrum, plot_spectrum


def load_nndc_tables(nndc_dir, radionuclides):

    nndc_tables = {}

    for rn in radionuclides:
        keV, intensity = load_radionuclide_nndc(nndc_dir, rn)
        nndc_tables[rn] = {"keV": keV, "intensity": intensity}
        
    return nndc_tables
    

def generate_templates(config, nndc_tables, outdir, savefigs):

    templates = {}
    for rn_name, rn_values in tqdm(nndc_tables.items()):
        #print(f"building template for {rn_name}")
        keV, intensity, _, _ = generate_spectrum(rn_values, config)
        templates[rn_name] = {"keV": keV, "intensity": intensity}
        if savefigs:
            plot_spectrum(keV, intensity, rn_name, outdir)

    return templates
        
def save_templates(dettype, templates, outfile):
    with h5py.File(outfile, 'a') as h5f:
        try:
            h5f.create_group(dettype)
        except: # does not create detector group if it already exists
            pass
        for k, v in templates.items():
            try:
                h5f[dettype].create_group(k)
            except: # does not create radionuclide group if it already exists
                pass
            for k2, v2 in v.items():
                try:
                    h5f[dettype][k].create_dataset(k2, data=v2)
                except: # overwrites existing data if data already exists
                    data = h5f[dettype][k][k2] 
                    data[...]= v2 


def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--rnlistfile", help="file containing list of radionuclides to use", default="ANSI_N42.34.json")
    parser.add_argument("-cf", "--configfile", help="configuration file for generating data", default="config_data.json")
    parser.add_argument("-out", "--outfile", help="output file for data", default="data/templates.h5")
    parser.add_argument("-det", "--dettype", help="detector type", default="HPGe,NaI,CZT")
    #parser.add_argument("-det", "--dettype", help="detector type", default="HPGe")
    parser.add_argument("-nndc", "--nndctables", help="location of NNDC tables data",  default="nuclides-nndc")
    parser.add_argument("-sf", "--savefigs", help="saves plots of templates", action="store_true")
    #parser.add_argument("-n", "--normalize", help="normalize templates by RMS", action="store_true")
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
        templates = generate_templates(config["DETECTORS"][dettype], nndc_tables, os.path.join(outdir, dettype), arg.savefigs)
        save_templates(dettype, templates, outfile)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
