import os
import sys
import time
import json
import argparse
import numpy as np
from tqdm import tqdm

from spectra_utils import load_radionuclide_nndc, generate_spectrum


def load_nndc_tables(nndc_dir, radionuclides):

    nndc_tables = {}

    for rn in radionuclides:
        keV, intensity = load_radionuclide_nndc(nndc_dir, rn)
        nndc_tables[rn] = {"keV": keV, "intensity": intensity}
        
    return nndc_tables

    
    

def generate_templates(config, nndc_tables):

    templates = {}
    for rn_name, rn_values in tqdm(nndc_tables.items()):
        #print(f"building template for {rn_name}")
        keV, intensity = generate_spectrum(rn_values, config)
        templates[rn_name] = {"keV": keV, "intensity": intensity}

    return templates
        

def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--rnlistfile", help="file containing list of radionuclides to use", default="ANSI_N42.34.json")
    parser.add_argument("-cf", "--configfile", help="configuration file for generating data", default="config_data.json")
    parser.add_argument("-out", "--outdir", help="output dir for data", default="data")
    #parser.add_argument("-det", "--dettype", help="detector type", default="GERMANIUM,NAI,CZT")
    parser.add_argument("-det", "--dettype", help="detector type", default="GERMANIUM")
    parser.add_argument("-nndc", "--nndctables", help="location of NNDC tables data",  default="nuclides-nndc")
    arg = parser.parse_args()

    # load configuration parameters
    with open(arg.configfile, 'r') as cfile:
        config = json.load(cfile)

    # make output dir if it does not exist
    os.makedirs(arg.outdir, exist_ok=True)

    # load NNDC tables for radionuclides
    nndc_tables = load_nndc_tables(arg.nndctables, config["RADIONUCLIDES"])

    for dettype in arg.dettype.split(','):
        print(f'Generating templates for detector {dettype}')
        os.makedirs(os.path.join(arg.outdir, dettype), exist_ok=True)
        templates = generate_templates(config["DETECTORS"][dettype], nndc_tables)
        #save_template(dettype, templates)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
