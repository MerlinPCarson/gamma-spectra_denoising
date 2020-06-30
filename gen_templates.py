import os
import sys
import time
import json
import argparse
import numpy as np

import matplotlib.pyplot as plt


def generate_spectrum(rn, calibration, num_channels):
    pass


def generate_templates(radionuclides, config):

    for rn in radionuclides:
        generate_spectrum(rn, config['ENER_FIT'], config['NUM_CHANNELS'])

def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--rnlistfile", help="file containing list of radionuclides to use", default="ANSI_N42.34.json")
    parser.add_argument("-cf", "--configfile", help="configuration file", default="config.json")
    parser.add_argument("-out", "--outdir", help="output dir for data", default="data")
    #parser.add_argument("-det", "--dettype", help="detector type", default="GERMANIUM,NAI,CZT")
    parser.add_argument("-det", "--dettype", help="detector type", default="GERMANIUM")
    parser.add_argument("-nndc", "--nndctables", help="location of NNDC tables data",  default="nuclides-nndc")
    arg = parser.parse_args()

    # load list of radionuclides to build
    with open(arg.rnlistfile, 'r') as rfile:
        radionuclides = json.load(rfile)['radionuclides']

    # load configuration parameters
    with open(arg.configfile, 'r') as cfile:
        config = json.load(cfile)

    # make output dir if it does not exist
    os.makedirs(arg.outdir, exist_ok=True)

    for dettype in arg.dettype.split(','):
        print(f'Generating templates for detector {dettype}')
        os.makedirs(os.path.join(arg.outdir, dettype), exist_ok=True)
        generate_templates(radionuclides, config["DETECTORS"][dettype])

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
