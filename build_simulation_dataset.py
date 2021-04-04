import os
from re import I
import sys
import time
import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

from spectra_utils import data_load_normalized
from spectra_utils import generate_spectrum_SNR
from plot_utils import compare_spectra


def generate_spectra(config, sims_dir, bg_files, outdir, params, showfigs=False):

    # load all background intensities
    backgrounds = [np.array(json.load(open(background))['HIT'], dtype=np.float32) for background in bg_files] 
    # create datastruct for dataset
    spectra = {"name": [], "spectrum": [], "noisy_spectrum": [], "noise": [], "compton_scale": [], "SNR": []} 

    # for all radionuclide spectra and parameter variations, build spectrum
    for rn in tqdm(config['RADIONUCLIDES']):
        for clean_spectrum in glob(os.path.join(sims_dir, f'{rn}*-nocompton.json')):
            # load peak data time normalized 
            keV, clean_hits = data_load_normalized(clean_spectrum)
            _, compton_hits = data_load_normalized(clean_spectrum.replace('nocompton','compton-only'))

            for background in backgrounds:
                for compton_scale in params['Compton']:
                    # scale compton
                    scaled_compton_hits = compton_hits * compton_scale

                    for snr in params['SNR']:
                        #print(f'{spectrum}:{compton_scale}:{snr}dB')
                        spectrum, noisy_spectrum, noise = generate_spectrum_SNR(clean_hits, background, scaled_compton_hits, snr)
                        spectra["name"].append(rn.encode('utf-8'))
                        spectra["spectrum"].append(spectrum)
                        spectra["noisy_spectrum"].append(noisy_spectrum)
                        spectra["noise"].append(noise)
                        spectra["compton_scale"].append(compton_scale)
                        spectra["SNR"].append(snr)
                        if showfigs:
                            compare_spectra(keV, [spectrum, noisy_spectrum, noise], 
                                                 [f'{rn} clean', f'{rn} noisy (SNR={snr})','noise'])

    spectra["keV"] = keV

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

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--configfile", help="configuration file for generating data", default="config_data_real.json")
    parser.add_argument("-out", "--outfile", help="output file for data", default="data/training.h5")
    parser.add_argument("-det", "--dettype", help="detector type", default="NaI")
    parser.add_argument("-mcnp", "--mcnp_spectra", help="location of mcnp_simulated spectra",  default="data/mcnp_spectra/preproc_spectra")
    parser.add_argument("-sf", "--showfigs", help="show figures when building dataset", default=False, action="store_true")
    parser.add_argument("-bg", "--background_dir", help="directory of background spectrum", default="background/NaI/Uranium")
    parser.add_argument("-maxsnr", "--maxsnr", help="maximum noise SNR", default=50.0, type=float)
    parser.add_argument("-minsnr", "--minsnr", help="minimum noise SNR", default=-25.0, type=float)
    parser.add_argument("-snrstep", "--snrstep", help="SNR step between min and max snr", default=5.0, type=float)
    parser.add_argument("-maxc", "--maxcompton", help="maximum compton scale", default=1.0, type=float)
    parser.add_argument("-minc", "--mincompton", help="minimum compton scale", default=0.0, type=float)
    parser.add_argument("-cstep", "--comptonstep", help="Compton scale step between min and max Compton", default=1.0, type=float)
    args = parser.parse_args()

    return args

def main(args):
    start = time.time()

    # setup vars
    dettype = args.dettype.upper()
    outdir = os.path.dirname(args.outfile)
    outfile = args.outfile

    # load configuration parameters
    with open(args.configfile, 'r') as cfile:
        config = json.load(cfile)

    # make output dir if it does not exist
    os.makedirs(outdir, exist_ok=True)

    # determines size of dataset based on number of noise and compton scales
    snrs = np.arange(args.minsnr, args.maxsnr+args.snrstep, args.snrstep)
    compton_scales = np.arange(args.mincompton, args.maxcompton+args.comptonstep, args.comptonstep)

    # dataset variation parameters for each radionuclide spectra
    params = {'SNR': snrs, 'Compton': compton_scales}

    # find all background files
    backgrounds = glob(os.path.join(args.background_dir, '*.json'))

    # Generate the dataset with all radionuclides in config file at all Compton/SNRs
    print(f'Generating dataset for {dettype} detector')
    dataset = generate_spectra(config, args.mcnp_spectra, backgrounds, outdir, 
                               params, showfigs=args.showfigs)

    # save dataset to H5 file
    save_dataset(dettype, dataset, outfile)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
