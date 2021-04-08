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


def load_background(bg_file):

    keV, hits = data_load_normalized(bg_file)

    return hits

def generate_spectra(config, sims_dir, bg_files, params, augment=False, showfigs=False):

    # load all background intensities
    backgrounds = [load_background(bg_file) for bg_file in bg_files] 

    # create datastruct for dataset
    spectra = {"name": [], "spectrum": [], "noisy_spectrum": [], "noise": [], "compton_scale": [], "SNR": []} 

    # for all radionuclide spectra and parameter variations, build spectrum
    for rn in tqdm(config['RADIONUCLIDES']):
        for clean_spectrum in glob(os.path.join(sims_dir, f'{rn}*-nocompton.json')):
            # load peak data time normalized 
            keV, clean_hits = data_load_normalized(clean_spectrum)
            _, compton_hits = data_load_normalized(clean_spectrum.replace('nocompton','compton-only'))

            # find zero keV index for removing noise at and below it
            zero_keV_idx = np.searchsorted(keV, 0.0, side='left')

            for background in backgrounds:

                # remove photoelectric below 0 keV bin
                background[:zero_keV_idx+1] = 0.0

                for compton_scale in params['Compton']:
                    # scale compton
                    scaled_compton_hits = compton_hits * compton_scale

                    for snr in params['SNR']:
                        spectrum, noisy_spectrum, noise = generate_spectrum_SNR(clean_hits, background, scaled_compton_hits, snr)
    
                        add_spectrum(spectra, rn, keV, spectrum, noisy_spectrum, noise, compton_scale, snr, showfigs)

                        if augment:
                            # 50/50 chance of augmenting
                            if np.random.choice([0,1,2]) < 2:
                                # 50/50 chance of shifting direction  
                                if np.random.choice([0,1]) == 1:
                                    # number of channels to shift right
                                    size_of_shift = np.random.randint(1,11)
                                else:
                                    # number of channels to shift left 
                                    size_of_shift = -np.random.randint(1,11)
                                
                                # shift spectrums
                                spectrum = np.roll(spectrum, size_of_shift)
                                noisy_spectrum = np.roll(noisy_spectrum, size_of_shift)
                                noise = np.roll(noise, size_of_shift)
                                add_spectrum(spectra, rn, keV, spectrum, noisy_spectrum, noise, compton_scale, snr, showfigs)

    spectra["keV"] = keV

    return spectra 

def add_spectrum(spectra, rn, keV, spectrum, noisy_spectrum, noise, compton_scale, snr, showfigs=False):
    spectra["name"].append(rn.encode('utf-8'))
    spectra["spectrum"].append(spectrum)
    spectra["noisy_spectrum"].append(noisy_spectrum)
    spectra["noise"].append(noise)
    spectra["compton_scale"].append(compton_scale)
    spectra["SNR"].append(snr)
    if showfigs:
        compare_spectra(keV, [spectrum, noisy_spectrum, noise], 
                             [f'{rn} clean', f'{rn} noisy (SNR={snr})','noise'])

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
    parser.add_argument("-ag", "--augment", help="add dataset augmentations", default=False, action="store_true")
    parser.add_argument("-bg", "--background_dir", help="directory of background spectrum", default="background/NaI")
    parser.add_argument("-maxsnr", "--maxsnr", help="maximum noise SNR", default=50.0, type=float)
    parser.add_argument("-minsnr", "--minsnr", help="minimum noise SNR", default=-25.0, type=float)
    parser.add_argument("-snrstep", "--snrstep", help="SNR step between min and max snr", default=5.0, type=float)
    parser.add_argument("-maxc", "--maxcompton", help="maximum compton scale", default=1.0, type=float)
    parser.add_argument("-minc", "--mincompton", help="minimum compton scale", default=0.0, type=float)
    parser.add_argument("-cstep", "--comptonstep", help="Compton scale step between min and max Compton", default=1.0, type=float)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    return args

def main(args):
    start = time.time()

    # for repeatability
    np.random.seed(args.seed)

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
    dataset = generate_spectra(config, args.mcnp_spectra, backgrounds,  
                               params, augment=args.augment, showfigs=args.showfigs)

    # save dataset to H5 file
    save_dataset(dettype, dataset, outfile)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
