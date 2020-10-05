import os
import sys
import time
import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm

from spectra_utils import load_nndc_tables, load_radionuclide_nndc, generate_spectrum_SNR, compare_spectra
#from spectra_utils import load_radionuclide_nndc, generate_spectrum, compare_spectra


def generate_spectra(config, nndc_tables, params, outdir):

    background = json.load(open("background/NaI/BG1200s-U.json"))
    bg_intensities = np.array(background["HIT"], dtype=np.float32)
    spectra = {"name": [], "spectrum": [], "noisy_spectrum": [], "noise": [], "compton_scale": [], "SNR": []} 
    for rn_name, rn_values in tqdm(nndc_tables.items()):
        for compton_scale, snr in params:
            spectrum_keV, spectrum, noisy_spectrum, noise = generate_spectrum_SNR(rn_values, config, bg_intensities, compton_scale=compton_scale, snr=snr)
            spectra["name"].append(rn_name.encode('utf-8'))
            spectra["spectrum"].append(spectrum)
            spectra["noisy_spectrum"].append(noisy_spectrum)
            spectra["noise"].append(noise)
            spectra["compton_scale"].append(compton_scale)
            spectra["SNR"].append(snr)

            if True:
                compare_spectra(spectrum_keV, spectrum, noisy_spectrum, noise, rn_name, outdir, title1='template', title2='noisy', show_plot=True)

    spectra["keV"] = spectrum_keV

    return spectra 
        
def save_dataset(dettype, dataset, outdir, outfile='training.h5'):
    with h5py.File(os.path.join(outdir, outfile), 'a') as h5f:
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
    parser.add_argument("-cf", "--configfile", help="configuration file for generating data", default="config_data_real.json")
    parser.add_argument("-out", "--outdir", help="output directory for data", default="data")
    #parser.add_argument("-det", "--dettype", help="detector type", default="HPGe,NaI,CZT")
    parser.add_argument("-det", "--dettype", help="detector type", default="NaI")
    parser.add_argument("-bg", "--background", help="background spectrum", default="background/NaI/BG1200s-U.json")
    parser.add_argument("-nndc", "--nndctables", help="location of NNDC tables data",  default="nuclides-nndc")
    parser.add_argument("-sf", "--savefigs", help="saves plots of templates", action="store_true")
    parser.add_argument("-maxsnr", "--maxsnr", help="maximum noise SNR", default=5.0, type=float)
    parser.add_argument("-minsnr", "--minsnr", help="minimum noise SNR", default=-5.0, type=float)
    parser.add_argument("-snrstep", "--snrstep", help="SNR step between min and max snr", default=5, type=int)
    parser.add_argument("-maxc", "--maxcompton", help="maximum Compton scale", default=0.5, type=float)
    parser.add_argument("-minc", "--mincompton", help="minimum Compton scale", default=0.0, type=float)
    parser.add_argument("-cstep", "--comptonstep", help="Compton scale step between min and max Compton", default=.5, type=int)
    arg = parser.parse_args()

    dettype = arg.dettype
    outdir = os.path.join(arg.outdir, dettype) 

    # load configuration parameters
    with open(arg.configfile, 'r') as cfile:
        config = json.load(cfile)

    # make output dir if it does not exist
    os.makedirs(outdir, exist_ok=True)

    # load NNDC tables for radionuclides
    nndc_tables = load_nndc_tables(arg.nndctables, config["RADIONUCLIDES"])

    # determines size of dataset based on number of noise and compton scales
    snrs = np.arange(arg.minsnr, arg.maxsnr+arg.snrstep, arg.snrstep)
    compton_scales = np.arange(arg.mincompton, arg.maxcompton+arg.comptonstep, arg.comptonstep)

    params = [(compton, snr) for compton in compton_scales for snr in snrs]

    # Generate the dataset with all radionuclides in config file at all Compton/SNRs
    dataset = generate_spectra(config["DETECTORS"][dettype.upper()], nndc_tables, params, outdir)

    # save dataset to HDF5 files
    save_dataset(dettype, dataset, outdir)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
