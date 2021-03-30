import os
import sys
import time
import json
import argparse
import numpy as np

from glob import glob
from tqdm import tqdm
from spectra_utils import data_load_normalized
from spectra_utils import data_smooth
from spectra_utils import split_radionuclide_name


def broaden_lines(keV, hits, config):
    # widen peaks to match detector resolution 
    broad_hits = data_smooth(keV, hits, **config['SMOOTH_PARAMS'])

    return broad_hits

def save_spec(keV, hits, postfix_title, spec_file, outdir):
    # load raw spectrum
    with open(spec_file, 'r') as spectrum:
        spec = json.load(spectrum)

    # overwrite new keV and hits values
    spec['KEV'] = keV
    spec['HIT'] = hits

    # write new spectrum
    new_spec_file = os.path.join(outdir, os.path.basename(spec_file).replace('.json', f'{postfix_title}.json'))
    with open(new_spec_file, 'w') as new_spectrum:
        json.dump(spec, new_spectrum)

def split_pe_compton(hits, keV, rn_pe):
    # new array for photoelectric only
    hits_pe = np.zeros_like(hits)
    hits_compton = np.zeros_like(hits)

    # find all photoelectric for radionuclide 
    pe_idxs = [np.searchsorted(keV, pe, side='left') for pe in rn_pe]
    hits_pe[pe_idxs] = np.array(hits)[pe_idxs]

    # remove photoelectric from spectrum, leaving just compton
    hits_compton = hits 
    hits_compton[pe_idxs] = 0 

    return hits_compton, hits_pe

def preproc_spectrum(spec_file, det_config, outdir, pe_keV, compton_dir):

    # load peak data time normalized 
    keV, hits = data_load_normalized(spec_file)

    # remove photoelectric that passed through detector, presents in 0 keV bin
    zero_keV_idx = np.searchsorted(keV, 0.0, side='left')
    hits[:zero_keV_idx+1] = 0.0

    # broaden peaks
    broad_hits = broaden_lines(keV, hits, det_config)
    # save broadened peaks with compton
    save_spec(keV.tolist(), broad_hits.tolist(), '-compton', spec_file, outdir)

    # seperate compton and photoelectric
    compton_hits, no_compton_hits = split_pe_compton(hits, keV, pe_keV)
    # save simulations without compton
    save_spec(keV.tolist(), no_compton_hits.tolist(), '-no-compton', spec_file, compton_dir)
    # save simulations with only compton
    save_spec(keV.tolist(), compton_hits.tolist(), '-compton-only', spec_file, compton_dir)

    # broaden peaks without compton
    broad_hits = broaden_lines(keV, no_compton_hits, det_config)
    # save broadened peaks with without compton
    save_spec(keV.tolist(), broad_hits.tolist(), '-no-compton', spec_file, outdir)

    # broaden peaks without compton
    broad_hits = broaden_lines(keV, compton_hits, det_config)
    # save broadened peaks with without compton
    save_spec(keV.tolist(), broad_hits.tolist(), '-compton-only', spec_file, outdir)

def get_rn_name(spec_file):
    basename = os.path.basename(spec_file)
    rn_name = basename.split('_')[0]
    return split_radionuclide_name(rn_name)

def parse_args():

    parser = argparse.ArgumentParser(description='Preprocess MCNP spectrum simulations')
    parser.add_argument('--spectra_dir', type=str, default='data/mcnp_spectra/spectra', help='location of MCNP simulations')
    parser.add_argument("-out", "--outdir", help="output directory for preprocessed spectra", default="data/mcnp_spectra/preproc_spectra")
    parser.add_argument("-cf", "--configfile", help="configuration file for generating data", default="config_data_real.json")
    parser.add_argument("-nndc", "--nndctables", help="location of NNDC tables data",  default="nuclides-nndc")
    parser.add_argument("-det", "--dettype", help="detector type", default="NaI")
    parser.add_argument('--r', type=str, default='data/mcnp_spectra/spectra', help='location of MCNP simulations')
    args = parser.parse_args()

    # make sure config filre exists
    assert os.path.isfile(args.configfile), f'Configuration file {args.configfile} not found!'

    # make sure spectra to preprocess directory exists
    assert os.path.isdir(args.spectra_dir), f'Directory containing MCNP simulated spectra {args.spectra_dir} not found!'

    # make detector type uppercase so no case issues
    args.dettype = args.dettype.upper()

    return args

def load_rn_pe_keVs(nndctables, rn_num, rn_letters):
        pe_keV = json.load(open(os.path.join(nndctables, rn_num, f'{rn_letters}.json'),'r'))['keV']

        # MCNP uses single decimal above 1000 keV, so drop them 
        pe_keV = [pe if pe < 1000 else float(f'{pe:.1f}') for pe in pe_keV]

        return pe_keV

def main(args):
    start = time.time()

    # load configuration parameters
    with open(args.configfile, 'r') as cfile:
        config = json.load(cfile)

    # make output directory if it does not exist
    os.makedirs(args.outdir, exist_ok=True)
    
    # make directory to save seperated compton and photoelectric versions of MCNP sims
    raw_compton_dir = os.path.join(args.spectra_dir,'compton')
    os.makedirs(raw_compton_dir, exist_ok=True)

    print(f'Preprocesing MCNP simulations in {args.spectra_dir}')
    for spec_file in tqdm(glob(os.path.join(args.spectra_dir, '*.json'))):
        rn_num, rn_letters = get_rn_name(spec_file)
        pe_keV = load_rn_pe_keVs(args.nndctables, rn_num, rn_letters)
        preproc_spectrum(spec_file, config['DETECTORS'][args.dettype], args.outdir, pe_keV, raw_compton_dir)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
