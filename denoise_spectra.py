import os
import sys
import time
import json
import pickle
import argparse
import numpy as np

import matplotlib
if not os.environ.get('DISPLAY', '').strip():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

from load_data_real import load_spectra
from PaperArtifacts import compare_spectra
from model import DnCNN, DnCNN_Res
from train_real import setup_device


def save_spectra(test_data, outdir):

    for hits, f_name in zip(test_data['denoised_spectrum'], test_data['spec_name']):
        json_data = json.load(open(f_name, 'r'))
        json_data['HIT'] = hits.tolist()
        outfile = os.path.join(outdir, os.path.basename(f_name.replace('.json','-denoised.json')))
        print(f'saving denoised spectra to {outfile}')
        json.dump(json_data, open(outfile, 'w'), indent=4)

def parse_args():
    parser = argparse. ArgumentParser(description='Gamma-Spectra Denoising')
    parser.add_argument('--dettype', type=str, default='NaI', help='detector type to train {HPGe, NaI, CZT}')
    parser.add_argument('--spectra', type=str, default='spectra/NaI/Uranium', help='directory of spectra or spectrum in json format')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for denoising')
    parser.add_argument('--model', type=str, default='models/best_model.pt', help='location of model to use')
    parser.add_argument('--outdir', type=str, help='location to save output plots')
    parser.add_argument('--outfile', type=str, help='location to save output data', default='denoised_spectra.h5')
    parser.add_argument('--saveresults', help='saves output to .h5 and json files', default=True, action='store_true')
    parser.add_argument('--savefigs', help='saves plots of each denoised spectra', default=True, action='store_true')
    parser.add_argument('--showfigs', help='shows plots of each denoised spectra', default=False, action='store_true')
    parser.add_argument('--min_keV', type=float, default=0.0, help='minimum keV to plot')
    parser.add_argument('--max_keV', type=float, default=1500.0, help='maximum keV to plot')
    args = parser.parse_args()

    return args


def main(args):
    start = time.time()

    # detect gpus/cpu and setup environment variables
    setup_device(args)

    # if output directory is not provided, save plots to model directory
    if not args.outdir:
        args.outdir = os.path.join(os.path.dirname(args.model), 'results')

    # make sure output dirs exists
    os.makedirs(args.outdir, exist_ok=True)
       
    # make sure data files exist
    assert os.path.exists(args.spectra), f'Cannot find testset spectrum files: {args.spectra}'

    print('Loading spectra to denoise')
    test_data, test_files = load_spectra(args.spectra)

    spectra = np.expand_dims(np.array(test_data['hits'], dtype=np.float32), axis=1)
    spectra_keV = np.array(test_data['keV'], dtype=np.float32)
    spectra_name = [os.path.basename(test_data['spec_name'][i]).replace('.json', '') for i in range(len(test_data['spec_name']))]

    print(spectra.shape)

    # load parameters for model
    params = pickle.load(open(args.model.replace('.pt','.npy'),'rb'))['model']

    # load training set statistics
    train_mean = params['train_mean'] 
    train_std = params['train_std'] 

    print(f'Number of examples to denoise: {len(spectra)}')

    # create batched data loaders for model
    spectra_loader = DataLoader(dataset=spectra, num_workers=os.cpu_count(), batch_size=args.batch_size, shuffle=False)

    print(f'Number of batches {len(spectra_loader)}')

    # create and load model
    if params['model_name'] == 'DnCNN':
        model = DnCNN(num_channels=params['num_channels'], num_layers=params['num_layers'],
                      kernel_size=params['kernel_size'], num_filters=params['num_filters'],
                      dilation_rate=params['dilation_rate']).to(args.device)
    elif params['model_name'] == 'DnCNN-res':
        model = DnCNN_Res(num_channels=params['num_channels'], num_layers=params['num_layers'],
                      kernel_size=params['kernel_size'], num_filters=params['num_filters'],
                      dilation_rate=params['dilation_rate']).to(args.device)
    else:
        print(f'Model name {params["model_name"]} is not supported.')
        return 1

    # loaded saved model
    print(f'Loading weights for {params["model_name"]} model from {args.model} for {params["model_type"]}')
    model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

    # Start denoising 
    print(f'Denoising spectra')
    model.eval() 

    denoised = []
    with torch.no_grad():
        for num, spectra in enumerate(spectra_loader):

            norm_factor = np.sqrt((spectra**2).sum()).float()

            # move batch to GPU or CPU
            noisy_spectra = Variable((spectra/norm_factor).to(args.device))

            # make predictions
            preds = model((noisy_spectra-train_mean)/train_std)
            preds = preds.cpu().numpy().astype(np.float32)

            # save denoised spectrum
            if params['model_type'] == 'Gen-spectrum':
                denoised_spectrum = preds
            else:
                denoised_spectrum = noisy_spectra-preds 

            denoised_spectrum = np.clip(denoised_spectrum, 0.0, None)
            denoised_spectrum *= norm_factor.numpy()

            # add batch of denoised spectra to list of denoised spectra
            denoised.extend(denoised_spectrum.tolist()) 

            infile = os.path.basename(test_files[num])
            print(f'[{num+1}/{len(spectra_loader)}] Denoising {infile}')
            if args.savefigs:
                outfile = os.path.join(args.outdir, infile.replace('.json','.pdf'))
                compare_spectra(spectra_keV[num], [spectra[0,0,:], denoised_spectrum[0,0,:]], 
                                [f'{spectra_name[num]} Spectrum', 'GS-DnCNN Denoised'], args.min_keV, args.max_keV,
                                outfile=outfile, savefigs=args.savefigs, showfigs=args.showfigs,
                                ylabel='Counts')
                outfile = os.path.join(args.outdir, infile.replace('.json','.png'))
                compare_spectra(spectra_keV[num], [spectra[0,0,:], denoised_spectrum[0,0,:]], 
                                [f'{spectra_name[num]} Spectrum', 'GS-DnCNN Denoised'], args.min_keV, args.max_keV,
                                outfile=outfile, savefigs=args.savefigs, showfigs=args.showfigs,
                                ylabel='Counts')

    # save denoised data to file, currently only supports entire dataset
    if args.saveresults:
        assert len(spectra_loader) == len(denoised), f'{len(spectra)} examples yet {len(denoised)} denoised' 
        denoised = np.squeeze(np.array(denoised))
        test_data['denoised_spectrum'] = denoised 
        outfile = os.path.join(args.outdir, args.outfile)
        print(f'Saving denoised spectrum to {outfile}')
        #save_dataset(args.dettype.upper(), test_data, outfile)
        save_spectra(test_data, args.outdir)

    print(f'Script completed in {time.time()-start:.2f} secs')
    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))

