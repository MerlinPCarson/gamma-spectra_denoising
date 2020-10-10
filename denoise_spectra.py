import os
import sys
import time
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
from build_dataset import save_dataset
from spectra_utils import compare_spectra
from model import DnCNN, DnCNN_Res

from tqdm import tqdm

        
def setup_gpus():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_ids = [i for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
    return device_ids

def main():
    start = time.time()

    parser = argparse. ArgumentParser(description='Gamma-Spectra Denoising Trainer')
    parser.add_argument('--dettype', type=str, default='NaI', help='detector type to train {HPGe, NaI, CZT}')
    parser.add_argument('--spectra', type=str, default='spectra/NaI/Uranium', help='directory of spectra or spectrum in json format')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for denoising')
    parser.add_argument('--model', type=str, default='models/best_model.pt', help='location of model to use')
    parser.add_argument('--outdir', type=str, help='location to save output plots')
    parser.add_argument('--outfile', type=str, help='location to save output data', default='denoised_spectra.h5')
    parser.add_argument('--saveresults', help='saves output to .h5 file', default=False, action='store_true')
    parser.add_argument('--savefigs', help='saves plots of results', default=False, action='store_true')
    args = parser.parse_args()

    # if output directory is not provided, save plots to model directory
    if not args.outdir:
        args.outdir = os.path.join(os.path.dirname(args.model), 'results')

    # make sure output dirs exists
    os.makedirs(args.outdir, exist_ok=True)
       
    # make sure data files exist
    assert os.path.exists(args.spectra), f'Cannot find testset spectrum files: {args.spectra}'

    # detect gpus and setup environment variables
    device_ids = setup_gpus()
    print(f'Cuda devices found: {[torch.cuda.get_device_name(i) for i in device_ids]}')

    print('Loading spectra to denoise')
    test_data, test_files = load_spectra(args.spectra)

    spectra = np.expand_dims(np.array(test_data['hits']), axis=1)
    spectra_keV = np.array(test_data['keV'])

    #spectra = np.expand_dims(spectra, axis=1)
    #spectra = np.reshape(spectra, (1,1,-1))
    print(spectra.shape)

    # load parameters for model
    params = pickle.load(open(args.model.replace('.pt','.npy'),'rb'))['model']

    train_mean = params['train_mean'] 
    train_std = params['train_std'] 

    print(f'Number of examples to denoise: {len(spectra)}')

    # create batched data loaders for model
    spectra_loader = DataLoader(dataset=spectra, num_workers=os.cpu_count(), batch_size=args.batch_size, shuffle=False)

    print(f'Number of batches {len(spectra_loader)}')

    # create and load model
    if params['model_name'] == 'DnCNN':
        model = DnCNN(num_channels=params['num_channels'], num_layers=params['num_layers'], \
                      kernel_size=params['kernel_size'], stride=params['stride'], num_filters=params['num_filters']) 
    elif params['model_name'] == 'DnCNN-res':
        model = DnCNN_Res(num_channels=params['num_channels'], num_layers=params['num_layers'], \
                      kernel_size=params['kernel_size'], stride=params['stride'], num_filters=params['num_filters']) 
    else:
        print(f'Model name {params["model_name"]} is not supported.')
        return 1

    # prepare model for data parallelism (use multiple GPUs)
    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

    # loaded saved model
    print(f'Loading weights for {params["model_name"]} model from {args.model} for {params["model_type"]}')
    model.load_state_dict(torch.load(args.model))

    # Main training loop

    print(f'Denoising spectra')
    model.eval() 

    denoised = []
    with torch.no_grad():
        for num, spectra in enumerate(spectra_loader):

            norm_factor = np.sqrt((spectra**2).sum()).float()

            # move batch to GPU
            noisy_spectra = Variable((spectra/norm_factor).cuda())

            # make predictions
            preds = model((noisy_spectra-train_mean)/train_std)

            # save denoised spectrum
            if params['model_type'] == 'Gen-spectrum':
                denoised_spectra = preds
            else:
                denoised_spectra = noisy_spectra-preds 

            denoised_spectra *= norm_factor

            # add batch of denoised spectra to list of denoised spectra
            denoised.extend(denoised_spectra.tolist()) 

            infile = os.path.basename(test_files[num])
            print(f'[{num+1}/{len(spectra_loader)}] Denoising {infile}')
            if args.savefigs:
                outfile = infile.replace('.json','')
                compare_spectra(spectra_keV[num], spectra[0,0,:], denoised_spectra[0,0,:].cpu(), outfile, args.outdir, title1='noisy', title2='denoised') 

    # save denoised data to file, currently only supports entire dataset
    if args.saveresults:
        assert len(spectra) == len(denoised), f'{len(spectra)} examples yet {len(denoised)} denoised' 
        denoised = np.squeeze(np.array(denoised))
        test_data['noisy_spectrum'] = denoised 
        outfile = os.path.join(args.outdir, args.outfile)
        print(f'Saving denoised spectrum to {outfile}')
        save_dataset(args.dettype.upper(), test_data, outfile)

    print(f'Script completed in {time.time()-start:.2f} secs')
    return 0

if __name__ == '__main__':
    sys.exit(main())

