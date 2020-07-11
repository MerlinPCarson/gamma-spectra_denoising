import os
import sys
import time
import h5py
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from load_data import load_data
from spectra_utils import compare_results
from model import DnCNN, DnCNN_Res
#from utils import weights_init_kaiming

from sklearn.model_selection import train_test_split
from tqdm import tqdm
#from tensorboardX import SummaryWriter
#from torchvision.utils import make_grid
from skimage.metrics import peak_signal_noise_ratio as psnr

        
def setup_gpus():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_ids = [i for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
    return device_ids

def psnr_of_batch(clean_imgs, denoised_imgs):
    batch_psnr = 0
    for i in range(clean_imgs.shape[0]):
        batch_psnr += psnr(clean_imgs[i,:], denoised_imgs[i,:], data_range=1)
    return batch_psnr/clean_imgs.shape[0]

def main():
    start = time.time()

    parser = argparse. ArgumentParser(description='Gamma-Spectra Denoising Trainer')
    parser.add_argument('--det_type', type=str, default='HPGe', help='detector type to train {HPGe, NaI, CZT}')
    parser.add_argument('--test_set', type=str, default='data/training.h5', help='h5 file with training vectors')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for validation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--model', type=str, default='models/best_model.pt', help='location of model to use')
    parser.add_argument('--outdir', type=str, default='tmp', help='location to save output plots')
    parser.add_argument('--savefigs', help='saves plots of results', default=False, action='store_true')
    args = parser.parse_args()


    # make sure data files exist
    assert os.path.exists(args.test_set), f'Cannot find testset vectors file {args.test_set}'

    # make sure output dirs exists
    os.makedirs(args.outdir, exist_ok=True)

    # detect gpus and setup environment variables
    device_ids = setup_gpus()
    print(f'Cuda devices found: {[torch.cuda.get_device_name(i) for i in device_ids]}')

    print('Loading datasets')
    test_data = load_data(args.test_set)
    noisy_spectra = test_data[args.det_type.upper()]['noisy']
    clean_spectra = test_data[args.det_type.upper()]['clean']
    spectra_keV = test_data[args.det_type.upper()]['keV']

    noisy_spectra = np.expand_dims(noisy_spectra, axis=1)
    clean_spectra = np.expand_dims(clean_spectra, axis=1)

    assert noisy_spectra.shape == clean_spectra.shape, 'Mismatch between shapes of training and target data'

    # applying random seed for reproducability
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # split data into train and validation sets
    _, x_val, _, y_val = train_test_split(noisy_spectra, clean_spectra, test_size = 0.1, random_state=args.seed)

    # get standardization parameters for model
    params = pickle.load(open(args.model.replace('.pt','.npy'),'rb'))['model']
    train_mean = params['train_mean'] 
    train_std = params['train_std'] 

    # load data for training
    val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))

    print(f'Number of validation examples: {len(x_val)}')

    # create batched data loaders for model
    val_loader = DataLoader(dataset=val_dataset, num_workers=os.cpu_count(), batch_size=args.batch_size, shuffle=False)
    print(f'Number of batches {len(val_loader)}')

    # create and load model
    print(f'Loading model {args.model}')
    model = DnCNN(num_channels=params['num_channels'], num_layers=params['num_layers'], \
                  kernel_size=params['kernel_size'], stride=params['stride'], num_filters=params['num_filters']) 
    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(args.model))

    # Main training loop

    print(f'Denoising spectra')
    model.eval() 
    total_psnr_noisy = 0
    total_psnr_denoised = 0
    with torch.no_grad():
        for num, (noisy_spectra, clean_spectra) in enumerate(val_loader, start=1):

            # move batch to GPU
            noisy_spectra = Variable(noisy_spectra.cuda())
            clean_spectra = Variable(clean_spectra.cuda())

            # make predictions
            preds = model((noisy_spectra-train_mean)/train_std)

            # calculate PSNR 
            clean_spectra = clean_spectra.cpu().numpy().astype(np.float32)
            noisy_spectra = noisy_spectra.cpu().numpy().astype(np.float32)
            preds = preds.cpu().numpy().astype(np.float32)
            psnr_noisy = psnr_of_batch(clean_spectra, noisy_spectra)
            psnr_denoised = psnr_of_batch(clean_spectra, preds)
            total_psnr_noisy += psnr_noisy
            total_psnr_denoised += psnr_denoised
            print(f'[{num}/{len(val_loader)}] PSNR {psnr_noisy} --> {psnr_denoised}, increase of {psnr_denoised-psnr_noisy}')
            if args.savefigs:
                compare_results(spectra_keV, clean_spectra[0,0,:], noisy_spectra[0,0,:], preds[0,0,:], args.outdir, str(num))

    avg_psnr_noisy = total_psnr_noisy/len(val_loader)
    avg_psnr_denoised = total_psnr_denoised/len(val_loader)

    print(f'Average PSNR: {avg_psnr_denoised}, average increase of {avg_psnr_denoised-avg_psnr_noisy}')

    print(f'Script completed in {time.time()-start:.2f} secs')
    return 0

if __name__ == '__main__':
    sys.exit(main())

