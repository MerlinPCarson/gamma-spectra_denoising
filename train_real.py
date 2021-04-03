import os
import sys
import time
import random
import pickle
import argparse
import numpy as np

import torch
# makes cuDNN deterministic
torch.backends.cudnn.benchmark = False
try:    # PyTorch 1.8
    torch.use_deterministic_algorithms(True)
except: # PyTorch 1.7
    torch.set_deterministic(True)

from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from load_data_real import load_data
from model import DnCNN, DnCNN_Res

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

def setup_device(args):
    # detect gpus and setup environment variables
    if torch.cuda.is_available() is True:
        args.device = 'cuda:0'
        args.device_ids = setup_gpus()
        print(f'Cuda devices found: {[torch.cuda.get_device_name(i) for i in args.device_ids]}')
    else:
        args.device = 'cpu'        
        args.device_ids = []

def setup_gpus():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_ids = [i for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
    return device_ids

# function to remove weight decay from output layer and batchnorm layers
def weight_decay(model):
    
    batchnorm_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            batchnorm_layers.append(name.split('.')[-1])

    params = []
    for name, param in model.named_parameters():
        # set all batchnorm layers L2 regularization to 0
        if name.split('.')[-2] in batchnorm_layers:
            #print(f'setting weight decay to 0 for layer {name}')
            params.append({'params': param, 'weight_decay': 0.0})
        else:
            params.append({'params': param})

    # set final layer L2 regularization to 0
    params[-1]['weight_decay'] = 0.0 

    return params

def l1_cost(model):
    cost = 0.0
    for param in model.parameters():
        cost += torch.sum(param.abs())
    return cost

def psnr_of_batch(clean_imgs, denoised_imgs):
    batch_psnr = 0
    for i in range(clean_imgs.shape[0]):
        batch_psnr += psnr(clean_imgs[i,:], denoised_imgs[i,:], data_range=1)
    return batch_psnr/clean_imgs.shape[0]

def create_data_loaders(args):

    print(f'Loading dataset {args.det_type}')
    training_data = load_data(args.train_set, args.det_type.upper())
    noisy_spectra = training_data['noisy_spectrum']
    target_spectra = training_data['spectrum']

    assert noisy_spectra.shape == target_spectra.shape, 'Mismatch between shapes of training and target data'
    noisy_spectra = np.expand_dims(noisy_spectra, axis=1)

    # if target is noise
    if args.gennoise:
        noise = training_data['noise']
        #assert noisy_spectra.shape == noise.shape, 'Mismatch between shapes of training and target data'
        # add noise to target data since noise will be the target not the clean spectra, still need clean data for PSNR
        target_spectra = np.stack((target_spectra,noise), axis=1)
    else:
        target_spectra = np.expand_dims(target_spectra, axis=1)

    # split data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(noisy_spectra, target_spectra, test_size = 0.1, random_state=args.seed)

    print(f'Training data: {x_train.shape}, validation data: {x_val.shape}')
    print(f'Training target: {y_train.shape}, validation target: {y_val.shape}')

    # get standardization parameters from training set
    args.train_mean = np.mean(x_train)
    args.train_std = np.std(x_train)

    # apply standardization parameters to training and validation sets
    x_train = (x_train-args.train_mean)/args.train_std
    #x_val = (x_val-train_mean)/train_std

    # input shape for each example to network, NOTE: channels first
    args.num_channels, args.num_features = 1, x_train.shape[2]
    print(f'Input shape to model forward will be: ({args.batch_size}, {args.num_channels}, {args.num_features})')

    # load data for training
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))

    print(f'Number of training examples: {len(x_train)}')
    print(f'Number of validation examples: {len(x_val)}')

    # create batched data loaders for model
    train_loader = DataLoader(dataset=train_dataset, num_workers=os.cpu_count(), 
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=os.cpu_count(), 
                              batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader

def build_model(args):
    # create model
    if not args.res:
        model = DnCNN(num_channels=args.num_channels, num_layers=args.num_layers, 
                      kernel_size=args.filter_size, stride=args.stride, num_filters=args.num_filters).to(args.device) 
    else:
        model = DnCNN_Res(num_channels=args.num_channels, num_layers=args.num_layers, 
                      kernel_size=args.filter_size, stride=args.stride, num_filters=args.num_filters).to(args.device)
    
    # if more than 1 GPU, prepare model for data parallelism (use multiple GPUs)
    if len(args.device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids).cuda()

    # setup loss and optimizer
    criterion = torch.nn.MSELoss(reduction='sum').to(args.device)
    params = model.parameters()
    #params = weight_decay(model)
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(.9, .999), eps=1e-8, weight_decay=args.l2, amsgrad=False)

    return model, criterion, optimizer

def train_model(model, criterion, optimizer, train_loader, val_loader, args):
    # make sure output dirs exists
    os.makedirs(args.model_dir, exist_ok=True)

    # data struct to track training and validation losses per epoch
    model_params = {'model_name': 'DnCNN-res' if args.res else 'DnCNN', 
                    'model_type': 'Gen-noise' if args.gennoise else 'Gen-spectrum', 
                    'train_seed': args.seed, 'num_channels':args.num_channels, 
                    'num_layers':args.num_layers, 'kernel_size':args.filter_size,
                    'stride':args.stride, 'num_filters':args.num_filters, 
                    'train_mean': args.train_mean, 'train_std': args.train_std}

    # save model parameters
    history = {'model': model_params, 'train':[], 'val':[], 'psnr':[]}
    pickle.dump(history, open(os.path.join(args.model_dir, 'model.npy'), 'wb'))

    # schedulers
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience//2)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    # intializiang best values for regularization via early stopping 
    best_val_loss = 99999
    best_psnr = 0
    best_epoch = 0
    epochs_since_improvement = 0

    # Main training loop
    for epoch in range(1, args.epochs+1):
        print(f'Starting epoch {epoch}/{args.epochs} with learning rate {optimizer.param_groups[0]["lr"]}')

        model.train()
        epoch_train_loss = 0

        # iterate through batches of training examples
        for noisy_spectra, target_data in tqdm(train_loader):
            model.zero_grad()

            # move batch to device (GPU or CPU) 
            noisy_spectra = Variable(noisy_spectra.to(args.device))
            if not args.gennoise:
                target = Variable(target_data[:,0:1,:].to(args.device))
            else:
                target = Variable(target_data[:,1:,:].cuda())

            # make predictions
            preds = model(noisy_spectra)

            # calculate loss
            loss = criterion(preds, target)/(2*len(noisy_spectra)) + args.l1 * l1_cost(model)
            epoch_train_loss += loss.item()

            # backprop
            loss.backward()
            optimizer.step()

        # start evaluation
        print(f'Validating Model')
        model.eval() 
        epoch_val_loss = 0
        epoch_psnr = 0
        with torch.no_grad():
            for noisy_spectra, target_data in tqdm(val_loader):

                # standardize during training so we can original data for PSNR
                x_val = (noisy_spectra-args.train_mean)/args.train_std
                
                # move batch to GPU
                x_val = Variable(x_val.to(args.device))

                if not args.gennoise:
                    target = Variable(target_data[:,0:1,:].to(args.device))
                else:
                    target = Variable(target_data[:,1:,:].to(args.device))

                # make predictions
                preds = model(x_val)

                # calculate loss
                val_loss = criterion(preds, target)/(2*len(noisy_spectra)) + args.l1 * l1_cost(model)
                epoch_val_loss += val_loss.item()

                # calculate PSNR 
                if not args.gennoise:
                    epoch_psnr += psnr_of_batch(target.cpu().numpy().astype(np.float32), 
                                                np.clip(preds.cpu().numpy().astype(np.float32),0,None))
                else:
                    denoised = noisy_spectra - preds.cpu()    # subtract predicted noise from noisy spectra
                    epoch_psnr += psnr_of_batch(target_data[:,0:1,:].numpy().astype(np.float32), 
                                                denoised.numpy().astype(np.float32))

        # epoch summary
        epoch_train_loss /= len(train_loader) 
        epoch_val_loss /= len(val_loader) 
        epoch_psnr /= len(val_loader)

        # reduce learning rate if validation has leveled off
        scheduler.step(epoch_val_loss)
        #scheduler.step(epoch_psnr)

        # exponential decay of learning rate
        #scheduler.step()

        # save epoch stats
        history['train'].append(epoch_train_loss)
        history['val'].append(epoch_val_loss)
        history['psnr'].append(epoch_psnr)
        print(f'Training loss: {epoch_train_loss}')
        print(f'Validation loss: {epoch_val_loss}')
        print(f'Validation PSNR: {epoch_psnr}')

        # save if best model
        if epoch_val_loss < best_val_loss:
        #if epoch_psnr > best_psnr:
            print('Saving best model')
            best_val_loss = epoch_val_loss
            best_psnr = epoch_psnr
            best_epoch = epoch
            epochs_since_improvement = 0

            # save model, multi-gpu must save module to be compatible for loading on other devices
            if len(args.device_ids) > 1:
                torch.save(model.module.state_dict(), os.path.join(args.model_dir, 'best_model.pt'))
            else:
                torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pt'))

            # save model training stats
            pickle.dump(history, open(os.path.join(args.model_dir, 'best_model.npy'), 'wb'))
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement > args.patience:
            print('Initiating early stopping')
            break

    print(f'Best Epoch:{best_epoch}')
    print(f'     PSNR: {best_psnr}')
    print(f'     loss: {best_val_loss}')

    # saving final model
    print('Saving final model')
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'final_model.pt'))
    pickle.dump(history, open(os.path.join(args.model_dir, 'final_model.npy'), 'wb'))

def parse_args():
    parser = argparse. ArgumentParser(description='Gamma-Spectra Denoising Trainer')
    parser.add_argument("-gn", "--gennoise", help="use noise as target", default=False, action="store_true")
    parser.add_argument('--det_type', type=str, default='NaI', help='detector type to train {HPGe, NaI, CZT}')
    parser.add_argument('--train_set', type=str, default='data/NAI/training.h5', help='h5 file with training vectors')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs of no improvment before early stopping')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='L2 coefficient')
    parser.add_argument('--l1', type=float, default=1e-5, help='L1 coefficient')
    parser.add_argument('--lr_decay', type=float, default=0.94, help='learning rate decay factor')
    parser.add_argument('--num_layers', type=int, default=5, help='number of CNN layers in network')
    parser.add_argument('--num_filters', type=int, default=16, help='number of filters per CNN layer')
    parser.add_argument('--filter_size', type=int, default=3, help='size of filter for CNN layers')
    parser.add_argument('--stride', type=int, default=1, help='filter stride for CNN layers')
    parser.add_argument('--res', default=False, help='use model with residual blocks', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--model_dir', type=str, default='models', help='location of model files')
    args = parser.parse_args()

    return args


def main(args):
    start = time.time()

    # make sure data files exist
    assert os.path.exists(args.train_set), f'Cannot find training vectors file {args.train_set}'

    # detect gpus/cpu and setup environment variables
    setup_device(args)

    # applying random seed for reproducability
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load all data and create the data loaders for training
    train_loader, val_loader = create_data_loaders(args)

    # create the model and prepare it for training
    model, criterion, optimizer = build_model(args)
    print(model)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # main training loop
    train_model(model, criterion, optimizer, train_loader, val_loader, args)
    
    print(f'Script completed in {time.time()-start:.2f} secs')
    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
