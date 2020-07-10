import os
import sys
import time
import h5py
import pickle
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from load_data import load_data
from model import DnCNN, DnCNN_Res
#from utils import weights_init_kaiming

from sklearn.model_selection import train_test_split
from tqdm import tqdm
#from tensorboardX import SummaryWriter
#from torchvision.utils import make_grid
#from skimage.metrics import peak_signal_noise_ratio as psnr

        
def setup_gpus():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_ids = [i for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
    return device_ids

def main():
    start = time.time()

    parser = argparse. ArgumentParser(description='Gamma-Spectra Denoising Trainer')
    parser.add_argument('--det_type', type=str, default='HPGe', help='detector type to train {HPGe, NaI, CZT}')
    parser.add_argument('--train_set', type=str, default='data/training.h5', help='h5 file with training vectors')
#    parser.add_argument('--val_set', type=str, default='val.h5', help='h5 file with validation vectors')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
#    parser.add_argument('--lr_decay', type=float, default=0.94, help='learning rate decay factor')
    parser.add_argument('--num_layers', type=int, default=5, help='number of CNN layers in network')
    parser.add_argument('--num_filters', type=int, default=64, help='number of filters per CNN layer')
    parser.add_argument('--filter_size', type=int, default=3, help='size of filter for CNN layers')
    parser.add_argument('--stride', type=int, default=1, help='filter stride for CNN layers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--log_dir', type=str, default='logs', help='location of log files')
    parser.add_argument('--model_dir', type=str, default='models', help='location of model files')
    args = parser.parse_args()


    # make sure data files exist
    assert os.path.exists(args.train_set), f'Cannot find training vectors file {args.train_set}'
#    assert os.path.exists(args.val_set), f'Cannot find validation vectors file {args.train_set}'

    # make sure output dirs exists
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # detect gpus and setup environment variables
    device_ids = setup_gpus()
    print(f'Cuda devices found: {[torch.cuda.get_device_name(i) for i in device_ids]}')

    print('Loading datasets')
    training_data = load_data(args.train_set)
    noisy_spectra = training_data[args.det_type.upper()]['noisy']
    clean_spectra = training_data[args.det_type.upper()]['clean']

    assert noisy_spectra.shape == clean_spectra.shape, 'Mismatch between shapes of training and target data'

    # applying random seed for reproducability
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # split data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(noisy_spectra, clean_spectra, test_size = 0.1, random_state=args.seed)


    # get standardization parameters from training set
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)

    # apply standardization parameters to training and validation sets
    x_train = (x_train-train_mean)/train_std
    x_val = (x_val-train_mean)/train_std

    # input shape for each example to network, NOTE: channels first
    num_channels, num_features = 1, x_train.shape[1]
    print(f'Input shape to model forward will be: ({args.batch_size}, {num_channels}, {num_features})')

    # load data for training
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))

    print(f'Number of training examples: {len(x_train)}')
    print(f'Number of validation examples: {len(x_val)}')

    # create batched data loaders for model
    train_loader = DataLoader(dataset=train_dataset, num_workers=os.cpu_count(), batch_size=args.batch_size*len(device_ids), shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=os.cpu_count(), batch_size=args.batch_size, shuffle=False)

    # create model
    model = DnCNN(num_channels=num_channels, num_layers=args.num_layers, \
                  kernel_size=args.filter_size, stride=args.stride, num_filters=args.num_filters) 

    # move model to available gpus
    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    print(model)

    # setup loss and optimizer
    criterion = torch.nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # data struct to track training and validation losses per epoch
    model_params = {'num_channels':num_channels, \
                    'num_layers':args.num_layers, 'kernel_size':args.filter_size,\
                    'stride':args.stride, 'num_filters':args.num_filters, \
                    'train_mean': train_mean, 'train_std': train_std}

    # save model parameters
    history = {'model': model_params, 'train':[], 'val':[], 'psnr':[]}
    pickle.dump(history, open(os.path.join(args.log_dir, 'model.npy'), 'wb'))

#    writer = SummaryWriter(args.log_dir)
#
#    # schedulers
#    #scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=10)
#    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
#
#    # Main training loop
#    best_val_loss = 999 
#    best_psnr_normal = 0
#    best_psnr_uniform = 0
#    best_psnr_pepper = 0
#    for epoch in range(args.epochs):
#        print(f'Starting epoch {epoch+1} with learning rate {optimizer.param_groups[0]["lr"]}')
#
#        model.train()
#        epoch_train_loss = 0
#        train_steps = 0
#
#        # iterate through batches of training examples
#        for clean_imgs in tqdm(train_loader):
#            model.zero_grad()
#
#            # generate additive white noise from gaussian distribution
#            # Model-S
#            #noise = torch.FloatTensor(data.size()).normal_(mean=0, std=noise_level)
#
#            # Model-B
#
#            noise_type = np.random.choice(noise_types)
#            noise = gen_noise(clean_imgs.size(), noise_type)
#
#            # pack input and target it into torch variable
#            noisy_imgs = Variable((clean_imgs + noise).cuda()) 
#            noise = Variable(noise.cuda())
#
#            # make predictions
#            preds = model(torch.clamp(noisy_imgs, 0.0, 1.0))
#
#            # calculate loss
#            loss = criterion(preds, noise)/(2*noisy_imgs.size()[0])
#            epoch_train_loss += loss.detach()
#
#            # backprop
#            loss.backward()
#            optimizer.step()
#
#            train_steps += 1
#
#        # start evaluation
#        model.eval() 
#        print(f'Validating Model')
#        epoch_val_loss = 0
#        epoch_psnr_normal = 0
#        num_normal = 0
#        epoch_psnr_uniform = 0
#        num_uniform = 0
#        epoch_psnr_pepper = 0
#        num_pepper = 0
#        val_steps = 0
#        with torch.no_grad():
#            for clean_imgs in tqdm(val_loader):
#                # generate additive white noise from gaussian distribution
#                # DnCNN-S
#                #noise = torch.FloatTensor(data.size()).normal_(mean=0, std=noise_level)
#
#                # DnCNN-M
#                noise_type = np.random.choice(noise_types)
#                noise = gen_noise(clean_imgs.size(), noise_type)
#
#
#                # pack input and target it into torch variable
#                noisy_imgs = Variable((clean_imgs + noise).cuda())
#                noise = Variable(noise.cuda())
#
#                # make predictions
#                preds = model(torch.clamp(noisy_imgs,0.0,1.0))
#
#                # calculate loss
#                val_loss = criterion(preds, noise)/(2*noisy_imgs.size()[0])
#                epoch_val_loss += val_loss.detach()
#
#                # calculate PSNR 
#                denoised_imgs = torch.clamp(noisy_imgs-preds, 0.0, 1.0)
#                if noise_type == 'normal':
#                    epoch_psnr_normal += psnr_of_batch(clean_imgs, denoised_imgs)
#                    num_normal += 1
#                elif noise_type == 'uniform':
#                    epoch_psnr_uniform += psnr_of_batch(clean_imgs, denoised_imgs)
#                    num_uniform += 1
#                elif noise_type == 'pepper':
#                    epoch_psnr_pepper += psnr_of_batch(clean_imgs, denoised_imgs)
#                    num_pepper += 1
#
#
#                val_steps += 1
#
#        # epoch summary
#        epoch_train_loss /= train_steps
#        epoch_val_loss /= val_steps
#        if num_normal > 0:
#            epoch_psnr_normal /= num_normal 
#        if num_uniform > 0:
#            epoch_psnr_uniform /= num_uniform 
#        if num_pepper > 0:
#            epoch_psnr_pepper /= num_pepper 
#
#        # reduce learning rate if validation has leveled off
#        #scheduler.step(epoch_val_loss)
#
#        # exponential decay of learning rate
#        scheduler.step()
#
#        # save epoch stats
#        history['train'].append(epoch_train_loss)
#        history['val'].append(epoch_val_loss)
#        history['psnr']['normal'].append(epoch_psnr_normal)
#        history['psnr']['uniform'].append(epoch_psnr_uniform)
#        history['psnr']['pepper'].append(epoch_psnr_pepper)
#        print(f'Training loss: {epoch_train_loss}')
#        print(f'Validation loss: {epoch_val_loss}')
#        print(f'Validation PSNR-uniform: {epoch_psnr_uniform}')
#        print(f'Validation PSNR-normal: {epoch_psnr_normal}')
#        print(f'Validation PSNR-pepper: {epoch_psnr_pepper}')
#        writer.add_scalar('loss', epoch_train_loss, epoch)
#        writer.add_scalar('val', epoch_val_loss, epoch)
#        writer.add_scalar('PSNR-normal', epoch_psnr_normal, epoch)
#        writer.add_scalar('PSNR-uniform', epoch_psnr_uniform, epoch)
#        writer.add_scalar('PSNR-pepper', epoch_psnr_pepper, epoch)
#
#        # save if best model
#        if epoch_val_loss < best_val_loss:
#            print('Saving best model')
#            best_val_loss = epoch_val_loss
#            torch.save(model, os.path.join(args.log_dir, 'best_model.pt'))
#            pickle.dump(history, open(os.path.join(args.log_dir, 'best_model.npy'), 'wb'))
#
#        # test model and save results 
#        if epoch % 5 == 0:
#            with torch.no_grad():
#                clean_pics = make_grid(clean_imgs, nrow=8, normalize=True, scale_each=True)
#                writer.add_image('clean images', clean_pics, epoch)
#                for noise_type in noise_types:
#                    denoised_imgs = make_grid(denoised_imgs.data, nrow=8, normalize=True, scale_each=True)
#                    writer.add_image(f'{noise_type} denoised images', denoised_imgs, epoch)
#
#    # saving final model
#    print('Saving final model')
#    torch.save(model, os.path.join(args.log_dir, 'final_model.pt'))
#    pickle.dump(history, open(os.path.join(args.log_dir, 'final_model.npy'), 'wb'))

    print(f'Script completed in {time.time()-start:.2f} secs')
    return 0

if __name__ == '__main__':
    sys.exit(main())
