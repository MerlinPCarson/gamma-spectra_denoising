import os
import sys
import time
import torch
import random
import pickle
import numpy as np

from train_real import parse_args
from train_real import setup_device 
from train_real import create_data_loaders 
from train_real import build_model
from train_real import train_model

from copy import deepcopy
from sklearn.model_selection import ParameterGrid


def create_grid():
    
    grid = {}

    start = 1e-5 
    end = 1 
    num_steps = 6

    # regularization
    #l1_params = [start * 10**x for x in range(num_steps)]
    #l1_params.insert(0, 0.0)
    #l1_params = [0.0] 
    #grid['l1'] = l1_params

    #l2_params = [start * 10**x for x in range(num_steps)]
    #l2_params.insert(0, 0.0)
    #l2_params = np.arange(0,2.0,0.1)
    #grid['l2'] = l2_params

    # layer sizes
    #width_params = [16,32,64,128]
    #grid['num_filters'] = width_params

    #depth_params = [5,10,15,20]
    #grid['num_layers'] = depth_params

    # training params 
    lr_params = [0.1, 0.01, 0.001, 0.0001]
    grid['lr'] = lr_params

    batch_params = [36, 64, 128, 256]
    grid['batch_size'] = batch_params

    return grid

def main(args):
    start = time.time()

    # make sure data files exist
    assert os.path.exists(args.train_set), f'Cannot find training vectors file {args.train_set}'

    # detect gpus/cpu and setup environment variables
    setup_device(args)

    # save base model dir
    model_dir = args.model_dir

    # make experiment directory
    os.makedirs(args.exp_dir)

    param_grid = create_grid() 
    num_exps = len(ParameterGrid(param_grid))
    print(f'Starting experiments with parameters and values: {list(ParameterGrid(param_grid))}')
    exp_results = [] 
    for i, params in enumerate(ParameterGrid(param_grid)):
        print(f'[{i+1}/{num_exps}] Starting experiment with params {params}')
        args.model_dir = os.path.join(args.exp_dir, f'{model_dir}_{i}')
        print(args.model_dir)
        for param, val in params.items():
            exec(f'args.{param}=val')
        
        # applying random seed for reproducability
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # load all data and create the data loaders for training
        train_loader, val_loader = create_data_loaders(args)

        # create the model and prepare it for training
        model, criterion, optimizer = build_model(args)
        print(model)
        args.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'number of parameters: {args.num_params}')

        # main training loop
        history = train_model(model, criterion, optimizer, train_loader, val_loader, args)

        # save results after each iteration, so nothing is lost if there is a failure
        exp_results.append({'params': vars(deepcopy(args)), 'history': history})
        pickle.dump(exp_results, open(os.path.join(args.exp_dir, 'exp_results.npy'), 'wb'))

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
