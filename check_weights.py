import sys
import time
import argparse
import pickle
import torch
from model import DnCNN, DnCNN_Res


def print_weights(model):
    total_weight_sum = 0.0
    for i, param in enumerate(model.parameters()):
        if param.requires_grad:
            print(f'{i}: {len(param.flatten())}')
            total_weight_sum += sum(param.flatten().abs())
    print(f'weight sum: {total_weight_sum}')

def parse_args():
    parser = argparse. ArgumentParser(description='Gamma-Spectra Denoising testing dataset')
    parser.add_argument('--model', type=str, default='aug_noise0.0/dilation_models/models-best-alldata-allcomp-12e-4/best_model.pt', help='location of model to use')
    return parser.parse_args()

def main(args):
    start = time.time()

    # load parameters for model
    params = pickle.load(open(args.model.replace('.pt','.npy'),'rb'))['model']

    model = DnCNN(num_channels=params['num_channels'], num_layers=params['num_layers'], 
                    kernel_size=params['kernel_size'], num_filters=params['num_filters'],
                    dilation_rate=params['dilation_rate'])

    print(model)
    print(f'number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # loaded saved model
    print(f'Loading weights for {params["model_name"]} model from {args.model} for {params["model_type"]}')
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    print_weights(model)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
