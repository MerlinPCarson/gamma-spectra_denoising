import os
import sys
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from experiments import create_grid
from sklearn.model_selection import ParameterGrid

def best_results(results):

    final_losses = np.array([exp['history']['val'][-(exp['params']['patience']+2)] for exp in results])
    final_PSNRs = np.array([exp['history']['psnr'][-(exp['params']['patience']+2)] for exp in results])

    best_exp = np.argmin(final_losses)
    print(f'{np.min(final_losses)} with params l1 = {results[best_exp]["params"]["l1"]}, l2 = {results[best_exp]["params"]["l1"]}')

    return final_losses, final_PSNRs
    #for psnr in PSNRs:
    #    print(f'{max(psnr):.4f} dB ?= {psnr[-12]:.4f}')
    #for i in range(len(results)):
    #    loss = results[i]['history']['val']
    #    print(min(loss), loss[-(results[i]['params']['patience']+2)])
    #print(losses)

def get_params_mesh(losses):

    param_grid = create_grid() 
    params = list(ParameterGrid(param_grid))

    x1 = [x['l1'] for x in params]
    x2 = [x['l2'] for x in params]

    x_vals = sorted(set(x1))
    y_vals = sorted(set(x2))

    X, Y = np.meshgrid(x_vals, y_vals)

    Z = losses.reshape((len(x_vals),len(y_vals))).T

    return X, Y, Z

def plot_contour(X, Y, Z, metric, title):
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='coolwarm', edgecolor='none')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=8)

    # new labels
    x_vals = [str(x) for x in X[0,:]]
    y_vals = [str(y) for y in Y[:,0]]

    ax.set_title(title)
    ax.set_xlabel('L1 (λ)')
    ax.set_xticklabels(x_vals)
    ax.set_ylabel('L2 (λ)')
    ax.set_yticklabels(y_vals)
    ax.set_zlabel(metric)

    plt.tight_layout()

    plt.show()

def parse_args():
    parser = argparse. ArgumentParser(description='Collect and show results of given experiment')
    parser.add_argument('--exp_dir', type=str, default='all_exps/exps_regularization', help='location of experiment model and results file ')
    parser.add_argument('--results_file', type=str, default='exp_results.npy', help='experiment results file')
    parser.add_argument('--exp_type', type=str, default='Regularization Experiments', help='experiment type')

    return parser.parse_args()

def main(args):
    start = time.time()

    results_file = os.path.join(args.exp_dir, args.results_file)
    results = pickle.load(open(results_file,'rb'))

    losses, psnrs = best_results(results)

    X, Y, Z = get_params_mesh(losses)
    plot_contour(X, Y, Z, 'validation loss', args.exp_type)

    X, Y, Z = get_params_mesh(psnrs)
    plot_contour(X, Y, Z, 'validation PSNR', args.exp_type)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
