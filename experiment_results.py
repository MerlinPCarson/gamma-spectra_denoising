import os
import sys
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from experiments import create_grid
from sklearn.model_selection import ParameterGrid

def best_results(results, param1, param2):

    final_losses = np.array([exp['history']['val'][-(exp['params']['patience']+2)] for exp in results])
    final_PSNRs = np.array([exp['history']['psnr'][-(exp['params']['patience']+2)] for exp in results])
    param1_vals = np.array([exp['params'][param1] for exp in results])
    param2_vals = np.array([exp['params'][param2] for exp in results])

    best_exp = np.argmin(final_losses)
    print(f'Best model model_{best_exp}')
    print(f'{final_losses[best_exp]} val loss with params {param1} = {results[best_exp]["params"][param1]}, \
            {param2} = {results[best_exp]["params"][param2]}')
    print(f'{final_PSNRs[best_exp]} dB with params {param1} = {results[best_exp]["params"][param1]}, \
            {param2} = {results[best_exp]["params"][param2]}')

    return final_losses, final_PSNRs, param1_vals, param2_vals

def get_params_mesh(losses, x1, x2):

    x_vals = sorted(set(x1))
    y_vals = sorted(set(x2))

    X, Y = np.meshgrid(x_vals, y_vals)

    Z = losses.reshape((len(x_vals),len(y_vals))).T

    return X, Y, Z

def plot_contour(X, Y, Z, metric, title, param1, param2):
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='coolwarm', edgecolor='none')

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=8)

    # new labels
    x_vals = [str(x) for x in X[0,:]]
    y_vals = [str(y) for y in Y[:,0]]

    ax.set_title(title)
    ax.set_xlabel(param1)
    ax.set_xticklabels(x_vals)
    ax.set_ylabel(param2)
    ax.set_yticklabels(y_vals)
    ax.set_zlabel(metric)

    plt.tight_layout()

    plt.show()

def parse_args():
    parser = argparse. ArgumentParser(description='Collect and show results of given experiment')
    parser.add_argument('--exp_dir', type=str, default='all_exps/exps_regularization', help='location of experiment model and results file ')
    parser.add_argument('--results_file', type=str, default='exp_results.npy', help='experiment results file')
    parser.add_argument('--exp_type', type=str, default='Regularization Experiments', help='experiment type')
    parser.add_argument('--param1', type=str, default='l1', help='first parameter of grid search')
    parser.add_argument('--param2', type=str, default='l2', help='second parameter of grid search')

    return parser.parse_args()

def main(args):
    start = time.time()

    results_file = os.path.join(args.exp_dir, args.results_file)
    results = pickle.load(open(results_file,'rb'))

    losses, psnrs, param1_vals, param2_vals = best_results(results, args.param1, args.param2)

    X, Y, Z = get_params_mesh(losses, param1_vals, param2_vals)
    plot_contour(X, Y, Z, 'validation loss', args.exp_type, args.param1, args.param2)

    X, Y, Z = get_params_mesh(psnrs, param1_vals, param2_vals)
    plot_contour(X, Y, Z, 'validation PSNR', args.exp_type, args.param1, args.param2)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
