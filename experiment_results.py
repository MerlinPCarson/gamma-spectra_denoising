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
    print(f'{final_losses[best_exp]} val loss with params {param1} = {results[best_exp]["params"][param1]},', \
            f'{param2} = {results[best_exp]["params"][param2]}')
    print(f'{final_PSNRs[best_exp]} dB with params {param1} = {results[best_exp]["params"][param1]},', \
            f'{param2} = {results[best_exp]["params"][param2]}')

    return final_losses, final_PSNRs, param1_vals, param2_vals

def get_params_mesh(losses, x1, x2):

    x_vals = sorted(set(x1))
    y_vals = sorted(set(x2))

    X, Y = np.meshgrid(x_vals, y_vals)

    Z = losses.reshape((len(x_vals),len(y_vals))).T

    return X, Y, Z

def plot_contour(X, Y, Z, metric, title, param1, param2, outdir):
    
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

    outfile = os.path.join(outdir, f'{metric}_{param1}_{param2}.pdf')
    plt.savefig(outfile, dpi=300)

    plt.show()

def plot_contours(X1, Y1, Z1, metric1, X2, Y2, Z2, metric2, title, param1, param2, outdir):
    import matplotlib.ticker as mticker 
    # new labels
    #x_vals = [f'{x:.2E}' for x in X1[0,:]]
    #x_vals = [f'{x}' for x in X1[0,:]]
    #y_vals = [f'{y}' for y in Y1[:,0]]
    #print(x_vals)

    fig = plt.figure()
    #plt.title(title)
    ax1 = fig.add_subplot(121,projection='3d')
    surf = ax1.plot_surface(X1, Y1, Z1, rstride=1, cstride=1,
                    cmap='coolwarm', edgecolor='none')
    ax1.set_xlabel(param1)
    ax1.set_ylabel(param2)

    # Regularization labels
    #ax1.set_xlabel('L1', fontsize=24, fontname='cmtt10', labelpad=20)
    #ax1.set_ylabel('L2', fontsize=24, fontname='cmtt10', labelpad=20)
    #ax1.set_zlabel(metric1, fontsize=24, fontname='cmtt10', labelpad=25)

    # Model size labels
    #ax1.set_xlabel('Num Kernels', fontsize=24, fontname='cmtt10', labelpad=15)
    #ax1.set_ylabel('Num Layers', fontsize=24, fontname='cmtt10', labelpad=15)
    #ax1.set_zlabel(metric1, fontsize=24, fontname='cmtt10', labelpad=15)

    # Model learning labels
    ax1.set_xlabel('Learning Rate', fontsize=24, fontname='cmtt10', labelpad=15)
    ax1.set_ylabel('Batch Size', fontsize=24, fontname='cmtt10', labelpad=15)
    ax1.set_zlabel(metric1, fontsize=24, fontname='cmtt10', labelpad=20)

#    ticks_loc = ax1.get_xticks().tolist()
#    ax1.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
#    x_vals = np.linspace(ticks_loc[0], ticks_loc[-1], len(ticks_loc))
    #x_vals = [f'{x:.2E}' for x in x_vals]
#    ax1.set_xticklabels(x_vals, fontsize=16)
#    ticks_loc = ax1.get_yticks().tolist()
#    ax1.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
#    y_vals = np.linspace(ticks_loc[0], ticks_loc[-1], len(ticks_loc))
#    ax1.set_yticklabels(y_vals, fontsize=16)
    #ticks_loc = ax1.get_zticks().tolist()
    #ax1.zaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    #z_vals = np.linspace(ticks_loc[0], ticks_loc[-1], len(ticks_loc))
    #z_vals = [f'{x:.3f}' for x in ticks_loc]
    #ax1.set_zticklabels(z_vals, fontsize=12)
    #tick_labels = ax1.get_zticklabels()
    ax1.tick_params(axis='both', labelsize=12)
    #ax1.tick_params(axis='z', labelsize=12, pad=5)
    ax1.tick_params(axis='z', labelsize=12, pad=7)

    ax2 = fig.add_subplot(122,projection='3d')
    surf = ax2.plot_surface(X2, Y2, Z2, rstride=1, cstride=1,
                    cmap='coolwarm', edgecolor='none')
    ax2.set_xlabel(param1)
    ax2.set_ylabel(param2)

    # Regularization labels
    #ax2.set_xlabel('L1', fontsize=24, fontname='cmtt10', labelpad=20)
    #ax2.set_ylabel('L2', fontsize=24, fontname='cmtt10', labelpad=20)
    #ax2.set_zlabel(metric1, fontsize=24, fontname='cmtt10', labelpad=25)

    # Model size and learning labels
    #ax2.set_xlabel('Num Kernels', fontsize=24, fontname='cmtt10', labelpad=15)
    #ax2.set_ylabel('Num Layers', fontsize=24, fontname='cmtt10', labelpad=15)
    ax2.set_xlabel('Learning Rate', fontsize=24, fontname='cmtt10', labelpad=15)
    ax2.set_ylabel('Batch Size', fontsize=24, fontname='cmtt10', labelpad=15)
    ax2.set_zlabel(metric2, fontsize=24, fontname='cmtt10', labelpad=10)

#    ticks_loc = ax2.get_xticks().tolist()
#    ax2.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
#    x_vals = np.linspace(ticks_loc[0], ticks_loc[-1], len(ticks_loc))
#    ax2.set_xticklabels(x_vals, fontsize=16)
#    ticks_loc = ax2.get_yticks().tolist()
#    ax2.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
#    y_vals = np.linspace(ticks_loc[0], ticks_loc[-1], len(ticks_loc))
#    ax2.set_yticklabels(y_vals, fontsize=16)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='z', labelsize=12, pad=2)

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=8)


    #ax.set_title(title)
    #ax.set_xlabel(param1)
    #ax.set_xticklabels(x_vals)
    #ax.set_ylabel(param2)
    #ax.set_yticklabels(y_vals)
    #ax.set_zlabel(metric)

    #plt.tight_layout()

    fig = plt.gcf()
    fig.set_size_inches(18, 11)

    outfile = os.path.join(outdir, f'{param1}_{param2}.pdf')
    plt.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

def parse_args():
    parser = argparse. ArgumentParser(description='Collect and show results of given experiment')
    parser.add_argument('--exp_dir', type=str, default='exps_dilation/exps_train3', help='location of experiment model and results file ')
    parser.add_argument('--results_file', type=str, default='exp_results.npy', help='experiment results file')
    parser.add_argument('--exp_type', type=str, default='Regularization Experiments', help='experiment type')
    parser.add_argument('--param1', type=str, default='lr', help='first parameter of grid search')
    parser.add_argument('--param2', type=str, default='batch_size', help='second parameter of grid search')

    return parser.parse_args()

def main(args):
    start = time.time()

    results_file = os.path.join(args.exp_dir, args.results_file)
    results = pickle.load(open(results_file,'rb'))

    losses, psnrs, param1_vals, param2_vals = best_results(results, args.param1, args.param2)

#    X, Y, Z = get_params_mesh(losses, param1_vals, param2_vals)
#    plot_contour(X, Y, Z, 'validation loss', args.exp_type, args.param1, args.param2, args.exp_dir)
#
#    X, Y, Z = get_params_mesh(psnrs, param1_vals, param2_vals)
#    plot_contour(X, Y, Z, 'validation PSNR', args.exp_type, args.param1, args.param2, args.exp_dir)

    X1, Y1, Z1 = get_params_mesh(losses, param1_vals, param2_vals)

    X2, Y2, Z2 = get_params_mesh(psnrs, param1_vals, param2_vals)

    plot_contours(X1, Y1, Z1, 'validation loss', X2, Y2, Z2, 'validation PSNR', args.exp_type, args.param1, args.param2, args.exp_dir)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
