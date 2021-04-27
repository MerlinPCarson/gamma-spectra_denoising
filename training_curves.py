import sys
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from matplotlib.ticker import MultipleLocater
from decimal import Decimal


def plot_curves(history, lr_rates, outfile):

    # color choices for learning rate lines
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # plot comparisons side by side
    fig, ax = plt.subplots(2,1)
    dims = np.arange(1, len(history['val'])+1)
    ax[0].plot(dims, history['val'], label='validation set', color='red')

    # clip first training loss since it is usually high and messes with scale
    bottom, top = ax[0].get_ylim()
    history['train'][0] = min(history['train'][0], top)

    ax[0].plot(dims, history['train'], label='training set', color='blue')
    ax[0].set_xlabel('Epoch', fontsize=24, fontweight='bold', fontname='cmtt10')
    ax[0].set_ylabel('Loss', fontsize=24, fontweight='bold', fontname='cmtt10')
    ax[0].set_xticks(np.arange(0, len(history['val']), 10))
    ax[0].set_xticks(np.arange(0, len(history['val']), 2), minor=True)
    ax[0].grid(axis='x', which='major', alpha=0.5)
    ax[0].grid(axis='x', which='minor', alpha=0.2)
    ax[0].grid(axis='y', which='major', alpha=0.5)
    ax[0].grid(axis='y', which='minor', alpha=0.2)
    ax[0].tick_params(axis='both', labelsize=12)

    for i in range(len(lr_rates[0])):
        # offset by since epoch 1 in index 0
        ax[0].axvline(lr_rates[1][i]+1, linestyle='--', color=cycle_colors[i], label=f'learning rate: {lr_rates[0][i]:.0E}')

    ax[0].legend(fancybox=True, shadow=True, fontsize=16, loc='upper right', framealpha=0.8)

    ax[1].plot(dims, history['psnr'], label='validation set', color='red')
    ax[1].set_xlabel('Epoch', fontsize=24, fontweight='bold', fontname='cmtt10')
    ax[1].set_ylabel('PSNR', fontsize=24, fontweight='bold', fontname='cmtt10')
    ax[1].set_xticks(np.arange(0, len(history['psnr']), 10))
    ax[1].set_xticks(np.arange(0, len(history['psnr']), 2), minor=True)
    ax[1].grid(axis='x', which='major', alpha=0.5)
    ax[1].grid(axis='x', which='minor', alpha=0.2)
    ax[1].grid(axis='y', which='major', alpha=0.5)
    ax[1].grid(axis='y', which='minor', alpha=0.2)
    ax[1].tick_params(axis='both', labelsize=12)

    for i in range(len(lr_rates[0])):
        # offset by since epoch 1 in index 0
        ax[1].axvline(lr_rates[1][i]+1, linestyle='--', color=cycle_colors[i], label=f'learning rate: {lr_rates[0][i]:.0E}')

    ax[1].legend(fancybox=True, shadow=True, fontsize=16, loc='lower right', framealpha=0.8)

    # make image full screen
    fig = plt.gcf()
    fig.set_size_inches(18, 11)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.show()

def find_lr_changes(lrs):

    lr_rates = np.array(sorted(list(set(lrs)), reverse=True))

    lr_epochs = []
    for lr in lr_rates:
        epoch = np.where(lrs==lr)[0][0]
        lr_epochs.append(epoch)

    return np.array([lr_rates, lr_epochs])

def parse_args():

    parser = argparse.ArgumentParser(description='Plot multiple spectra')
    parser.add_argument("-mf", "--modelfile", help="model history file", default="models-vispa/best_model.npy")
    args = parser.parse_args()

    return args


def main(args):
    start = time.time()

    model_history = pickle.load(open(args.modelfile, 'rb'))

    lr_rates = find_lr_changes(model_history['lr'])

    outfile = args.modelfile.replace('.npy', '.pdf')
    plot_curves(model_history, lr_rates, outfile)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
