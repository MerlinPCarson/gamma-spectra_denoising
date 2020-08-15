import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_history(loss, valloss, psnr, show_plot=True):

    #plt.figure(figsize=(20,10))

    plt.subplot(1,2,1)
    plt.plot(loss, label='Training', color='blue')
    plt.plot(valloss, label='Validation', color='red')
    
    ax = plt.gca()
    ax.set_xlabel('epoch', fontsize=18, fontweight='bold', fontname='cmtt10')
    ax.set_ylabel('loss', fontsize=18, fontweight='bold', fontname='cmtt10')
    ax.set_xticks(np.arange(0, len(loss), 20))
    ax.set_xticks(np.arange(0, len(loss), 5), minor=True)
    ax.grid(axis='x', which='major', alpha=0.5)
    ax.grid(axis='x', which='minor', alpha=0.2)

    plt.legend(fancybox=True, shadow=True, fontsize=11)
    plt.tight_layout()

    plt.subplot(1,2,2)
    plt.plot(psnr, label='Validation', color='red')
    
    ax = plt.gca()
    ax.set_xlabel('epoch', fontsize=18, fontweight='bold', fontname='cmtt10')
    ax.set_ylabel('PSNR', fontsize=18, fontweight='bold', fontname='cmtt10')
    ax.set_xticks(np.arange(0, len(loss), 20))
    ax.set_xticks(np.arange(0, len(loss), 5), minor=True)
    ax.grid(axis='x', which='major', alpha=0.5)
    ax.grid(axis='x', which='minor', alpha=0.2)

    plt.legend(fancybox=True, shadow=True, fontsize=11, loc=4)
    plt.tight_layout()

    plt.savefig('model_history.png', format='png')

    if show_plot:
        plt.show()

    plt.close()


def main():
    if len(sys.argv) == 2:
        history_file = sys.argv[1]
    else:
        print("Usage: python plot_history.py <history csv file> ")
        return 1

    if not os.path.isfile(history_file):
        print(f"{history_file} not found. ")
        return 1

    history = pickle.load(open(history_file, 'rb'))

    plot_history(history['train'][1:], history['val'][1:], history['psnr'][1:])

    return 0

if __name__ == "__main__":
    sys.exit(main())
