import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spectra_utils import data_load_normalized
from spectra_utils import split_radionuclide_name

def plot_accuracy(model, SSLCA, SSLCA_BS, outdir):

    
    labels = ['raw spectrum', 'background subtraction', 'GS-DnCNN denoised']
    bars = [SSLCA['total_correct']/model['num']*100, SSLCA_BS['total_correct']/SSLCA_BS['num']*100, model['total_correct']/model['num']*100]
    styles = {'colors': ['red', 'green', 'blue'],
              'hatch': ['+', '\\', 'x']}

    for model in range(len(labels)):
        plt.bar(labels[model], bars[model], edgecolor='black', color=styles['colors'][model], hatch=styles['hatch'][model], label=labels[model], zorder=3)

    plt.xticks([0,1,2], ['', '', ''])
    #plt.grid(zorder=0)
    plt.legend(fancybox=True, shadow=True, fontsize=11, loc='upper right', framealpha=0.6)
    plt.ylabel('Accuracy (%)', font='cmtt10', fontsize=16, fontweight='bold')
    plt.xlabel('SSLCA-Results', font='cmtt10', fontsize=16, fontweight='bold')

    bottom = 0.0
    top = 75.0
    ax = plt.gca()
    ax.set_yticks(np.arange(bottom, top, 10))
    ax.set_yticks(np.arange(bottom, top, 2), minor=True)
    ax.grid(axis='y', which='major', alpha=0.5)
    ax.grid(axis='y', which='minor', alpha=0.2)

    plt.tight_layout()

    plt.savefig(os.path.join(outdir, 'results.pdf'),format='pdf', dpi=300)
    plt.show()

def plot_accuracy_RN(model, SSLCA, SSLCA_BS, outdir):
    labels = [split_radionuclide_name(rn) for rn in model['RN'].keys()]
    labels = ["${}^{"+rn_num+"}{"+rn_name+"}$" for rn_num, rn_name in labels]

    results = []
    models = []
    models.append('raw spectrum')
    results.append(get_model_RN_results(SSLCA))
    models.append('background subtraction')
    results.append(get_model_RN_results(SSLCA_BS))
    models.append('GS-DnCNN Denoised')
    results.append(get_model_RN_results(model))

    barWidth = 0.3
    locs = []
    locs.append(np.arange(len(labels)))
    locs.append([x + barWidth for x in locs[0]])
    locs.append([x + 2*barWidth for x in locs[0]])

    styles = {'colors': ['red', 'green', 'blue'],
              'hatch': ['+', '\\', 'x']}

    for idx in range(len(locs)):
        plt.bar(locs[idx], results[idx], width = barWidth, color = styles['colors'][idx], 
                edgecolor='black', hatch=styles['hatch'][idx], label=models[idx], zorder=3)

    plt.legend(fancybox=True, shadow=True, fontsize=11, loc='upper right')
    plt.ylabel('Accuracy (%)', font='cmtt10', fontsize=16, fontweight='bold')
    plt.xlabel('SSLCA-Results', font='cmtt10', fontsize=16, fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(labels))], labels, fontsize=12, fontname='cmtt10')

    bottom = 0.0
    top = 101.0
    ax = plt.gca()
    ax.set_yticks(np.arange(bottom, top, 10))
    ax.set_yticks(np.arange(bottom, top, 2), minor=True)
    ax.grid(axis='y', which='major', alpha=0.5)
    ax.grid(axis='y', which='minor', alpha=0.2)

    plt.tight_layout()

    plt.savefig(os.path.join(outdir, 'resultsRN.pdf'),format='pdf', dpi=300)
    plt.show()

def plot_accuracy_SNRS(model, SSLCA, SSLCA_BS, outdir):

    fig, ax = plt.subplots(3, 1)

    bottom, top = plt.ylim()
    bottom = -20.0
    top = 25.1

    #x1 = np.arange(1,len(model['incorrect_snrs'])+1)
    #x2 = np.arange(len(model['incorrect_snrs'])+1, len(model['incorrect_snrs'])+len(model['correct_snrs'])+1)

    #ax[2].scatter(x1, sorted(model['incorrect_snrs']), color='red', marker='x', label='incorrect classifications')
    #ax[2].scatter(x2, sorted(model['correct_snrs']), color='blue', marker='+', label='correct classifications')

    x1, y1, x2, y2 = get_snr_scatter(model['incorrect_snrs'], model['correct_snrs'])
    ax[2].scatter(x1, y1, color='red', marker='x', label='SSLCA incorrect classifications')
    ax[2].scatter(x2, y2, color='blue', marker='+', label='SSLCA correct classifications')

    ax[2].set_xticks(np.arange(0, len(x1)+len(x2)+1, 5))
    ax[2].set_xticks(np.arange(0, len(x1)+len(x2)+1, 1), minor=True)
    ax[2].set_yticks(np.arange(bottom, top, 10))
    ax[2].set_yticks(np.arange(bottom, top, 2), minor=True)
    ax[2].grid(axis='y', which='major', alpha=0.5)
    ax[2].grid(axis='y', which='minor', alpha=0.2)
    ax[2].grid(axis='x', which='major', alpha=0.5)
    ax[2].grid(axis='x', which='minor', alpha=0.2)

    ax[2].set_ylabel('SNR (dB)', font='cmtt10', fontsize=16, fontweight='bold')
    ax[2].set_xlabel('Testset Spectra (GS-DnCNN Denoised)', font='cmtt10', fontsize=16, fontweight='bold')
    ax[2].legend(fancybox=True, shadow=True, fontsize=11, loc='upper left')

    #x1 = np.arange(1,len(SSLCA['incorrect_snrs'])+1)
    #x2 = np.arange(len(SSLCA['incorrect_snrs'])+1, len(SSLCA['incorrect_snrs'])+len(SSLCA['correct_snrs'])+1)

    #ax[0].scatter(x1, sorted(SSLCA['incorrect_snrs']), color='red', marker='x', label='SSLCA incorrect classifications')
    #ax[0].scatter(x2, sorted(SSLCA['correct_snrs']), color='blue', marker='+', label='SSLCA correct classifications')

    x1, y1, x2, y2 = get_snr_scatter(SSLCA['incorrect_snrs'], SSLCA['correct_snrs'])
    ax[0].scatter(x1, y1, color='red', marker='x', label='SSLCA incorrect classifications')
    ax[0].scatter(x2, y2, color='blue', marker='+', label='SSLCA correct classifications')

    ax[0].set_xticks(np.arange(0, len(x1)+len(x2)+1, 5))
    ax[0].set_xticks(np.arange(0, len(x1)+len(x2)+1, 1), minor=True)
    ax[0].set_yticks(np.arange(bottom, top, 10))
    ax[0].set_yticks(np.arange(bottom, top, 2), minor=True)
    ax[0].grid(axis='y', which='major', alpha=0.5)
    ax[0].grid(axis='y', which='minor', alpha=0.2)
    ax[0].grid(axis='x', which='major', alpha=0.5)
    ax[0].grid(axis='x', which='minor', alpha=0.2)

    ax[0].set_ylabel('SNR (dB)', font='cmtt10', fontsize=16, fontweight='bold')
    ax[0].set_xlabel('Testset Spectra (Raw Spectrum)', font='cmtt10', fontsize=16, fontweight='bold')
    ax[0].legend(fancybox=True, shadow=True, fontsize=11, loc='upper left')

    #x1 = np.arange(1,len(SSLCA_BS['incorrect_snrs'])+1)
    #x2 = np.arange(len(SSLCA_BS['incorrect_snrs'])+1, len(SSLCA_BS['incorrect_snrs'])+len(SSLCA_BS['correct_snrs'])+1)

    #ax[1].scatter(x1, sorted(SSLCA_BS['incorrect_snrs']), color='red', marker='x', label='SSLCA incorrect classifications')
    #ax[1].scatter(x2, sorted(SSLCA_BS['correct_snrs']), color='blue', marker='+', label='SSLCA correct classifications')

    x1, y1, x2, y2 = get_snr_scatter(SSLCA_BS['incorrect_snrs'], SSLCA_BS['correct_snrs'])
    ax[1].scatter(x1, y1, color='red', marker='x', label='SSLCA incorrect classifications')
    ax[1].scatter(x2, y2, color='blue', marker='+', label='SSLCA correct classifications')

    ax[1].set_xticks(np.arange(0, len(x1)+len(x2)+1, 5))
    ax[1].set_xticks(np.arange(0, len(x1)+len(x2)+1, 1), minor=True)
    ax[1].set_yticks(np.arange(bottom, top, 10))
    ax[1].set_yticks(np.arange(bottom, top, 2), minor=True)
    ax[1].grid(axis='y', which='major', alpha=0.5)
    ax[1].grid(axis='y', which='minor', alpha=0.2)
    ax[1].grid(axis='x', which='major', alpha=0.5)
    ax[1].grid(axis='x', which='minor', alpha=0.2)

    ax[1].set_ylabel('SNR (dB)', font='cmtt10', fontsize=16, fontweight='bold')
    ax[1].set_xlabel('Testset Spectra (Background Subtracted)', font='cmtt10', fontsize=16, fontweight='bold')
    ax[1].legend(fancybox=True, shadow=True, fontsize=11, loc='upper left')

    plt.tight_layout()
    fig.set_size_inches(11, 8.5)
    
    plt.savefig(os.path.join(outdir, 'resultsSNR.pdf'),format='pdf', dpi=300)
    plt.show()

def get_snr_scatter(incorrect_snrs, correct_snrs):
    snrs = np.array(incorrect_snrs + correct_snrs)
    target = np.array([0] * len(incorrect_snrs) + [1] * len(correct_snrs))
    order = snrs.argsort(axis=0)
    snrs = snrs[order]
    target = target[order]

    x1 = []
    y1 = []
    x2 = [] 
    y2 = []
    for i in range(len(snrs)):
        if target[i] == 0:
            x1.append(i)
            y1.append(snrs[i])
        else:
            x2.append(i)
            y2.append(snrs[i])

    return x1, y1, x2, y2

def get_model_RN_results(model):

    results = []
    for rn in model['RN']:
        results.append(model['RN'][rn]['correct']/model['RN'][rn]['num']*100)
    
    return results

def snr(source, noise):
    # calculate RMS of signal, noise and Compton
    sRMS = np.sqrt(np.mean(source**2))
    nRMS = np.sqrt(np.mean(noise**2))

    snr = 20*np.log10(sRMS/nRMS)

    return snr

def rn_results(rn, results):

    correct = 0
    incorrect = 0
    no_detection = 0
    spectra = [] 

    lines = results.split('\n')
    for line_num in range(len(lines)):
        if 'Spectra' in lines[line_num]:
            spectrum = lines[line_num].split(' ')[-1].replace('-denoised', '').replace('denoised/', '')
            line_num += 1
            if 'Template' in lines[line_num] and lines[line_num].split(' ')[-1] == rn:
                while line_num < len(lines):
                    line_num += 1
                    if 'Correct' in lines[line_num]:
                        correct += 1
                        spectra.append({'spectrum': spectrum, 'correct': 1})
                        break 
                    if 'Incorrect' in lines[line_num]:
                        incorrect += 1
                        spectra.append({'spectrum': spectrum, 'correct': 0})
                        break 
                    if 'No Detection' in lines[line_num]:
                        no_detection += 1
                        spectra.append({'spectrum': spectrum, 'correct': 0})
                        break

    return {'correct': correct, 'incorrect': incorrect, 'no_detection': no_detection, 'spectra': spectra}

def get_results_by_RN(results):

    rns = []

    for line in results.split('\n'):
        if 'Template' in line:
            rns.append(line.split(' ')[-1])

    rns = list(set(rns))
    resultsRN = {rn: rn_results(rn, results) for rn in rns}

    return resultsRN


def get_results(results_loc):

    results = {}

    with open(results_loc, 'r') as f:
        data = f.read()

    results['correct'] = data.count('Correct')
    results['incorrect'] = data.count('Incorrect')
    results['no_detection'] = data.count('No Detection')

    results['RN'] = get_results_by_RN(data)

    return results

def get_testset_results(testset, results_dir):

    results = {'testsets': []} 

    df = pd.read_csv(testset)

    for ts in df['location']:
        results_loc = os.path.join(results_dir, os.path.basename(ts), 'SSLCA-results.txt')
        print(f'loading results from {results_loc}')
        ts_results = get_results(results_loc)
        results['testsets'].append(ts_results)

    total_correct, num = get_accuracy(results)
    results['total_correct'] = total_correct
    results['num'] = num

    correct_snrs, incorrect_snrs = get_snrs(results)
    results['correct_snrs'] = correct_snrs
    results['incorrect_snrs'] = incorrect_snrs

    rn_results = get_accuracy_RN(results)

    return results

def get_accuracy(results):

    total_correct = 0
    num = 0

    for res in results['testsets']:
        total_correct += res['correct']
        num += res['correct'] + res['incorrect'] + res['no_detection']

    return total_correct, num

def get_accuracy_RN(results):

    rns = {}

    for ts in results['testsets']:
        for rn in ts['RN']:
            correct = ts['RN'][rn]['correct']
            num = ts['RN'][rn]['correct'] + ts['RN'][rn]['incorrect'] + ts['RN'][rn]['no_detection']
            if rn not in rns:
                rns[rn] = {'correct': correct, 'num': num}
            else:
                rns[rn]['correct'] += correct
                rns[rn]['num'] += num

    results['RN'] = rns

def get_snrs(results):

    correct_snrs = []
    incorrect_snrs = []

    for ts in results['testsets']:
        for rn in ts['RN']:
            for spec in ts['RN'][rn]['spectra']:
                _, hits = data_load_normalized(spec['spectrum'])
                background = os.path.join(os.path.dirname(spec['spectrum']), 'background.json')
                _, back_hits = data_load_normalized(background)
                spec_snr = snr(hits-back_hits, back_hits)
                spec['SNR'] = spec_snr 
                if spec['correct']:
                    correct_snrs.append(spec_snr)
                else:
                    incorrect_snrs.append(spec_snr)

    return correct_snrs, incorrect_snrs

def plot_results(model, SSLCA, SSLCA_BS, outdir):

    plot_accuracy(model, SSLCA, SSLCA_BS, outdir)

    plot_accuracy_RN(model, SSLCA, SSLCA_BS, outdir)

    plot_accuracy_SNRS(model, SSLCA, SSLCA_BS, outdir)


def parse_args():
    parser = argparse. ArgumentParser(description='Gamma-Spectra Denoising testing with SSLCA identification')
    parser.add_argument('--testset', type=str, default='spectra/NaI/testset.csv', help='csv file of testset spectra locations')
    parser.add_argument('--model_results_dir', type=str, default='models/results', help='location of model testet results')
    parser.add_argument('--SSLCA_results_dir', type=str, default='SSLCA_results/NaI', help='location of SSLCA testet results without background subtraction')
    parser.add_argument('--SSLCA_BS_results_dir', type=str, default='SSLCA_results/NaI-BS', help='location of SSLCA testet results with background subtraction')

    return parser.parse_args()

def main(args):
    start = time.time()

    results_model = get_testset_results(args.testset, args.model_results_dir)

    results_SSLCA = get_testset_results(args.testset, args.SSLCA_results_dir)

    results_SSLCA_BS = get_testset_results(args.testset, args.SSLCA_BS_results_dir)

    plot_results(results_model, results_SSLCA, results_SSLCA_BS, args.model_results_dir)

    print(f'GS-DnCNN accuracy for entire test is {results_model["total_correct"]/results_model["num"]:.4f}')
    print(f'SSLCA accuracy without background subtraction for entire test is {results_SSLCA["total_correct"]/results_SSLCA["num"]:.4f}')
    print(f'SSLCA accuracy with background subtraction for entire test is {results_SSLCA_BS["total_correct"]/results_SSLCA_BS["num"]:.4f}')

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
