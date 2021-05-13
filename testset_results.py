import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from spectra_utils import data_load_normalized


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

    return {'correct': correct, 'incorrect': incorrect, 'no_detction': no_detection, 'spectra': spectra}

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

    return results

def get_accuracy(results):

    total_correct = 0
    num = 0

    for res in results['testsets']:
        total_correct += res['correct']
        num += res['correct'] + res['incorrect'] + res['no_detection']

    return total_correct, num

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

    print(f'GS-DnCNN accuracy for entire test is {results_model["total_correct"]/results_model["num"]:.4f}')
    print(f'SSLCA accuracy without background subtraction for entire test is {results_SSLCA["total_correct"]/results_SSLCA["num"]:.4f}')
    print(f'SSLCA accuracy with background subtraction for entire test is {results_SSLCA_BS["total_correct"]/results_SSLCA_BS["num"]:.4f}')

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
