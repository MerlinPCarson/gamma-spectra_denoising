import os
import sys
import time
import argparse
import pandas as pd

from subprocess import call
from compare_denoising import convert_map


def load_testset_locs(locs_file):

    df = pd.read_csv(locs_file)

    testset = {'directory': df['location'].tolist(), 'map-file': df['map-file'].tolist()}
    
    return testset

def get_results(file):

    with open(file,'r') as f:
        data = f.read()

    correct = data.count('Correct')
    incorrect = data.count('Incorrect')
    no_pred = data.count('No Detection')

    print(f'correct:{correct}, incorrect:{incorrect}, no detection:{no_pred}')

    return correct, correct+incorrect+no_pred

def parse_args():
    parser = argparse. ArgumentParser(description='Gamma-Spectra Denoising testing with SSLCA identification')
    parser.add_argument('--testset', type=str, default='spectra/NaI/testset_full.csv', help='csv file of testset spectra locations')
    parser.add_argument('--model', type=str, default='models/best_model.pt', help='location of model to use')
    parser.add_argument('--SSLCA', type=str, default='../DTRA_SSLCA/psu_dtra/SSLCA-RID.py', help='location of SSLCA script')
    parser.add_argument('--nndc_dir', type=str, default='nuclides-nndc', help='location of NNDC radionuclide gamma-ray tables')
    parser.add_argument('--smooth', help='smooth noisy spectra before denoising', default=False, action='store_true')

    return parser.parse_args()

def main(args):
    start = time.time()

    testset = load_testset_locs(args.testset)

    total_correct = 0
    total_num = 0
    for ts, mf in zip(testset['directory'], testset['map-file']):
        denoised_outdir = os.path.join(ts, 'denoised')
        os.makedirs(denoised_outdir, exist_ok=True)
        
        cmd = ['python', 'denoise_spectra.py', '--model', args.model, '--spectra', ts, '--outdir', denoised_outdir]
        if args.smooth:
            cmd.append('--smooth')

        call(cmd)
        denoised_mapfile = os.path.join(denoised_outdir, mf)
        convert_map(os.path.join(ts, mf), denoised_outdir)

        results_dir = os.path.join(os.path.dirname(args.model), 'results', os.path.basename(ts))
        os.makedirs(results_dir)
        log_file = os.path.join(results_dir, 'SSLCA-results')
        call(['python', args.SSLCA, 'mapped-build-and-apply', '--t-half', '-2', denoised_outdir , args.nndc_dir,
          denoised_mapfile, '--save-output', log_file])

        correct, total = get_results(log_file+'.txt')
        total_correct += correct
        total_num += total
        print(f'Accuracy for {ts} is {correct/total:.2f}')

    print(f'Accuracy for entire test is {total_correct/total_num:.2f}')
    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
