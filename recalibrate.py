import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum(keV1, spec1, keV2, spec2):
    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(keV1, spec1, label='Original Spectrum')
    ax1.legend()

    ax2.plot(keV2, spec2, label='Recalibrated Spectrum')
    ax2.legend()

    plt.show()

def load_spec_json(fname):
    """Loads Gamma-Spectrum from .json file specified and returns (keV, hits, measurement time)
    """

    try:
        data = json.load(open(fname))
        e0, e1 = data['ENER_FIT'][0], data['ENER_FIT'][1]
        keV = np.arange(len(data['HIT'])) * float(e1) + float(e0)
        hits = np.asarray(data['HIT']).astype(float)

        assert 'MEAS_TIM' in data, 'Measurement time not found in spectrum file'
        measurement_time = float(data['MEAS_TIM'].split(' ')[0])

    except Exception as e:
        raise ValueError(f'While loading {fname}') from e

    return keV, hits, measurement_time 

def recalibrate(keV, hits, A, B, std):

    # create data structs for new calibration
    hits_recal = np.zeros(len(hits))
    keV_recal = np.arange(len(hits)) * B + A 

    # go through old hits array and rebin to new calibration
    for i in range(len(hits)):
        # find new bin to place channel's hits in
        ki = np.searchsorted(keV_recal, keV[i], side='right')-1
        ki = min(ki, keV_recal.shape[0]-1)

        # place hits in recalibrated bin
#        for _ in range(int(hits[i])):
#            sample = np.random.normal()
#            if -std <= sample <= std:
#                hits_recal[ki] += 1
#            elif sample < -std:  # shift one bin to the left if possible
#                new_ki = max(ki-1, 0)
#                hits_recal[new_ki] += 1
#            elif sample > std:  # shift one bin to the right if possible
#                new_ki = min(ki+1, len(hits_recal)-1)
#                hits_recal[new_ki] += 1

        hits_recal[ki] = hits[i]

    return keV_recal, hits_recal

def parse_args():
    parser = argparse.ArgumentParser(description='Gamma-Spectra Recalibration')
    parser.add_argument('--spec', type=str, default='background/NaI/BG1200s-Pu.json', help='Gamma-Spectra in .json format to recalibrate')
    parser.add_argument('--cal', type=str, default='-11.9922,0.35721000000000025', help='Comma seperated new calibration values (e.g. A,B)-->B*keV+A')
    parser.add_argument('--std', type=float, default=1.0, help='Number of standard deviations above or below mean to shift rebinned hit')
    args = parser.parse_args()

    assert os.path.isfile(args.spec), f'Spectrum file: {args.spec} not found' 

    cal = np.array(args.cal.split(','), dtype=np.float)
    assert len(cal) == 2, 'Incorrect number of calibration values'

    args.A, args.B = cal

    return args

def main(args):
    start = time.time()

    print(f'New calibration = {args.B}*keV + {args.A}')

    keV, hits, meas_time = load_spec_json(args.spec)

    keV_recal, hits_recal = recalibrate(keV, hits, args.A, args.B, args.std)

    print(f'num counts original: {np.sum(hits)}, num counts recalibrated: {np.sum(hits_recal)}')
    plot_spectrum(keV, hits, keV_recal, hits_recal)

    for hits in hits_recal[600:1400]:
        print(hits)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
