import os
import sys
import time
import pandas as pd
import numpy as np
import json
import re

from spectra_utils import split_radionuclide_name

def get_rn_name(spec_file):
    basename = os.path.basename(spec_file)
    rn_name = basename.split('_')[0]
    return split_radionuclide_name(rn_name)

def spec_to_json(A, B, keV, hits, meas_time, out_file):

    rn_num, rn_letters = get_rn_name(out_file)

    jsonData = {'RADIONUCLIDE': f'{rn_num}{rn_letters}',
                'MEAS_TIM': f'{meas_time} {meas_time}', 
                'ENER_FIT': [str(A), str(B)],
                'HIT': hits, 'KEV': keV}

    with open(out_file, 'w') as fp:
        print(f'Writing spectrum to {out_file}')
        json.dump(jsonData, fp)


def main():
    start = time.time()

    csv_file = sys.argv[1]
    #csv_file = 'data/mcnp_spectra/spectra/1000000counts.csv'

    df = pd.read_csv(csv_file)

    # save spectra names to list 
    spectra = df.columns[1:].tolist()
    print(f'Spectra names: {spectra}')
    print(df)

    A = df['E(keV)'][0]
    B = df['E(keV)'][1] - df['E(keV)'][0]

    # recalc energies from calibration
    #keV = [B*x+A for x in range(len(df['E(keV)']))] 

    # use calibrations from spreadsheet
    keV = df['E(keV)'].tolist()

    # verifiying calibration and bins line up
    #print(keV[34])
    #print(np.searchsorted(keV, 0.0, side='left'))
    #print(keV[70])
    #print(np.searchsorted(keV, 13.0, side='left'))
    #print(keV[124])
    #print(np.searchsorted(keV, 32.194, side='left'))
    #print(keV[1006])
    #print(np.searchsorted(keV, 347.14, side='left'))
    #print(keV[1886])
    #print(np.searchsorted(keV, 661.657, side='left'))
    #print(keV[3318])
    #print(np.searchsorted(keV, 1173.2, side='left'))

    for spectrum in spectra:
        meas_time = re.findall(r'\d*\.?\d+[s]', spectrum)
        if len(meas_time) == 1:
            meas_time = meas_time[0].replace('s','')
        else:
            print(f'Unable to determine measurement time from title {spectrum}')
            continue
        #meas_time = 1

        out_file = f"{os.path.join(os.path.dirname(csv_file), spectrum.replace(' ',''))}.json"
        spec_to_json(A, B, keV, df[spectrum].tolist(), meas_time, out_file)


    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
