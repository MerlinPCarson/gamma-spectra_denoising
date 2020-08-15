import sys
import time
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr

from spectra_utils import plot_spectrum, compare_spectra
from load_templates import load_templates
import cvx


def main():
    start = time.time()

    data = h5py.File('data/training.h5', 'r')['HPGE']

    print(data.keys())
    names = data['name'][()].astype(np.str)
    noise_scales = data['noise_scale'][()].astype(np.float32)
    compton_scales = data['compton_scale'][()].astype(np.float32)
    training_data = np.stack((names, noise_scales, compton_scales), axis=1)

    x_train, x_val, _, _ = train_test_split(training_data, training_data, test_size = 0.1, random_state=42)

    rn = '152Eu'
    eu152 = np.where(x_val[:,0] == rn)[0]
    specs = x_val[eu152,:]

    max_noise_val = np.amax(specs[:,1].astype(np.float32))
    max_noise = np.argwhere(specs[:,1].astype(np.float32) == max_noise_val) 
    print(f"spectra with highest noise scale {max_noise_val}")
    print(specs[max_noise,:])

    max_compton_val = np.amax(specs[:,2].astype(np.float32))
    max_compton = np.argwhere(specs[:,2].astype(np.float32) == max_compton_val) 
    print(f"spectra with highest compton scale {max_compton_val}")
    print(specs[max_compton,:])

    #data_idx = np.where((training_data[:,0] == rn) & (training_data[:,1].astype(np.float32) == max_noise_val) & (training_data[:,2] == max_compton_val))
    data_idx = np.where((training_data[:,0] == rn) & (training_data[:,1] == str(max_noise_val)) & (training_data[:,2] == str(max_compton_val)))

    match_spec = { 'name' : data['name'][data_idx][0].decode('utf-8'), 'spectrum': data['spectrum'][data_idx][0], \
                   'noisy_spectrum': data['noisy_spectrum'][data_idx][0], 'keV': data['keV'][()]}

    plot_spectrum(match_spec['keV'], match_spec['spectrum'], match_spec['name'], 'tmp', title="Template", show_plot=True)
    plot_spectrum(match_spec['keV'], match_spec['noisy_spectrum'], match_spec['name'], 'tmp', title="Noisy", show_plot=True)

    # load bases, clean radionuclide templates
    templates = load_templates('data/templates.h5', 'HPGE')
    bases = np.array(templates['intensity'])

    # Eu152 template
    temp = bases[7]# / np.sqrt(np.sum(bases[7]**2))
    print(f'diff: {np.sum(np.abs(temp - match_spec["spectrum"]))}')

    #compare_spectra(match_spec['keV'], match_spec['spectrum'], temp, match_spec['name'], 'tmp', show_plot=True)
    #spectrum /= np.sqrt(np.sum(spectrum**2))

#   Test prediction on noisy data
    (ampl0, q) = cvx.decompose(bases, match_spec['spectrum'], False, 1)
    pred = np.argmax(ampl0)
    print(f"Pred is {templates['name'][pred]}, Target is {match_spec['name']} ({ampl0[pred]})")

    data = h5py.File('data/denoised.h5', 'r')['HPGE']
    print(data.keys())

    data_idx = np.where((data['name'][()].astype(np.str) == rn) & (data['compton_scale'][()].astype(np.float32) == max_compton_val) & \
                        (data['noise_scale'][()].astype(np.float32) == max_noise_val))
    
    match_spec_dn = { 'name' : data['name'][data_idx][0].decode('utf-8'), 'spectrum': data['spectrum'][data_idx][0], \
                   'noisy_spectrum': data['noisy_spectrum'][data_idx][0], 'keV': data['keV'][()]}

#   Test prediction on noisy data
    (ampl0, q) = cvx.decompose(bases, match_spec_dn['noisy_spectrum'], False, 1)
    pred = np.argmax(ampl0)
    print(f"Pred is {templates['name'][pred]}, Target is {match_spec_dn['name']} ({ampl0[pred]})")

    dn_psnr = psnr(match_spec['noisy_spectrum'], match_spec_dn['noisy_spectrum'])

    print(f'PSNR: {dn_psnr}')
    plot_spectrum(match_spec_dn['keV'], match_spec_dn['noisy_spectrum'], match_spec_dn['name'], 'tmp', title='Denoised', show_plot=True)

    compare_spectra(match_spec['keV'], match_spec['noisy_spectrum'], match_spec_dn['noisy_spectrum'], match_spec['name'], 'tmp', \
                    title1="Noisy", title2="Denoised", show_plot=True)

    compare_spectra(match_spec['keV'], match_spec['spectrum'], match_spec_dn['noisy_spectrum'], match_spec['name'], 'tmp', \
                    title1="Template", title2="Denoised", show_plot=True)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0
    
if __name__ == '__main__':
    sys.exit(main())
