import os
import sys
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from spectra_utils import plot_data
from load_data_real import load_data
from build_dataset import save_dataset
from svd_denoise import Denoiser
from pca_denoise import Denoiser as PCA_Denoiser
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.decomposition import PCA
from scipy.signal import medfilt

MAX_KEV = 1450
MIN_KEV = 50
#def load_data(datafile, det, show_data=False):
#    with h5py.File(datafile, 'r') as h5f:
#        dataset = {"name": h5f[det]["name"][()], "keV": h5f[det]["keV"][()], "spectrum": h5f[det]["spectrum"][()], \
#                            "noisy_spectrum": h5f[det]["noisy_spectrum"][()], "noise": h5f[det]["noise"][()], \
#                            "compton_scale": h5f[det]["compton_scale"][()], "SNR": h5f[det]["SNR"][()]}
#    if show_data:
#        plot_data(dataset)
#
#    return dataset

def plot_denoised(keV, clean, noisy, denoised, type=''):
    plt.plot(keV, clean, label='Clean Spectrum')
    plt.plot(keV, noisy, label='Noisy Spectrum')
    plt.plot(keV, denoised, label=f'{type} Denoised Spectrum')
    plt.legend()
    plt.show()

def denoise_SVD(dataset, start=0.5, step=0.1):
    for i in range(len(dataset["name"])):
        spectrum = np.reshape(dataset["noisy_spectrum"][i], (1,-1))
        u, s, vh = np.linalg.svd(spectrum, full_matrices=False)
        threshold = start
        while (threshold < 1.0):
            #tmp = len(s)
            #new_s  = s[:]
            #for i in range (tmp):
            #    j = tmp - i - 1
            #    if (i >= tmp * threshold):
            #        break
            #    new_s[j] = 0.0
            denoised = np.array(np.dot(u * s, vh))
            plt.plot(dataset["keV"], dataset["noisy_spectrum"][i])
            plt.plot(dataset["keV"], denoised.reshape(-1))
            plt.show()

def denoise_medfilter(dataset, show_data=False):
    
    denoised_spectra = []
    denoised_psnr = 0.0
    windowsize = 61
    min_idx = np.searchsorted(dataset["keV"], MIN_KEV, side='right')-1
    max_idx = np.searchsorted(dataset["keV"], MAX_KEV, side='right')-1
    for i in tqdm(range(len(dataset["name"]))):
        background_est = medfilt(dataset["noisy_spectrum"][i], windowsize) * 0.5
        denoised = np.clip(dataset["noisy_spectrum"][i] - background_est, a_min=0.0, a_max=None).astype(np.float32)
        denoised_spectra.append(denoised)

        denoised_psnr += psnr(dataset["spectrum"][i][min_idx:max_idx], denoised[min_idx:max_idx])

        if show_data:
            plot_denoised(dataset["keV"], dataset["spectrum"][i], dataset["noisy_spectrum"][i], denoised, type='Median Filter')

    dataset["noisy_spectrum"] = np.array(denoised_spectra)
    outfile = os.path.join('real_models', 'medfilt_denoised.h5')
    print(f'Saving denoised spectrum to {outfile}')
    save_dataset('NAI', dataset, outfile)
    denoised_psnr /= len(dataset["name"])
    return denoised_psnr

def denoise_SVD2(dataset, show_data=False):
    
    denoiser = Denoiser()
    denoised_spectra = []
    denoised_psnr = 0.0
    min_idx = np.searchsorted(dataset["keV"], MIN_KEV, side='right')-1
    max_idx = np.searchsorted(dataset["keV"], MAX_KEV, side='right')-1
    for i in tqdm(range(len(dataset["name"]))):
        denoised = denoiser.denoise(dataset["noisy_spectrum"][i], 21)
        denoised_spectra.append(denoised)
        denoised_psnr += psnr(dataset["spectrum"][i][min_idx:max_idx], denoised[min_idx:max_idx])

        if show_data:
            plot_denoised(dataset["keV"][min_idx:max_idx], dataset["spectrum"][i][min_idx:max_idx], dataset["noisy_spectrum"][i][min_idx:max_idx], denoised[min_idx:max_idx], type='SVD')

    dataset["noisy_spectrum"] = np.array(denoised_spectra)
    outfile = os.path.join('real_models', 'svd_denoised.h5')
    print(f'Saving denoised spectrum to {outfile}')
    save_dataset('NAI', dataset, outfile)
    denoised_psnr /= len(dataset["name"])
    return denoised_psnr

def denoise_PCA(dataset, show_data=False):

    #pca = PCA(0.5)
    denoiser = PCA_Denoiser()
    denoised_spectra = []
    denoised_psnr = 0.0
    min_idx = np.searchsorted(dataset["keV"], MIN_KEV, side='right')-1
    max_idx = np.searchsorted(dataset["keV"], MAX_KEV, side='right')-1
    for i in tqdm(range(len(dataset["name"]))):
        denoised = denoiser.denoise(dataset["noisy_spectrum"][i], 7)
        #noisy_spec = dataset["noisy_spectrum"][i].reshape((1,-1))
        #pca = PCA(n_components=10).fit(noisy_spec)
        #components = pca.transform(noisy_spec)
        #denoised = np.squeeze(pca.inverse_transform(components))
        denoised_spectra.append(denoised)
        denoised_psnr += psnr(dataset["spectrum"][i][min_idx:max_idx], denoised[min_idx:max_idx])

        if show_data:
            plot_denoised(dataset["keV"], dataset["spectrum"][i], dataset["noisy_spectrum"][i], denoised, type='PCA')

    dataset["noisy_spectrum"] = np.array(denoised_spectra)
    outfile = os.path.join('real_models', 'pca_denoised.h5')
    print(f'Saving denoised spectrum to {outfile}')
    save_dataset('NAI', dataset, outfile)
    denoised_psnr /= len(dataset["name"])
    return denoised_psnr


def calc_snr_DNN(dataset, denoised):
    denoised_psnr = 0.0
    assert len(dataset["name"]) == len(denoised["name"]), "Mismatch in size of noisy and denoised spectrum"
    min_idx = np.searchsorted(dataset["keV"], MIN_KEV, side='right')-1
    max_idx = np.searchsorted(dataset["keV"], MAX_KEV, side='right')-1
    for i in range(len(dataset["name"])):
        denoised_psnr += psnr(dataset["spectrum"][i][min_idx:max_idx].astype(np.float32), denoised["noisy_spectrum"][i][min_idx:max_idx].astype(np.float32))

    denoised_psnr /= len(dataset["name"])
    return denoised_psnr

def clac_snr_dataset(dataset):
    denoised_psnr = 0.0
    min_idx = np.searchsorted(dataset["keV"], MIN_KEV, side='right')-1
    max_idx = np.searchsorted(dataset["keV"], MAX_KEV, side='right')-1
    for i in range(len(dataset["name"])):
        denoised_psnr += psnr(dataset["spectrum"][i][min_idx:max_idx].astype(np.float32), dataset["noisy_spectrum"][i][min_idx:max_idx].astype(np.float32))

    denoised_psnr /= len(dataset["name"])
    return denoised_psnr


def main():
    start = time.time()

    denoised_file = 'real_models/models_15_57.74/denoised.h5'
    train_file = 'real_models/models_15_57.74/training.h5'
    det_type = 'NAI'

    tr_data = load_data(train_file, det_type, show_data=False)
    psnr = clac_snr_dataset(tr_data)
    print(f'PSNR of dataset: {psnr}')

    # Median-Filter denoising 
    tr_data = load_data(train_file, det_type, show_data=False)
    psnr_medfilt = denoise_medfilter(tr_data, show_data=False)
    print(f'PSNR of Median Filter: {psnr_medfilt}')

    # SVD denoising with partial circulant matrix
    tr_data = load_data(train_file, det_type, show_data=False)
    psnr_SVD = denoise_SVD2(tr_data, show_data=False)
    print(f'PSNR of SVD: {psnr_SVD}')

    # PCA denoising
    tr_data = load_data(train_file, det_type, show_data=False)
    psnr_PCA = denoise_PCA(tr_data, show_data=False)
    print(f'PSNR of PCA: {psnr_PCA}')

    tr_data = load_data(train_file, det_type, show_data=False)
    dn_data = load_data(denoised_file, det_type, show_data=False)
    psnr_DNN = calc_snr_DNN(tr_data, dn_data)
    print(f'PSNR of DNN: {psnr_DNN}')

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
