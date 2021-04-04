import os
import re
import sys
import json
import scipy.stats
import functools
import numpy as np

from compton import compton_continuum

# incase of no graphical display (e.g. SSH)
import matplotlib
if not os.environ.get('DISPLAY', '').strip():
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def data_load_normalized(fname, normalize=True, recalibrate=False):
    """Loads the .json file specified and returns (keV, hits), where hits
    has been normalized by the measurement time.
    """
    try:
        data = json.load(open(fname))

        # get keV values
        if recalibrate:
            e0, e1 = data['ENER_FIT'][0], data['ENER_FIT'][1]
            keV = np.arange(len(data['HIT'])) * float(e1) + float(e0)
        else:
            keV = np.asarray(data['KEV'])

        # get intensity values
        hits = np.asarray(data['HIT']).astype(float)

        # time normalize
        if normalize:
            if 'MEAS_TIM' not in data:
                print(f"no MEAS_TIME in {fname}")
            else:
                hits = hits / float(data['MEAS_TIM'].split(' ')[0])
    except Exception as e:
        raise ValueError(f'While loading {fname}') from e

    return keV, hits

def load_nndc_tables(nndc_dir, radionuclides):

    nndc_tables = {}

    for rn in radionuclides:
        keV, intensity = load_radionuclide_nndc(nndc_dir, rn)
        nndc_tables[rn] = {"keV": keV, "intensity": intensity}
        
    return nndc_tables

def split_radionuclide_name(rn_name):
    # split numeric prefix from radionuclide name
    num_letter = re.compile("([0-9]+)([a-zA-Z]+)") 
    return num_letter.match(rn_name).groups() 

def load_radionuclide_nndc(root, rn):

    rn_num, rn_name = split_radionuclide_name(rn)

    # build path for location of NNDC values for radionuclide
    loc = os.path.join(root, rn_num, rn_name + '.json')

    # load radionuclide NNDC table values
    with open(loc, 'r') as rn_nndc_file:
        radionuclide = json.load(rn_nndc_file)

    return radionuclide['keV'], radionuclide['intensity']


def rn_table_to_spec(spectrum_keV, peak_keVs, peak_intensities, bucket_size, max_keV):

    spectrum = np.zeros_like(spectrum_keV, dtype=np.float32)

    # load peak values into spectrum
    for k, i in zip(peak_keVs, peak_intensities):

        # check bounds 
        if k >= max_keV + bucket_size:
            continue

        ki = np.searchsorted(spectrum_keV, k, side='right')-1
        ki = min(ki, spectrum_keV.shape[0]-1)
        spectrum[ki] += i

    return spectrum

def generate_compton(spectrum, spectrum_keV, peak_keVs, peak_intensities, max_keV, compton_scale, det_material):

    compton = np.zeros_like(spectrum_keV, dtype=np.float32)

    for ke, i in zip(peak_keVs, peak_intensities):
        # don't add compton to templates for PE above keV vals in spectrum or below min efficiency
        if (ke >= max_keV):
            continue
        # find new intensity given detecotr efficiency
        ki = np.searchsorted(spectrum_keV, ke, side='right')-1
        ki = min(ki, spectrum_keV.shape[0]-1)
        i = spectrum[ki]
        compton += compton_continuum(ke, i, spectrum_keV, spectrum, compton_scale, det_material)

    return compton

def generate_clean_spectra(rn_table, config, bucket_size, max_keV, spectrum_keV):

    assert len(rn_table["keV"]) == len(rn_table["intensity"]), "Mismatch in number of energy and intensity values!"

    # account for detector efficiency
    peak_intensities = apply_efficiency_curve(rn_table["keV"], rn_table["intensity"], config['EFFICIENCY'])

    # convert NNDC radionuclide table features to spectrum
    spectrum = rn_table_to_spec(spectrum_keV, rn_table["keV"], rn_table["intensity"], bucket_size, max_keV)

    # widen peaks to match detector resolution 
    spectrum = data_smooth(spectrum_keV, spectrum, **config['SMOOTH_PARAMS'])

    return spectrum, peak_intensities

def generate_spectrum_SNR(spectrum, background, compton, snr):
    
    # normalize spectrum vector by its magnitude
    spectrum = spectrum / np.sqrt(np.sum(spectrum**2))

    # normalize compton vector by spectrum magnitude to maintain proportion
    #compton = compton / np.sqrt(np.sum(spectrum**2))

    # normalize background vector by its magnitude
    background = background / np.sqrt(np.sum(background**2))

    # calculate RMS of signal, noise and Compton
    sRMS = np.sqrt(np.mean(spectrum**2))
    nRMS = np.sqrt(np.mean(background**2))
    cRMS = np.sqrt(np.mean(compton**2))

    #print(f"SNR background: {20*np.log10(sRMS/nRMS)}")
    #print(f"SNR Compton: {20*np.log10(sRMS/cRMS)}")

    # SNR in linear terms
    snr_lin = np.power(10, snr/20.0)

    # calculate scalar value on noise to combine sources to match desired SNR
    if sRMS > 0 and nRMS > 0:
        noise_scale = (sRMS+cRMS)/(nRMS*snr_lin)
        background *= noise_scale

        #nRMS = np.sqrt(np.mean(background**2))
        #print(f"SNR adjusted background: {20*np.log10(sRMS/nRMS):.2f}")

    # combine background and compton as noise source
    noise = background + compton

    # combine clean spectrum and noise sources
    noise_spec = spectrum + noise
    #noise_spec = noise_spec / np.sqrt(np.sum(noise_spec**2))

    # normalize clean spectrum vector by its magnitude
    #spectrum = spectrum / np.sqrt(np.sum(spectrum**2))

    return spectrum, noise_spec, noise

def generate_spectrum(rn_table, config, compton_scale=0.0, min_efficiency=50, alpha=0.0035, noise_scale=0.0):

    # create data structure and stats for spectram
    min_keV = config["ENER_FIT"][0]
    bucket_size = config["ENER_FIT"][1]
    max_keV = bucket_size * (config["NUM_CHANNELS"]-1) + min_keV
    spectrum_keV = np.linspace(min_keV, max_keV, config["NUM_CHANNELS"])
    spectrum = np.zeros_like(spectrum_keV)
    compton = np.zeros_like(spectrum_keV)
    noise = np.zeros_like(spectrum_keV)

    assert len(rn_table["keV"]) == len(rn_table["intensity"]), "Mismatch in number of energy and intensity values!"

    # account for detector efficiency
    peak_intensities = apply_efficiency_curve(rn_table["keV"], rn_table["intensity"], config['EFFICIENCY'])

    # load peak values into spectrum
    for k, i in zip(rn_table["keV"], peak_intensities):

        # check bounds 
        if (k < min_keV) or (k >= max_keV + bucket_size):
            continue

        ki = np.searchsorted(spectrum_keV, k, side='right')-1
        ki = min(ki, spectrum_keV.shape[0]-1)
        spectrum[ki] += i

    # normalize spectrum by RMS
    spectrum /= np.sqrt(np.sum(spectrum**2))

    # generate compton for each peak
    if compton_scale > 0:
       for ke, i in zip(rn_table["keV"], peak_intensities):
           # don't add compton to templates for PE above keV vals in spectrum or below min efficiency
           if (ke >= spectrum_keV[-1] + bucket_size) or (ke < min_efficiency):
               continue
           # find new intensity given detecotr efficiency
           ki = np.searchsorted(spectrum_keV, ke, side='right')-1
           ki = min(ki, spectrum_keV.shape[0]-1)
           i = spectrum[ki]
           compton += compton_continuum(ke, i, spectrum_keV, spectrum, compton_scale, config["ATOMIC_NUM"])

    smooth_spectrum = data_smooth(spectrum_keV, spectrum, **config['SMOOTH_PARAMS'])

    if noise_scale > 0:
        # generate exponential decay curve for noise distribution
        noise_slope = np.exp(-alpha*spectrum_keV)
        noise = np.random.poisson(noise_slope).astype(np.float32)

        # scale noise by detector efficency
        noise = np.clip(apply_efficiency_curve(spectrum_keV, noise, config['EFFICIENCY']),a_min=0.0,a_max=None)

        # normalize noise by RMS and scale
        noise /= np.sqrt(np.sum(noise**2))
        noise *= noise_scale

    # smooth spectrum with compton and noise
    noisy_spectrum = data_smooth(spectrum_keV, spectrum+compton+noise, **config['SMOOTH_PARAMS'])
    
    # noise only
    noise = data_smooth(spectrum_keV, compton+noise, **config['SMOOTH_PARAMS'])

    #compare_results(spectrum_keV, smooth_spectrum, noisy_spectrum, noise, 'tmp', 'test.png', show_plot=True)
    return spectrum_keV, smooth_spectrum, noisy_spectrum, noise


def apply_efficiency_curve(keV, intensity, eff_vals):

    assert len(keV) == len(intensity), "Mismatch in number of energy and intensity values!"

    eff_intensity = np.zeros_like(intensity, dtype=np.float32)

    for n, (k, i) in enumerate(zip(keV, intensity)):
        if 0 <= k < eff_vals['LOW']['MAX_KEV']:
            eff_intensity[n] = apply_efficiency(k, i, eff_vals['LOW'])
        elif eff_vals['LOW']['MAX_KEV'] <= k < eff_vals['HIGH']['MAX_KEV']:
            eff_intensity[n] = apply_efficiency(k, i, eff_vals['HIGH'])
        else:
            eff_intensity[n] = 0

    return np.clip(eff_intensity, a_min=0, a_max=None)


def apply_efficiency(keV, intensity, eff_vals):

    efficiency = 0
    for coeff, pwr in zip(eff_vals['COEFS'], eff_vals['POWERS']):
        efficiency += coeff*keV**pwr

    return intensity * efficiency


# The following smoothing code is
# CONTRIBUTED BY: Walt Woods

def data_smooth(kev, hits, k0=3.458, k1=0.28, k2=0):
    """Applies kernel smoothing to the given kev, hits records.  Assumes a
    normal distribution of error.

    Args:
        kev: Array of keV values.
        hits: Array of hits/s values.
        k0: Intercept.  See k2
        k1: Slope.
        k2: The width of the Gaussian kernel used for smoothing is
                k0 + k1 * keV ** 0.5 + k2 * keV.

                Based on Adam M's 152EU spectra, the following spreads can
                be seen (estimating 95% confidence, so values are two stds):
                    121.75 - 0.75
                    344.29 - 0.8
                    778.9 - 1.
                    1086 - 1.4
                    1408 - 1.5

                Fitting k0 + k1 * keV ** 0.5 + k2 * keV to these values with least squares
                (scipy.optimize.root(method='lm')) yields k0 and k1's defaults.

                Code:
                    scipy.optimize.root(
                        lambda a: abs(a[0])
                            + abs(a[1]) * np.asarray([121.75,344.29,778.9,1408])**0.5
                            + abs(a[2]) * np.asarray([121.75,344.29,778.9,1408])
                            - np.asarray([0.8,1.,1.2,1.755]),
                        [1,1,1], method='lm')
    """
    assert len(kev) == len(hits)
    smooth_hits = np.zeros_like(hits)

    keV_tuple = tuple(kev)
    # Use last to determine bucket size; first can be identical due to
    # zero-cropping...
    for i, (hkev, h) in enumerate(zip(kev, hits)):
        # Uses realistic variance based on 95% confidence intervals in measured
        # data (see k0 and k1).
        # Also assuming each keV is the center of bucket at the moment...
        # it's possible some of the channels represent negative keVs, ignore these
        if hkev < 0: continue
        p, left, right = _data_smooth_cdf_range(hkev, k0, k1, k2, keV_tuple)
        smooth_hits[left:right] += h * p[left:right]
    #if smooth_hits.sum() == 0:
    #    raise ValueError("HU")
    return smooth_hits

def data_smooth_get_std(kev, k0=3.458, k1=0.28, k2=0):
    """Returns the standard deviation for smoothing at kev.

    Assumes default parameters on data_smooth.
    """
    # 0.5 would be 99% confidence, want that divided by 3.
    return 0.2 * (k0 + k1 * kev ** 0.5 + k2 * kev)

@functools.lru_cache(maxsize=10000)
def _data_smooth_cdf_range(keV, k0, k1, k2, keV_spectrum):
    """It turns out scipy.stats.norm().cdf() is quite expensive when you
    call it a lot.  As such, this is a cached version.

    Returns array to multiply the hit count for the bucket keV by to get hit
    count distribution across keV_spectrum.

    Note that the math works as following:

        Events
        H = hit in keV bucket B
        Q = an emission from keV bucket A

        Want P(Q|H)
        Have P(H), P(H|Q)

        P(Q|H) = P(H|Q)P(Q) / P(H)
        P(Q|H) ~ P(H|Q)  (without better knowledge, P(Q), P(H) are uniform
    """
    hksp = 0.5 * (keV_spectrum[-1] - keV_spectrum[-2])
    std = data_smooth_get_std(keV, k0, k1, k2)
    kernel = scipy.stats.norm(keV, std)
    s = np.asarray(keV_spectrum)
    left = np.searchsorted(keV_spectrum, keV - 5 * std, 'left')
    right = np.searchsorted(keV_spectrum, keV + 5 * std, 'right')
    return kernel.cdf(s + hksp) - kernel.cdf(s - hksp), left, right
