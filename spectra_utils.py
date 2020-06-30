import os
import re
import sys
import json
import scipy.stats
import functools
import numpy as np

import matplotlib.pyplot as plt


def load_radionuclide_nndc(root, rn):
    # split numeric prefix from radionuclide name
    num_letter = re.compile("([0-9]+)([a-zA-Z]+)") 
    rn_split = num_letter.match(rn).groups() 

    # build path for location of NNDC values for radionuclide
    loc = os.path.join(root, rn_split[0], rn_split[1] + '.json')

    # load radionuclide NNDC table values
    with open(loc, 'r') as rn_nndc_file:
        try:
            radionuclide = json.load(rn_nndc_file)
        except:
            sys.stderr.write(f"** Error loading NNDC table for {rn} in file {loc}\n")
            raise

    return radionuclide['keV'], radionuclide['intensity']


def generate_spectrum(rn_table, config):
    min_keV = config["ENER_FIT"][0]
    bucket_size = config["ENER_FIT"][1]
    max_keV = bucket_size * (config["NUM_CHANNELS"]-1) + min_keV
    keV = np.linspace(min_keV, max_keV, config["NUM_CHANNELS"])
    intensity = np.zeros(keV.shape)

    for k, i in zip(rn_table["keV"], rn_table["intensity"]):

        # check bounds 
        if (k < min_keV) or (k >= max_keV + bucket_size):
            continue

        ki = np.searchsorted(keV, k, side='right')-1
        ki = min(ki, keV.shape[0]-1)
        intensity[ki] += i

    # account for detector efficiency before adding compton
    #spectrum = apply_efficiency_curves(spectrum_keV, spectrum, det_type)

        
    intensity = data_smooth(keV, intensity, **config['SMOOTH_PARAMS'])

    return keV, intensity


def data_smooth_get_std(kev, k0=3.458, k1=0.28, k2=0):
    """Returns the standard deviation for smoothing at kev.

    Assumes default parameters on data_smooth.
    """
    # 0.5 would be 99% confidence, want that divided by 3.
    return 0.2 * (k0 + k1 * kev ** 0.5 + k2 * kev)


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