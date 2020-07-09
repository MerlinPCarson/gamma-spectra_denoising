""" Compton.py
    Author: Merlin Carson
    Predicts the distribution for the compton scatter effect """

from math import cos, acos, sin, log, sqrt, radians, ceil, pi as PI
import numpy as np
import pickle

# Electron Constants for Klein-Nishina
BARNS = 1E+26
RESTEENERGY = 511       # Rest energy of electron (keV)
ERADIUS_SQR = 7.9406E-28    # Classical radius of an electron squared
FINITESTRUCT = 1/137.04
ALPHA_CUBED = FINITESTRUCT **4
COMPTONWAVELENGTH = 0.38616e-12 # pm 
TWOPI = 2 * PI
#KNCONSTANTS = TWOPI * 0.5*(FINITESTRUCT**2*COMPTONWAVELENGTH**2) * 1E+28
#KNCONSTANTS = TWOPI * (FINITESTRUCT**2*COMPTONWAVELENGTH**2) * 1E+28
KNCONSTANTS = 0.5 * 0.079408
DTCONSTANTS = TWOPI * RESTEENERGY
SQRT_TWO = sqrt(2)
TWOPI_R2_BARNS = TWOPI * ERADIUS_SQR * BARNS 
EIGHTPI_R2_BY_3 = 8 * PI * (ERADIUS_SQR * BARNS)/3
DETECTOR_EFFICIENCY_MIN_KEV = 50  # Per Adam Hecht e-mail, 2018-10-29


# calculate the energy ratio to actual energy to scattered energy
def energy_ratio(alpha,theta):
    return 1/(1+alpha*(1-cos(theta)))

# calculate klein nishina
def klein_nishina(theta, eRatio):
    return KNCONSTANTS * eRatio**2*(eRatio+1/eRatio-sin(theta)**2)

# differntiate electron energy in regards to angle
def diffEe(alpha,theta,keV):
    return ((1+alpha*(1-cos(theta)))**2*RESTEENERGY)/(keV**2)
    
# predict the compton scatter given an energy peak and it's counts
def compton_continuum(keV, counts, spectrum_keV, spectrum, compton_scale, det_material):
    # get amount of compton scattering
    bin_width = spectrum_keV[-1] - spectrum_keV[-2]
    CS_PE_ratio = cs_to_pe_ratio(keV, det_material)
    cs_counts = counts * CS_PE_ratio

    # ratio of energy peak to resting energy of an electron
    alpha = keV/RESTEENERGY

    # find highest keV of scattered energy
    compton_edge = keV-(keV*energy_ratio(alpha,PI))

    kns = []
    dSdT = []
    dSdE = []
    scatts = []
    keV_Te = 0.0
    while keV_Te <= compton_edge:
        ePrime = keV-keV_Te
        eRatio = ePrime/keV
        cos_theta = 1-(ePrime**(-1)-float(keV)**(-1))*RESTEENERGY
        theta = acos(cos_theta)

        kns.append(klein_nishina(theta, eRatio))
        dSdT.append(DTCONSTANTS * kns[-1] * ePrime**(-2))
        dSdE.append(det_material * dSdT[-1])
        scatts.append(keV_Te)

        keV_Te += bin_width

    compton_c = np.zeros_like(spectrum)

    total_area = sum(dSdE)
    for i, scatt in enumerate(scatts):
        ki = np.searchsorted(spectrum_keV, scatt, side='right')-1
        ki = min(ki, spectrum_keV.shape[0]-1)
        if ki < len(spectrum):
            compton_c[ki] += compton_scale * cs_counts * dSdE[i]/total_area#* dsigma[i]

    return compton_c

def cs_formula(keV, det_material):
    Emc2 = keV/RESTEENERGY
    A = (1+Emc2)/Emc2**2
    A1 = 2*(1+Emc2)/(1+2*Emc2)
    A2 = log(1+2*Emc2)/Emc2
    B = 0.5*A2
    C = (1+3*Emc2)/((1+2*Emc2)**2)
    CS_ratio = TWOPI_R2_BARNS * (A * (A1 - A2) + B - C)
    CS_barns = det_material * CS_ratio

    return CS_barns 

def pe_formula(keV, det_material):
    PE_formula = 4
    power_exp = 3
    z_fifth = det_material ** 5
    eg_power = (RESTEENERGY/keV)**power_exp
    PE_barns = PE_formula * ALPHA_CUBED * SQRT_TWO * z_fifth * EIGHTPI_R2_BY_3 * eg_power

    return PE_barns

def cs_to_pe_ratio(keV, det_material):
    CS_barns = cs_formula(keV, det_material)
    PE_barns = pe_formula(keV, det_material)
    CS_PE_ratio = CS_barns/PE_barns
    return CS_PE_ratio 
