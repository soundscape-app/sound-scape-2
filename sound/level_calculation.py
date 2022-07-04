import numpy as np
import math
from numpy import pi, convolve
from scipy.signal.filter_design import bilinear
from scipy.signal import lfilter
from numpy import polymul


def A_weighting(wav, sr):
    """Design of an A-weighting filter.

    B, A = A_weighting(Fs) designs a digital A-weighting filter for
    sampling frequency Fs. Usage: y = lfilter(B, A, x).
    Warning: Fs should normally be higher than 20 kHz. For example,
    Fs = 48000 yields a class 1-compliant filter.

    Originally a MATLAB script. Also included ASPEC, CDSGN, CSPEC.

    Author: Christophe Couvreur, Faculte Polytechnique de Mons (Belgium)
            couvreur@thor.fpms.ac.be
    Last modification: Aug. 20, 1997, 10:00am.

    http://www.mathworks.com/matlabcentral/fileexchange/69
    http://replaygain.hydrogenaudio.org/mfiles/adsgn.m
    Translated from adsgn.m to PyLab 2009-07-14 endolith@gmail.com

    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.

    """
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997
    NUMs = [(2 * pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    DENs = convolve([1, +4 * pi * f4, (2 * pi * f4) ** 2],
                    [1, +4 * pi * f1, (2 * pi * f1) ** 2], mode='full')
    DENs = convolve(convolve(DENs, [1, 2 * pi * f3], mode='full'),
                    [1, 2 * pi * f2], mode='full')
    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    B, A = bilinear(NUMs, DENs, sr)
    x = lfilter(B, A, wav)
    return x, sr


def C_weighting(wav, sr):
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f4 = 12194.217
    C1000 = 0.0619
    NUMs = [(2 * pi * f4) ** 2 * (10 ** (C1000 / 20.0)), 0, 0]
    DENs = polymul([1, 4 * pi * f4, (2 * pi * f4) ** 2.0], [1, 4 * pi * f1, (2 * pi * f1) ** 2])
    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    B, A = bilinear(NUMs, DENs, sr)
    x = lfilter(B, A, wav)
    return x, sr


def Leq(wav, sr):
    p0 = 2 * 0.00001
    T = 1 / sr
    wav_length = len(wav) / sr
    integral_result = np.trapz(wav ** 2, dx=T, )
    Leq = 10 * math.log10(wav_length * 1 / p0 ** 2 * integral_result)
    return Leq

