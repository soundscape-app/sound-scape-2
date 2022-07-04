import numpy as np
from .level_calculation import A_weighting, C_weighting, Leq
from typing import *
from .freq_separation import get_fft_and_freq, get_split_ranges, get_filtered_audio, ONE_OCTAVE_BANDS


# raw data를 받아서 LAeq, LCeq, Leq, LCeq-LAeq를 계산해주는 헬퍼함수
def calculate_levels(wav: np.array, sr: int) -> Dict:
    Leq_ = Leq(wav, sr)

    x, sr = A_weighting(wav, sr)
    LAeq = Leq(x, sr)

    x, sr = C_weighting(wav, sr)
    LCeq = Leq(x, sr)

    return {
        'laeq': LAeq,
        'lceq': LCeq,
        'leq': Leq_,
        'lceq-laeq': LCeq - LAeq
    }


# raw data를 받아서 ONE_OCTAVE_BANDS 에 맞춰서 각 구간별 LAeq, LCeq, Leq, LCeq-LAeq를 계산해주는 헬퍼함수
def calculate_levels_by_frequencies(
    wav: np.array,
    sr: int
) -> Dict:
    fft, freq = get_fft_and_freq(wav, sr)
    ranges = get_split_ranges(freq, iter(ONE_OCTAVE_BANDS))

    result = {}
    for rng in ranges:
        filtered = get_filtered_audio(fft, rng)
        result[f'{freq[rng[0]]:.3f}-{freq[rng[1]-1]:.3f}'] = calculate_levels(filtered, sr)
    return result


if __name__ == "__main__":
    import librosa
    sig, sr = librosa.load('test-audio.wav', sr=None)
    print(calculate_levels_by_frequencies(sig, sr))

