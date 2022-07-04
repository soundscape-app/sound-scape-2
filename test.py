import librosa
import numpy as np
import soundfile as sf

sig, sr = librosa.load('test-audio.wav')

original = np.fft.rfft(sig)
freq = np.fft.rfftfreq(len(sig), d=1./sr)

freq_separators = [100, 1000, 3000, 10000]
bef = 0
ranges = []
for f in freq_separators:
    idx = np.argmin(np.abs(freq - f))
    ranges.append((bef, idx))
    bef = idx
ranges.append((bef, len(freq)))
print(ranges)

for rng in ranges:
    fft = np.array(original, copy=True)
    # [i*chunk_len, (i+1)*chunk_len) 영역 제외하고 모두 0으로 채움
    fft[range(0, rng[0])] = 0
    fft[range(rng[1], len(original))] = 0
    # 수정된 fft 결과를 다시 time 도메인으로 변환
    new_data = np.fft.irfft(fft)
    sf.write(f'out({freq[rng[0]]:.2f}-{freq[rng[1]-1]:.2f}).wav', new_data, sr, 'PCM_24')




