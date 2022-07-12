
import matplotlib.pyplot as plt
import mne
import HandyEEGpreset as preset

# keydict : dict타입이고, 여러가지 메타데이터를 가지고 있습니다.
# HandyEEGpreset.py 에서 가져옵니다.
keydict = preset.OPO()


# 이하 keydict에서 직접 조회하는 변수들
cmap = keydict['cmap']
ch_list = keydict['ch_list']
sfreq = keydict['sfreq']

# PSD : Power Spectrum Diagram 이라는 뜻 입니다.
# type : float으로 이루어진 list, 
# len : 19. keydict['ch_list'] 와 1:1 대응합니다.

PSD = [0.104, 0.120, 0.119, 0.115, 0.116, 0.118, 0.118, 0.219, 0.172, 
       0.172, 0.169, 0.172, 0.209, 0.168, 0.197, 0.199, 0.201, 0.160, 
       0.219] 

# vmax, vmin 은 이 값중 가장 큰 값입니다.
vmax = max(PSD)
vmin = min(PSD)

# info : mne에서 제공/요구하는 mne.info 인스턴스입니다.
# standard_1020, eeg 등 str 타입은 mne모듈에서 요구하는 바에 따릅니다.
info = mne.create_info(ch_names = ch_list,
                       sfreq = sfreq,
                       ch_types = 'eeg')
montage = mne.channels.make_standard_montage('standard_1020', 
                                             head_size =  0.1 )
info.set_montage(montage)

# plotting 하는 mne에서 제공하는 함수.
fig, ax = plt.subplots(figsize=(4, 4), gridspec_kw=dict(top=0.9),
                       sharex=True, sharey=True)

mne.viz.plot_topomap(PSD, info, cmap = cmap, names = ch_list, 
                     show_names = True, axes=ax,show = False, 
                     vmax = vmax, vmin = vmin)

# fig.show()

plt.savefig('fig1.png', dpi=300)