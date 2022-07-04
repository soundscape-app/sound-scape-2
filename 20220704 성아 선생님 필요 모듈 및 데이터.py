# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:27:44 2022

@author: 하준혁(keita268@gmail.com)
"""
import sys # HandyEEGpreset.py가 있는 폴더를 지정해주시면 됩니다.
sys.path.append('D:/HJH/20220627 뇌파분석/Code/HandyEEG')
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
# 이후 모델에 의해 도출되야할 데이터이고, 
# 그 모델 설계나 이론에 대해서는 다시 말씀 드리겠습니다.
PSD = [0.104, 0.120, 0.119, 0.115, 0.116, 0.118, 0.118, 0.219, 0.172, 
       0.172, 0.169, 0.172, 0.209, 0.168, 0.197, 0.199, 0.201, 0.160, 
       0.219] 

# vmax, vmin 은 이 값중 가장 큰 값입니다.
vmax = max(PSD)
vmin = min(PSD)

# info : mne에서 제공/요구하는 mne.info 인스턴스입니다.
# standard_1020, eeg 등 str 타입은 mne모듈에서 요구하는 바에 따릅니다.
# 수정하실 필요 없이, 이대로면 됩니다.
info = mne.create_info(ch_names = ch_list,
                       sfreq = sfreq,
                       ch_types = 'eeg')
montage = mne.channels.make_standard_montage('standard_1020', 
                                             head_size =  0.1 )
info.set_montage(montage)

# plotting 하는 mne에서 제공하는 함수.
# 세이브하거나 플롯하는건 편하신대로 하시면 될 것 같습니다.
fig, ax = plt.subplots(figsize=(4, 4), gridspec_kw=dict(top=0.9),
                       sharex=True, sharey=True)

mne.viz.plot_topomap(PSD, info, cmap = cmap, names = ch_list, 
                     show_names = True, axes=ax,show = False, 
                     vmax = vmax, vmin = vmin)

fig.show()

# 아마도, 이미지는 세이브할 필요 없지만,
# PSD데이터는 저장할 필요가 있을 수도 있겠습니다.
# 어렵지 않다면, list형태 혹은 숫자19개를 저장할 수 있는 비슷한 형태로
# 설문 등과 같이 저장 될 수 있으면 좋겠습니다.
# 2일 이상 소요될 정도로 어렵다고 생각되시면 무시하셔도 좋습니다.