import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pyloudnorm as pyln
import soundfile as sf
import glob
import tqdm
from scipy.spatial.distance import euclidean
#讀音訊檔案
audio_path1="C:\\Users\\USER\\Desktop\\A.wav"
audio_path2="C:\\Users\\USER\\Desktop\\output.wav"

#載入兩個音檔，一個標準一個測試
y1,sr1=librosa.load(audio_path1)
y2,sr2=librosa.load(audio_path2)

#1.前處理
Audio_data1=librosa.effects.trim(audio_path1)
Audio_data2=librosa.effects.trim(audio_path2)

#兩個音檔的特徵

sample_rate_audio1=librosa.get_samplerate
trim_db=librosa.amplitude_to_db
# 提取 Mel 频谱图特征向量
mel1 = librosa.feature.melspectrogram(y=y1, sr=sr1)
mel2 = librosa.feature.melspectrogram(y=y2, sr=sr2)

# # 计算欧几里得距离
# dist = euclidean(mel1.reshape(-1), mel2.reshape(-1))
# print('欧几里得距离为：', dist)

# # 计算余弦相似度
# similarity = librosa.core.cosine_similarity(mel1, mel2)
# print('余弦相似度为：', similarity)
#
#2.取音框
#3.AMDF演算(Average Magnitude Difference Function) 
#4.High clipping
#5. 找local minima及算出頻率
#正規化:
#解決特徵參數長短不一的問題：Interpolation
#解決麥克風差異性：Linear Scaling
#解決個人音高差異性：Linear Shifting
#解決未知的通道效應：Cepstral Mean Subtraction

# 畫出波形圖1
plt.figure(figsize=(10, 4))
plt.plot(y1)
plt.title('Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# 計算音訊的STFT_1
D1 = librosa.stft(y1)

# 畫出頻譜圖1
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(D1, ref=np.max), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

# 畫出波形圖2
plt.figure(figsize=(10, 4))
plt.plot(y2)
plt.title('Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# 計算音訊的STFT_2
D2 = librosa.stft(y2)

# 畫出頻譜圖2
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(D2, ref=np.max), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

#擷取音訊特徵
feature1 = librosa.feature.mfcc(y=y1, sr=sr1)
feature2 = librosa.feature.mfcc(y=y2, sr=sr2)

# 繪製 MFCC1 特徵
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.feature.mfcc(y=y1, sr=sr1), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.tight_layout()
plt.show()

# 繪製 MFCC2 特徵
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.feature.mfcc(y=y2, sr=sr2), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.tight_layout()
plt.show()