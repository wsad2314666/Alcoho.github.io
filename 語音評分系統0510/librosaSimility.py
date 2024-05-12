import os
import librosa # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import pyloudnorm as pyln
import soundfile as sf
import glob
import tqdm
from scipy.spatial.distance import euclidean
from scipy.signal import medfilt
#讀音訊檔案
audio_path1="D:\wsad2314666\Alcoho.github.io\語音評分系統0510\A.wav"
audio_path2="C:\\Users\\USER\\Desktop\\output.wav"

#載入兩個音檔，一個標準一個測試
y1,sr1=librosa.load(audio_path1)
y2,sr2=librosa.load(audio_path2)

#1.前處理
Effect_y1, _ = librosa.effects.trim(y1)
Effect_y2, _ = librosa.effects.trim(y2)


#兩個音檔的特徵
sample_rate_audio1=librosa.get_samplerate
trim_db=librosa.amplitude_to_db
# 提取 Mel 频谱图特征向量
mel1 = librosa.feature.melspectrogram(y=y1, sr=sr1)
mel2 = librosa.feature.melspectrogram(y=y2, sr=sr2)

def slience(mel,Audio_data,y,sr):
    frame_threshold=10#该参数决定去掉连续多少帧的静音段，比如某段语音检测到有12帧的静音帧，则去掉这一段的语音，而如果检测到只有8帧，那么不操作
    # 求取MFCCs参数
    #y, sr = librosa.load(audio_path1, sr=441000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24,win_length=1024,hop_length=512,n_fft=1024)

    # # 对mfcc进行中值滤波
    Mfcc1 = medfilt(mfccs[0, :], 9)
    pic = Mfcc1
    start = 0
    end = 0
    points = []
    min_data = min(pic) * 0.9

    for i in range((pic.shape[0])):
        if (pic[i] < min_data and start == 0):
            start = i
        if (pic[i] < min_data and start != 0):
            end = i

        elif (pic[i] > min_data and start != 0):
            hh = [start, end]
            points.append(hh)
            start = 0

    # 解决 文件的最后为静音
    if (pic[-1] < min_data and start != 0):
        hh = [start, end]
        points.append(hh)
        start = 0
    distances = []
    for i in range(len(points)):

        two_ends = points[i]
        distance = two_ends[1] - two_ends[0]
        if (distance > frame_threshold):
            distances.append(points[i])
    

    # 保存到本地文件夹
    name = Audio_data.split('\\')[-1]

    # 取出来端点，按照端点，进行切割,分情况讨论：1.如果未检测到静音段 2.检测到静音段

    if (len(distances) == 0):
        # print('检测到的静音段的个数为： %s 未对文件进行处理：' % len(distances))
        return y
        # sf.write(slience_clean, clean_data, 16000)

    else:
        slience_data = []
        for i in range(len(distances)):
            if (i == 0):
                start, end = distances[i]
                # 将左右端点转换到 采样点

                if (start == 1):
                    internal_clean = y[0:0]
                else:
                    # 求取开始帧的开头
                    start = (start - 1) * 512
                    # 求取结束帧的结尾
                    end = (end - 1) * 512 + 1024
                    internal_clean = y[0:start - 1]

            else:
                _, end = distances[i - 1]
                start, _ = distances[i]
                start = (start - 1) * 512
                end = (end - 1) * 512 + 1024
                internal_clean = y[end + 1:start]

            hhh = np.array(internal_clean)
            # 开始拼接
            slience_data.extend(internal_clean)

        # 开始 添加 最后一部分,需要分情况讨论，1. 文件末尾本来就是静音的  2.文件末尾不是静音的
        ll = len(distances)
        _, end = distances[ll - 1]
        end = (end - 1) * 512 + 1024
        end_part_clean = y[end:len(y)]
        slience_data.extend(end_part_clean)
        # 写到本地
        # sf.write("./data/{}.wav".format(name), slience_data, 16000)

        return slience_data
if __name__ == '__main__':  
      
	Audio_process1=slience(mel1,audio_path1,Effect_y1,sr=sr1)
    #Audio_process=slience(mel2,audio_path2,Effect_y2,sr=sr2)
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
plt.plot(Effect_y1)
plt.title('Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# 計算音訊的STFT_1
D1 = librosa.stft(Effect_y1)

# 畫出頻譜圖1
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(D1, ref=np.max), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

# 畫出波形圖2
plt.figure(figsize=(10, 4))
plt.plot(Effect_y2)
plt.title('Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()

# 計算音訊的STFT_2
D2 = librosa.stft(Effect_y1)

# 畫出頻譜圖2
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(D2, ref=np.max), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

#擷取音訊特徵
feature1 = librosa.feature.mfcc(y=Effect_y1, sr=sr1)
feature2 = librosa.feature.mfcc(y=Effect_y2, sr=sr2)

# 繪製 MFCC1 特徵
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.feature.mfcc(y=Effect_y1, sr=sr1), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.tight_layout()
plt.show()

# 繪製 MFCC2 特徵
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.feature.mfcc(y=Effect_y2, sr=sr2), y_axis='log', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.tight_layout()
plt.show()