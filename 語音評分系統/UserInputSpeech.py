import os
from sys import argv
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from python_speech_features import mfcc
from sklearn.metrics.pairwise import cosine_similarity
import pyaudio
import wave
import matplotlib.pyplot as plt
from scipy.optimize import fmin

def load_wav(file):
    """讀取.wav檔案並回傳取樣率和音訊資料"""
    rate, data = wav.read(file)
    print("Shape of data:", data.shape)
    print("Data type of data:", data.dtype)
    return rate, data

def load_reference_data(wav_filename):
    """載入參考語音數據"""
    # 讀取參考語音檔案
    rate_ref, data_ref = wav.read(wav_filename)

    # 計算參考語音的MFCC特徵
    reference_mfcc = calculate_mfcc(data_ref, rate_ref)
    
    return reference_mfcc

def calculate_mfcc(data, rate):
    """計算MFCC特徵"""
    return mfcc(data, rate, nfft=2048)


def normalize_features(features):
    """正規化特徵"""
    # 音量強度曲線
    def average_magnitude(frames):
        """計算音量強度曲線"""
        mag = np.abs(frames)
        return np.mean(mag, axis=1)

    # 基頻軌跡
def average_magnitude_difference(frames):
    """計算基頻軌跡"""
    # 計算相鄰帧之間的幅度變化
    magnitude_diff = np.abs(frames[:, 1:] - frames[:, :-1])
    # 取每個特徵維度的平均值
    avg_magnitude_diff = np.mean(magnitude_diff, axis=1)
    return avg_magnitude_diff


#     # CMS (Cepstral Mean Subtraction)
# def cepstral_mean_subtraction(features):
#     """CMS"""
#     # 計算每個特徵的平均值
#     means = np.mean(features, axis=0)
#     # 減去平均值
#     return features - means

#     # 正規化特徵
#     norm_features = np.zeros_like(features)
#     # 正規化音量強度曲線
#     norm_features[:, 0] = average_magnitude(features)
#     # 正規化基頻軌跡
#     norm_features[:, 1] = average_magnitude_difference(features)
#     # 正規化CMS
#     norm_features[:, 2:] = cepstral_mean_subtraction(features[:, 2:])

#     return norm_features

def similarity(input_features, reference_features):
    """計算輸入語音與參考語音之間的相似度"""
    # 如果輸入特徵是三維的，將其展平為二維
    if input_features.ndim == 3:
        input_features = input_features.reshape(input_features.shape[0], -1)
    return cosine_similarity(input_features, reference_features)

def record_audio(filename, duration=3):
    """錄製與標準語音相同長度的使用者輸入的語音"""
    CHUNK = 4096  # 進一步增加帧大小
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Start recording...")

    frames = []
    seconds = duration

    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recording stopped")

    stream.stop_stream()
    stream.close()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def voice_score(params, a1, a2, a3, b1, b2, b3, w1, w2, w3):
    """計算語音評分"""
    distances = params[0]
    score = w1 * (100 / (1 + a1 * distances[0] ** b1)) + \
            w2 * (100 / (1 + a2 * distances[1] ** b2)) + \
            w3 * (100 / (1 + a3 * distances[2] ** b3))
    return score

def main():
    # 標準語音檔案路徑
    wav_filename = r"C:\Users\USER\Desktop\A.wav"

    # 讀取標準語音檔案並計算MFCC特徵
    reference_mfcc = load_reference_data(wav_filename)

    # 錄製與標準語音相同長度的使用者輸入語音
    user_input_filename = "user_input.wav"
    record_audio(user_input_filename)

    # 讀取使用者輸入的語音檔案並計算MFCC特徵
    rate_user, data_user = load_wav(user_input_filename)
    input_mfcc = calculate_mfcc(data_user, rate_user)

    # 計算使用者輸入語音與標準語音的相似度
    sim_scores = similarity(input_mfcc, reference_mfcc)

    # 將相似度矩陣轉換為距離列表
    distances = [1 - sim for sim in sim_scores.flatten()]

    # 正規化特徵
    norm_input_mfcc = normalize_features(input_mfcc)
    norm_reference_mfcc = normalize_features(reference_mfcc)


    # 計算語音評分
    params = (distances, a1, a2, a3, b1, b2, b3, w1, w2, w3)
    score = voice_score(params, a1, a2, a3, b1, b2, b3, w1, w2, w3)

    # 將相似度比對結果繪製成圖像
    plt.imshow(sim_scores, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Similarity')
    plt.xlabel('Standard Speech')
    plt.ylabel('User Input')
    plt.title('Similarity Matrix')
    plt.show()


if __name__ == "__main__":
    main()