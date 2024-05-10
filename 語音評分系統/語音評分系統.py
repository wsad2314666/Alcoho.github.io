import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from python_speech_features import mfcc
from sklearn.metrics.pairwise import cosine_similarity
import pyaudio
import wave
import matplotlib.pyplot as plt

def load_wav(file):
    """讀取.wav檔案並回傳取樣率和音訊資料"""
    rate, data = wav.read(file)
    return rate, data

def record_audio(filename):
    """錄製與標準語音相同長度的使用者輸入的語音"""
    CHUNK = 4096  # 進一步增加帧大小
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Start recording...")

    frames = []
    seconds = 4  # 錄製3秒，可以根據需求調整

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

def calculate_mfcc(data, rate):
    """計算MFCC特徵"""
    return mfcc(data, rate)

def load_reference_data(wav_filename):
    """載入參考語音數據"""
    # 讀取參考語音檔案
    rate_ref, data_ref = load_wav(wav_filename)

    # 計算參考語音的MFCC特徵
    reference_mfcc = calculate_mfcc(data_ref, rate_ref)
    
    return reference_mfcc

def similarity(input_features, reference_features):
    """計算輸入語音與參考語音之間的相似度"""
    # 如果輸入特徵是三維的，將其展平為二維
    if input_features.ndim == 3:
        input_features = input_features.reshape(input_features.shape[0], -1)
    return cosine_similarity(input_features, reference_features)


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

    print("使用者輸入語音與標準語音的相似度比對結果：", sim_scores)
    score=100/1+2000*(sim_scores)**3
    print("語音評分結果：", score)
    scores = np.sum(score) * 100 / len(score)
    # # 將相似度比對結果繪製成圖像
    # plt.imshow(score, cmap='viridis', interpolation='nearest')
    # plt.colorbar(label='Similarity')
    # plt.xlabel('Standard Speech')
    # plt.ylabel('User Input')
    # plt.title('Similarity Matrix')
    # plt.show()

    plt.plot(scores)
    plt.xlabel('Frame')
    plt.ylabel('Score')
    plt.title('Speech Similarity Score')
    plt.show()

if __name__ == "__main__":
    main()