import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from python_speech_features import mfcc
from sklearn.metrics.pairwise import cosine_similarity
import pyaudio
import wave

def convert_to_wav(mp4_file):
    """將MP4格式的語音檔案轉換為.wav格式"""
    wav_file = mp4_file.replace('.mp4', '.wav')
    os.system(f"ffmpeg -i {mp4_file} {wav_file}")
    return wav_file

def load_wav(file):
    """讀取.wav檔案並回傳取樣率和音訊資料"""
    rate, data = wav.read(file)
    return rate, data

def calculate_mfcc(data, rate):
    """計算MFCC特徵"""
    return mfcc(data, rate)

def load_reference_data():
    """載入參考語音數據"""
    # 在這裡載入預先訓練好的語音數據，可以是一組.wav檔案，也可以是MFCC特徵向量的集合
    # 這裡只是一個示例，實際上需要根據你的需求來載入數據
    return np.random.rand(1, 13)  # 假設有1個語音樣本，每個樣本有13個MFCC特徵向量

def similarity(input_features, reference_features):
    """計算輸入語音與參考語音之間的相似度"""
    return cosine_similarity([input_features], reference_features)

def record_audio():
    """錄製使用者輸入的語音"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Start recording...")

    frames = []
    seconds = 3

    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recording stopped")

    stream.stop_stream()
    stream.close()

    wf = wave.open("user_input.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return "user_input.wav"

def main():
    # 上傳.mp4檔
    mp4_filename = input("請輸入.mp4檔案的路徑：")

    # 將MP4轉換為WAV
    wav_filename = convert_to_wav(mp4_filename)

    # 讀取.wav檔
    rate, data = load_wav(wav_filename)

    # 計算輸入語音的MFCC特徵
    input_mfcc = calculate_mfcc(data, rate)

    # 載入參考語音數據
    reference_data = load_reference_data()

    # 計算輸入語音與參考語音之間的相似度
    sim_scores = similarity(input_mfcc, reference_data)

    print("相似度比對結果：", sim_scores)

    # 錄製使用者輸入的語音
    user_audio_file = record_audio()

    # 讀取使用者輸入的語音檔案
    rate_user, data_user = load_wav(user_audio_file)

    # 計算使用者輸入語音的MFCC特徵
    input_mfcc_user = calculate_mfcc(data_user, rate_user)

    # 計算使用者輸入語音與參考語音之間的相似度
    sim_scores_user = similarity(input_mfcc_user, reference_data)

    print("使用者輸入語音與參考語音的相似度比對結果：", sim_scores_user)

if __name__ == "__main__":
    main()