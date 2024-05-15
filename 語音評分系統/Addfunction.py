import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from python_speech_features import mfcc
from scipy.ndimage import maximum_filter1d
from sklearn.metrics.pairwise import cosine_similarity
import pyaudio
import wave
import matplotlib.pyplot as plt
import librosa

def load_wav(file):
    '''
    读取一个wav文件，返回声音信号的时域谱矩阵、播放时间、通道数和帧数
    '''
    load_wav_file = wave.open(file, "rb")  # 開啟wav檔
    load_num_frames = load_wav_file.getnframes()  # 音框
    load_num_channels = load_wav_file.getnchannels()  # 通道数
    load_framerate = load_wav_file.getframerate()  # 音框赫茲數
    load_num_sample_width = load_wav_file.getsampwidth()  # 得到實際的bit寬度，即每一幀率的字节数
    
    str_data = load_wav_file.readframes(load_num_frames)  # 讀取全部的幀
    load_wav_file.close()  # 關閉
    load_wave_data = np.frombuffer(str_data, dtype=np.int16)  # 将声音文件数据转换为数组矩阵形式
    load_wave_data = load_wave_data.reshape(-1, load_num_channels)  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    return load_wave_data, load_framerate, load_num_channels, load_num_frames# 音框

def record_audio(filename):
    """录制与标准语音相同长度的用户输入的语音"""
    load_wave_data, load_framerate, load_num_channels, load_num_frames = load_wav(filename)  # 将文件名传递给 load_wav 函数
    record_audio_frames = load_num_frames  # 音框
    record_FORMAT = pyaudio.paInt16
    record_num_channels = load_num_channels #通道數
    record_RATE = load_framerate # 音檔赫茲數

    p = pyaudio.PyAudio()
    stream = p.open(format=record_FORMAT, channels=record_num_channels, rate=record_RATE, input=True, frames_per_buffer=record_audio_frames)
    print("Start recording...")

    record_num_frames = []
    seconds = 4  # 录制4秒，可以根据需求调整

    for i in range(0, int(record_RATE / record_audio_frames * seconds)):
        record_wave_data = stream.read(record_audio_frames)
        record_num_frames.append(record_wave_data)
    print("Recording stopped")

    stream.stop_stream()
    stream.close()

    wf = wave.open(filename, 'wb')
    record_num_channels=wf.setnchannels(record_num_channels)
    record_num_sample_width=wf.setsampwidth(p.get_sample_size(record_FORMAT))
    record_framerate=wf.setframerate(record_RATE)
    wf.writeframes(b''.join(record_num_frames))
    wf.close()
    return record_wave_data, record_framerate, record_num_channels, record_num_frames# 音框


def endpoint_detection(data_signal, frame_size, overlap, framerate):  
    """端点检测"""
    energy = np.sum(data_signal ** 2, axis=1)  # 沿着第二个维度求和
    threshold = np.mean(energy) * 1.5  # 设置能量阈值
    endpoints = np.where(energy > threshold)[0]  # 找到超过阈值的帧索引
    start = endpoints[0] - int(frame_size / 2)  # 起始点为第一个超过阈值的帧的前一半帧
    end = endpoints[-1] + int(frame_size / 2)  # 终点为最后一个超过阈值的帧的后一半帧
    return start, end

def calculate_volume_curve(signal, frame_size, overlap, framerate):  
    """计算音量强度曲线"""
    print("Signal shape:", signal.shape)  # 添加调试语句
    start, end = endpoint_detection(signal, frame_size, overlap, framerate)  # 传递帧率参数
    frames = signal[start:end]
    # 在这里计算音量强度曲线，假设计算结果存储在 mag_curve 变量中
    # 计算能量
    energy = librosa.feature.rms(y=signal, frame_length=frame_size, hop_length=frame_size - overlap)[0]
    mag_curve = energy  # 这里填入计算音量强度曲线的代码
    return mag_curve

# def plot_volume_curve(signal, rate, frame_size, overlap):
#     """绘制音量强度曲线"""
#     mag_curve = calculate_volume_curve(signal, frame_size, overlap, rate)  # 传递帧率参数
#     time = np.arange(len(mag_curve)) * (frame_size * (1 - overlap)) / rate
#     plt.plot(time, mag_curve)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Average Magnitude')
#     plt.title('Volume Intensity Curve')
#     plt.show()

# def plot_volume_curve(signal, framerate, frame_size, overlap):
#     """绘制音量强度曲线"""
#     time = librosa.frames_to_time(range(len(signal)), sr=framerate, hop_length=frame_size - overlap)
#     mag_curve = calculate_volume_curve(signal, framerate, frame_size, overlap)
#     plt.plot(time, mag_curve)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Average Magnitude')
#     plt.title('Volume Intensity Curve')
#     plt.show()

def calculate_mfcc(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs

# 正規化 MFCC 特徵
def normalize_mfcc(mfccs):
    return (mfccs - np.mean(mfccs)) / np.std(mfccs)

# 計算相似度分數
def compute_similarity_score(mfccs_A, mfccs_B):
    return np.mean(np.abs(mfccs_A - mfccs_B)) * 100

# 去除靜音部分
def remove_silence(audio, threshold=0.02):
    non_silent_intervals = librosa.effects.split(audio, top_db=threshold)
    non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
    return non_silent_audio

# 預處理雜音
def preprocess_audio(audio):
    # 雜音預處理，例如濾波等
    # 這裡我們使用 maximum_filter1d 函數來濾波
    filtered_audio = maximum_filter1d(audio, size=3)
    return filtered_audio

def main():
    # 錄製語音
    user_input_filename = 'D:/wsad231466/user_input.wav'
    user_wave_data, user_framerate, user_num_channels, user_num_frames= record_audio(user_input_filename)
    user_mfcc = calculate_mfcc(user_wave_data, user_framerate)

    # 標準語音路徑
    wav_filename = 'D:/wsad2314666/Alcoho.github.io/語音評分系統/A.wav'
    # 讀取標準語音檔案並計算MFCC特徵
    load_wave_data, load_framerate, load_num_channels, load_num_frames = load_wav(wav_filename)
    load_mfcc = calculate_mfcc(load_wave_data, load_framerate)
    # 打印通道数
    print("Number of channels:\n", load_num_channels)
    print("Number of wave_data:\n", load_wave_data)
    print("Number of framerate:\n", load_framerate)
    print("Number of num_frames:\n", load_num_frames)
    # 計算使用者輸入語音與標準語音的相似度
    sim_scores = compute_similarity_score(load_mfcc, user_mfcc)
    print("使用者輸入語音與標準語音的相似度比對結果：", sim_scores)
    # 设置音框大小和重叠参数
    frame_size = 512
    overlap = 170
    #plot_volume_curve(wave_data, framerate, frame_size, overlap)
    #plot_volume_curve(wave_data[:, 0], framerate, frame_size, overlap)
if __name__ == "__main__":
    main()